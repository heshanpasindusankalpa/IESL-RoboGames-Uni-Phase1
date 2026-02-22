import math
import time
from dataclasses import dataclass
from pymavlink import mavutil

def wrap_pi(a: float) -> float:
    """Wrap angle to (−π, π]."""
    while a > math.pi:  a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a

def set_param(master, name: str, value: float):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode("utf-8"), float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )

def force_arm(master):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 21196, 0, 0, 0, 0, 0,
    )

def is_armed(master) -> bool:
    hb = master.recv_match(type="HEARTBEAT", blocking=False)
    if not hb:
        return False
    return bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

def takeoff(master, target_alt: float):
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, target_alt,
    )

def send_body_velocity(master, vx: float, vy: float, vz: float, yaw_rate: float):
    mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )
    master.mav.set_position_target_local_ned_send(
        0,
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        mask,
        0, 0, 0,
        vx, vy, vz,
        0, 0, 0,
        0, yaw_rate,
    )

@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    out_limit: float
    i_limit: float

    def __post_init__(self):
        self.i      = 0.0
        self.prev_e = 0.0

    def reset(self):
        self.i      = 0.0
        self.prev_e = 0.0

    def update(self, e: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        self.i = max(-self.i_limit, min(self.i_limit, self.i + e * dt))
        de = (e - self.prev_e) / dt
        self.prev_e = e
        u = self.kp * e + self.ki * self.i + self.kd * de
        return max(-self.out_limit, min(self.out_limit, u))


class LowPass:
    """1st-order IIR: y += α·(x − y),  α ∈ (0, 1)."""
    def __init__(self, alpha: float, init: float = 0.0):
        self.alpha  = float(alpha)
        self.y      = float(init)
        self.inited = False

    def reset(self, v: float = 0.0):
        self.y      = float(v)
        self.inited = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.inited:
            self.y      = x
            self.inited = True
            return self.y
        self.y += self.alpha * (x - self.y)
        return self.y


class SlewRateLimiter:
    """Clamp |dy/dt| ≤ rate_per_s."""
    def __init__(self, rate_per_s: float, init: float = 0.0):
        self.rate   = float(rate_per_s)
        self.y      = float(init)
        self.inited = False

    def reset(self, v: float = 0.0):
        self.y      = float(v)
        self.inited = False

    def seed(self, v: float):
        self.y      = float(v)
        self.inited = True

    def update(self, x: float, dt: float) -> float:
        x = float(x)
        if dt <= 0:
            return self.y
        if not self.inited:
            self.y      = x
            self.inited = True
            return self.y
        step  = self.rate * dt
        delta = max(-step, min(step, x - self.y))
        self.y += delta
        return self.y

class AltitudeTracker:
    def __init__(self, stable_needed: int = 6):
        self.stable_needed = stable_needed
        self.stable_count  = 0
        self.last_alt      = None

    def reset(self):
        self.stable_count = 0
        self.last_alt     = None

    def update(self, master, target_alt: float, tol: float):
        msg = master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if not msg:
            return False, self.last_alt
        self.last_alt = msg.relative_alt / 1000.0
        if self.last_alt >= (target_alt - tol):
            self.stable_count += 1
        else:
            self.stable_count = 0
        return self.stable_count >= self.stable_needed, self.last_alt

class LineFollower:
    # ── Tuning ────────────────────────────────────────────────────────────────
    FWD_V             = 0.5
    START_V           = 0.05
    ACCEL_MPS2        = 0.18
    YAW_ENABLE_V      = 0.45
    ANGLE_DEADBAND    = math.radians(4.0)
    PITCH_SOFT        = math.radians(8.0)
    PITCH_HARD        = math.radians(15.0)
    PITCH_RATE_SOFT   = math.radians(20.0)
    MAX_LAT_V         = 0.6
    MAX_YAW_RATE      = 0.8
    CMD_HZ            = 20.0

    def __init__(self, yaw_sign: float = 1.0, tuning: dict | None = None):
        assert yaw_sign in (1.0, -1.0)
        self.yaw_sign = yaw_sign
        self.cmd_dt   = 1.0 / self.CMD_HZ

        t = tuning or {}

        self._lat_pid = PID(kp=t.get("lat_kp",  0.9),
                            ki=t.get("lat_ki",  0.0),
                            kd=t.get("lat_kd",  0.15),
                            out_limit=self.MAX_LAT_V, i_limit=0.5)
        self._yaw_pid = PID(kp=t.get("yaw_kp",  1.8),
                            ki=t.get("yaw_ki",  0.0),
                            kd=t.get("yaw_kd",  0.25),
                            out_limit=self.MAX_YAW_RATE, i_limit=0.8)
        self._off_lpf = LowPass(alpha=t.get("off_alpha", 0.25))
        self._ang_lpf = LowPass(alpha=t.get("ang_alpha", 0.25))
        self._vx_slew = SlewRateLimiter(rate_per_s=t.get("vx_slew", 0.45))
        self._vy_slew = SlewRateLimiter(rate_per_s=t.get("vy_slew", 1.2))
        self._yr_slew = SlewRateLimiter(rate_per_s=t.get("yr_slew", 2.0))

        self._tuning = t

        self._last_cmd_t  = 0.0
        self._started_at  = None
        self.last_vx      = 0.0
        self.last_vy      = 0.0

    def reset(self):
        self._lat_pid.reset()
        self._yaw_pid.reset()
        self._off_lpf.reset(0.0)
        self._ang_lpf.reset(0.0)
        self._vx_slew.reset(0.0)
        self._vy_slew.reset(0.0)
        self._yr_slew.reset(0.0)
        self._last_cmd_t = 0.0
        self._started_at = time.time()
        self.last_vx     = 0.0
        self.last_vy     = 0.0

    def update(self, master, line_m: dict,
               img_w: int,
               att_pitch=None, att_pitch_rate=None) -> bool:
        now = time.time()
        dt  = now - self._last_cmd_t if self._last_cmd_t > 0 else self.cmd_dt
        if (now - self._last_cmd_t) < self.cmd_dt:
            return False
        self._last_cmd_t = now

        elapsed    = (now - self._started_at) if self._started_at else 0.0
        ramp_v_max = min(self.FWD_V,
                         self.START_V + self.ACCEL_MPS2 * max(0.0, elapsed))

        if not line_m.get("found", False):
            vx_cmd  = min(0.2, ramp_v_max)
            vy_raw  = 0.0
            yr_raw  = 0.0 if ramp_v_max < self.YAW_ENABLE_V else 0.2
        else:
            offset_px  = line_m["cx"] - (img_w / 2.0)
            offset_n   = offset_px / (img_w / 2.0)
            ang_err    = float(line_m["angle_error_rad"])

            offset_n_f = self._off_lpf.update(offset_n)
            ang_err_f  = self._ang_lpf.update(ang_err)

            vy_raw = self._lat_pid.update(offset_n_f, dt)

            angle_db = self._tuning.get("angle_deadband", self.ANGLE_DEADBAND)
            if abs(ang_err_f) < angle_db:
                ang_err_f = 0.0
            if ramp_v_max < self.YAW_ENABLE_V:
                yr_raw = 0.0
                self._yaw_pid.reset()
            else:
                yr_raw = self.yaw_sign * (-self._yaw_pid.update(ang_err_f, dt))

            err_mag = min(1.0, abs(offset_n_f) + 0.6 * abs(ang_err_f))
            vx_cmd  = max(0.25, self.FWD_V * (1.0 - 0.5 * err_mag))
            vx_cmd  = min(vx_cmd, ramp_v_max)

        if att_pitch is not None:
            p_abs = abs(att_pitch)
            if p_abs >= self.PITCH_HARD:
                vx_cmd = min(vx_cmd, 0.10)
            elif p_abs > self.PITCH_SOFT:
                frac    = (p_abs - self.PITCH_SOFT) / (self.PITCH_HARD - self.PITCH_SOFT)
                vx_cmd *= max(0.2, 1.0 - frac)
        if att_pitch_rate is not None and abs(att_pitch_rate) > self.PITCH_RATE_SOFT:
            vx_cmd *= 0.7

        vx_out = self._vx_slew.update(vx_cmd, dt)
        vy_out = self._vy_slew.update(vy_raw,  dt)
        yr_out = self._yr_slew.update(yr_raw,  dt)

        send_body_velocity(master, vx_out, vy_out, 0.0, yr_out)
        self.last_vx = vx_out
        self.last_vy = vy_out
        return True

class TagHoverController:
    # ── Tuning ────────────────────────────────────────────────────────────────
    MAX_LAT_V         = 0.2
    MAX_FWD_V         = 0.2
    MAX_YAW_RATE      = 0.25
    YAW_KP            = 0.8
    YAW_TOL           = 0.1     # rad
    STOP_TOL          = 0.07    
    CENTER_TOL        = 0.1     
    HOVER_NEED        = 20      
    BRAKE_S           = 1.5     
    REACQ_MAX_S       = 1.2    
    REACQ_GAIN        = 0.3
    PITCH_HOLD        = math.radians(3.5)
    PITCH_RATE_HOLD   = math.radians(8.0)
    EY_VX_SIGN        = -1.0   

    def __init__(self):
        self._x_pid       = PID(kp=0.35, ki=0.02, kd=0.25,
                                out_limit=self.MAX_LAT_V, i_limit=0.2)
        self._y_pid       = PID(kp=0.35, ki=0.02, kd=0.25,
                                out_limit=self.MAX_FWD_V, i_limit=0.2)
        self._x_lpf       = LowPass(alpha=0.1)
        self._y_lpf       = LowPass(alpha=0.1)
        self._vx_slew     = SlewRateLimiter(rate_per_s=0.2)
        self._vy_slew     = SlewRateLimiter(rate_per_s=0.2)

        self.hover_good   = 0
        self._lock_yaw    = None
        self._lock_t      = None
        self._last_t      = 0.0
        self._last_tag    = None
        self._last_tag_t  = 0.0

    @property
    def lock_yaw(self):
        return self._lock_yaw

    def enter(self, lock_yaw, entry_vx: float = 0.0, entry_vy: float = 0.0):
        self._lock_yaw   = lock_yaw
        self._lock_t     = time.time()
        self._last_t     = time.time()
        self.hover_good  = 0
        self._last_tag   = None
        self._last_tag_t = 0.0
        self._x_pid.reset();    self._y_pid.reset()
        self._x_lpf.reset(0.0); self._y_lpf.reset(0.0)
        self._vx_slew.seed(entry_vx)
        self._vy_slew.seed(entry_vy)

    def update(self, master, tag: dict, img,
               att_yaw=None, att_pitch=None, att_pitch_rate=None) -> int:
        now    = time.time()
        dt     = (now - self._last_t) if self._last_t > 0 else 0.05
        self._last_t = now

        lock_elapsed = (now - self._lock_t) if self._lock_t else 0.0

        if self._lock_yaw is None and att_yaw is not None:
            self._lock_yaw = att_yaw

        # Yaw hold rate
        yaw_err       = 0.0
        yaw_rate_hold = 0.0
        if self._lock_yaw is not None and att_yaw is not None:
            yaw_err = wrap_pi(self._lock_yaw - att_yaw)
            yaw_rate_hold = max(-self.MAX_YAW_RATE,
                                min(self.MAX_YAW_RATE, self.YAW_KP * yaw_err))

        # ── Braking phase ─────────────────────────────────────────────────────
        if lock_elapsed < self.BRAKE_S:
            vx = self._vx_slew.update(0.0, dt)
            vy = self._vy_slew.update(0.0, dt)
            send_body_velocity(master, vx, vy, 0.0, yaw_rate_hold)
            self.hover_good = 0
            return self.hover_good

        # ── Pitch guard ───────────────────────────────────────────────────────
        pitch_unstable = (
            (att_pitch      is not None and abs(att_pitch)      > self.PITCH_HOLD)
            or
            (att_pitch_rate is not None and abs(att_pitch_rate) > self.PITCH_RATE_HOLD)
        )

        # ── Tag lost ──────────────────────────────────────────────────────────
        if not tag.get("found", False):
            if self._last_tag is not None and (now - self._last_tag_t) <= self.REACQ_MAX_S:
                h, w = img.shape[:2]
                cx, cy  = self._last_tag["center"]
                ex_mem  = (cx - w / 2.0) / (w / 2.0)
                ey_mem  = (cy - h / 2.0) / (h / 2.0)
                vy = max(-self.MAX_LAT_V * 0.5,
                         min(self.MAX_LAT_V * 0.5, self.REACQ_GAIN * ex_mem))
                vx = max(-self.MAX_FWD_V * 0.5,
                         min(self.MAX_FWD_V * 0.5,
                             self.REACQ_GAIN * self.EY_VX_SIGN * ey_mem))
                if pitch_unstable:
                    vx = 0.0
                vx = self._vx_slew.update(vx, dt)
                vy = self._vy_slew.update(vy, dt)
                send_body_velocity(master, vx, vy, 0.0, yaw_rate_hold)
            else:
                send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_hold)
            self.hover_good = 0
            return self.hover_good

        # ── Tag visible ───────────────────────────────────────────────────────
        self._last_tag   = tag
        self._last_tag_t = now
        h, w = img.shape[:2]
        cx, cy = tag["center"]

        ex = (cx - w / 2.0) / (w / 2.0)
        ey = (cy - h / 2.0) / (h / 2.0)

        ex_f = self._x_lpf.update(ex)
        ey_f = self._y_lpf.update(ey)

        vy = 0.0 if abs(ex_f) < self.STOP_TOL else self._x_pid.update(ex_f, dt)

        # Pitch-proportional forward velocity suppression
        p_abs  = abs(att_pitch)      if att_pitch      is not None else 0.0
        pr_abs = abs(att_pitch_rate) if att_pitch_rate is not None else 0.0
        suppress = max(0.0, min(1.0, 1.0 - (p_abs / self.PITCH_HOLD) * 1.5))
        if pr_abs > math.radians(5.0):
            suppress *= 0.3

        if pitch_unstable:
            vx = 0.0
            self._y_pid.reset()
        else:
            raw_vx = (0.0 if abs(ey_f) < self.STOP_TOL
                      else self.EY_VX_SIGN * self._y_pid.update(ey_f, dt))
            vx = raw_vx * suppress

        vx = self._vx_slew.update(vx, dt)
        vy = self._vy_slew.update(vy, dt)
        send_body_velocity(master, vx, vy, 0.0, yaw_rate_hold)

        # Centred check
        yaw_ok = abs(yaw_err) < self.YAW_TOL
        if abs(ex_f) < self.CENTER_TOL and abs(ey_f) < self.CENTER_TOL and yaw_ok:
            self.hover_good += 1
        else:
            self.hover_good = 0

        return self.hover_good