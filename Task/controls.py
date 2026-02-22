import time
import math
from dataclasses import dataclass
from pymavlink import mavutil

def set_param(master, name: str, value: float):
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        name.encode("utf-8"),
        float(value),
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )

def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def force_arm(master): 
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1,      # 1 to arm, 0 to disarm
        21196,  # Force-arm magic number for ArduPilot
        0, 0, 0, 0, 0
    )

def is_armed(master) -> bool:
    hb = master.recv_match(type='HEARTBEAT', blocking=False)
    if not hb:
        return False
    return (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0

def takeoff(master, target_alt):
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0,
        0, 0,
        target_alt
    )


class AltitudeTracker:
    def __init__(self):
        self.stable_count = 0
        self.last_alt = None
        self.stable_needed = 6

    def update(self, master, target_alt, tol):
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if not msg:
            return False, self.last_alt
        
        self.last_alt = msg.relative_alt / 1000.0
        if self.last_alt >= (target_alt - tol):
            self.stable_count += 1
        else:
            self.stable_count = 0

        return self.stable_count >= self.stable_needed, self.last_alt
    
@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    out_limit: float
    i_limit: float

    def __post_init__(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.prev_t = None

    def update(self, e: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        
        # Integral
        self.i += e * dt
        if self.i > self.i_limit: self.i = self.i_limit
        if self.i < -self.i_limit: self.i = -self.i_limit

        # Derivative on error
        de = (e - self.prev_e) / dt
        self.prev_e = e

        u = self.kp * e + self.ki * self.i + self.kd * de

        # Output clamp
        if u > self.out_limit: u = self.out_limit
        if u < -self.out_limit: u = -self.out_limit
        return u
    
class LowPass:
    """1st-order low-pass: y = y + a*(x-y), a in (0..1)"""
    def __init__(self, alpha: float, init: float = 0.0):
        self.alpha = float(alpha)
        self.y = float(init)
        self.inited = False

    def reset(self, v: float = 0.0):
        self.y = float(v)
        self.inited = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.inited:
            self.y = x
            self.inited = True
            return self.y
        self.y = self.y + self.alpha * (x - self.y)
        return self.y


class SlewRateLimiter:
    """Limits rate of change: |dy/dt| <= rate"""
    def __init__(self, rate_per_s: float, init: float = 0.0):
        self.rate = float(rate_per_s)
        self.y = float(init)
        self.inited = False

    def reset(self, v: float = 0.0):
        self.y = float(v)
        self.inited = False

    def update(self, x: float, dt: float) -> float:
        x = float(x)
        if dt <= 0:
            return self.y
        if not self.inited:
            self.y = x
            self.inited = True
            return self.y

        max_step = self.rate * dt
        delta = x - self.y
        if delta > max_step: delta = max_step
        if delta < -max_step: delta = -max_step
        self.y += delta
        return self.y


@dataclass
class AxisStabilityChecker:
    vxy_tol: float
    vz_tol: float
    rate_tol: float
    stable_needed: int

    def __post_init__(self):
        self.count = 0

    def reset(self):
        self.count = 0

    def update(self, vx, vy, vz, roll_rate, pitch_rate, yaw_rate) -> bool:
        samples = (vx, vy, vz, roll_rate, pitch_rate, yaw_rate)
        if any(v is None for v in samples):
            self.count = 0
            return False

        stable = (
            abs(vx) <= self.vxy_tol and
            abs(vy) <= self.vxy_tol and
            abs(vz) <= self.vz_tol and
            abs(roll_rate) <= self.rate_tol and
            abs(pitch_rate) <= self.rate_tol and
            abs(yaw_rate) <= self.rate_tol
        )

        if stable:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.stable_needed


def yaw_hold_rate(target_yaw, current_yaw, kp: float, max_rate: float) -> float:
    if target_yaw is None or current_yaw is None:
        return 0.0
    err = wrap_pi(target_yaw - current_yaw)
    return float(max(-max_rate, min(max_rate, kp * err)))


def ramp_xy_to_stop(vx_slew: SlewRateLimiter, vy_slew: SlewRateLimiter, dt: float):
    return vx_slew.update(0.0, dt), vy_slew.update(0.0, dt)


def tag_xy_commands(
    ex: float,
    ey: float,
    dt: float,
    x_pid: PID,
    y_pid: PID,
    x_lpf: LowPass,
    y_lpf: LowPass,
    vx_slew: SlewRateLimiter,
    vy_slew: SlewRateLimiter,
    stop_tol: float,
    ey_to_vx_sign: float,
):
    ex_f = x_lpf.update(ex)
    ey_f = y_lpf.update(ey)

    vy_cmd = 0.0 if abs(ex_f) < stop_tol else x_pid.update(ex_f, dt)
    vx_raw = 0.0 if abs(ey_f) < stop_tol else (ey_to_vx_sign * y_pid.update(ey_f, dt))

    vx_cmd = vx_slew.update(vx_raw, dt)
    vy_cmd = vy_slew.update(vy_cmd, dt)
    return vx_cmd, vy_cmd, ex_f, ey_f


def send_body_velocity(master, vx: float, vy: float, vz: float, yaw_rate: float):
    """
    BODY_NED frame:
      vx: forward (+)
      vy: right (+)
      vz: down (+)  -> for altitude hold, keep near 0
    yaw_rate: rad/s (+) is yaw right (generally)
    """
    # Ignore everything except vx,vy,vz and yaw_rate
    type_mask = (
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
        mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )

    master.mav.set_position_target_local_ned_send(
        0,  
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_BODY_NED,
        type_mask,
        0, 0, 0,          # x,y,z position 
        vx, vy, vz,       # velocities
        0, 0, 0,          # accelerations 
        0,                # yaw 
        yaw_rate
    )
