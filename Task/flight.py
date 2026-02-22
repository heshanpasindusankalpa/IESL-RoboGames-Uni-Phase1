import time
import math
import socket
import struct
import numpy as np
import cv2
from pymavlink import mavutil
from states import DroneState
import controls
from controls import wrap_pi
import vision

# ─── Connections ──────────────────────────────────────────────────────────────
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
# master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')

CAM_HOST = "127.0.0.1"
CAM_PORT = 5599

cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cam_sock.connect((CAM_HOST, CAM_PORT))
cam_sock.settimeout(0.2)
print("Connected to camera stream")

master.wait_heartbeat()
print(f"Connected to system {master.target_system}, component {master.target_component}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def set_mode(mode_name: str):
    modes = master.mode_mapping()
    if mode_name not in modes:
        raise RuntimeError(f"Mode {mode_name} not in {list(modes.keys())}")
    mode_id = modes[mode_name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )

def recv_exact(sock, nbytes: int) -> bytes:
    data = b""
    while len(data) < nbytes:
        try:
            chunk = sock.recv(nbytes - len(data))
        except socket.timeout:
            raise TimeoutError("Camera recv timeout")
        if not chunk:
            raise ConnectionError("Camera stream closed")
        data += chunk
    return data

def tag_inside_roi(tag_measurement, roi):
    if not tag_measurement.get("found", False) or roi is None:
        return False
    cx, cy = tag_measurement["center"]
    if len(roi) == 2:
        x0, x1 = roi
        y0, y1 = 0, float("inf")
    else:
        x0, x1, y0, y1 = roi
    return (x0 <= cx < x1) and (y0 <= cy < y1)

# ─── State ────────────────────────────────────────────────────────────────────
current_state = DroneState.INITIALIZING

# ─── Takeoff / Hover ──────────────────────────────────────────────────────────
ARM_TIMEOUT_S           = 5.0
arm_requested_at        = None
last_img                = None
last_frame_t            = 0.0
TARGET_ALT              = 2.5   # meters
ALT_TOL                 = 0.2   # meters
TAKEOFF_TIMEOUT_S       = 35.0
TAKEOFF_CLIMB_SPEED_MPS = 0.6
TAKEOFF_ACCEL_Z_MPS2    = 0.8
TAKEOFF_MIN_ACCEPT_ALT  = TARGET_ALT - 0.35
alt_tracker             = controls.AltitudeTracker()
takeoff_started_at      = None
HOVER_STABILIZE_S       = 2.0
hover_started_at        = None
ALT_LOG_HZ              = 2.0
ALT_LOG_DT              = 1.0 / ALT_LOG_HZ
last_alt_log_t          = 0.0

# ─── Vision / ROI ─────────────────────────────────────────────────────────────
LINE_THRESH_DEBUG        = 155
LINE_ROI_TOP_RATIO_FOLLOW = 0.30
LINE_ROI_TOP_RATIO_FULL   = 1.00
line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FOLLOW

# ─── Line following ───────────────────────────────────────────────────────────
FOLLOW_FWD_V              = 0.5       # m/s forward cruise
MAX_LAT_V                 = 0.6       # m/s max lateral
MAX_YAW_RATE              = 0.8       # rad/s
CMD_HZ                    = 20.0
CMD_DT                    = 1.0 / CMD_HZ
FOLLOW_START_V            = 0.05      # m/s initial ramp speed
FOLLOW_ACCEL_MPS2         = 0.18      # m/s² ramp rate
FOLLOW_YAW_ENABLE_V       = 0.45      # don't yaw below this speed
FOLLOW_ANGLE_DEADBAND_RAD = math.radians(4.0)
PITCH_SOFT_LIMIT_RAD      = math.radians(8.0)   # start scaling down vx
PITCH_HARD_LIMIT_RAD      = math.radians(15.0)  # hard cap vx to 0.10 m/s
PITCH_RATE_SOFT_RAD_S     = math.radians(20.0)  # extra scale if pitching fast

lat_pid = controls.PID(kp=0.9,  ki=0.0, kd=0.15, out_limit=MAX_LAT_V,   i_limit=0.5)
yaw_pid = controls.PID(kp=1.8,  ki=0.0, kd=0.25, out_limit=MAX_YAW_RATE, i_limit=0.8)

off_lpf  = controls.LowPass(alpha=0.25)
ang_lpf  = controls.LowPass(alpha=0.25)

vx_slew  = controls.SlewRateLimiter(rate_per_s=0.45)
vy_slew  = controls.SlewRateLimiter(rate_per_s=1.2)
yr_slew  = controls.SlewRateLimiter(rate_per_s=2.0)

last_cmd_t      = 0.0
follow_started_at = None
last_vx_cmd     = 0.0
last_vy_cmd     = 0.0

# ─── AprilTag hover (shared for both tags) ────────────────────────────────────
TAG_MAX_LAT_V           = 0.2
TAG_MAX_FWD_V           = 0.2
TAG_MAX_YAW_RATE        = 0.4

tag_x_pid = controls.PID(kp=0.35, ki=0.02, kd=0.25, out_limit=TAG_MAX_LAT_V, i_limit=0.2)
tag_y_pid = controls.PID(kp=0.35, ki=0.02, kd=0.25, out_limit=TAG_MAX_FWD_V, i_limit=0.2)

tag_x_lpf = controls.LowPass(alpha=0.1)
tag_y_lpf = controls.LowPass(alpha=0.1)

tag_hover_good          = 0
TAG_HOVER_NEED          = 20        # consecutive frames inside tolerance
TAG_CENTER_TOL          = 0.1       # normalised units
TAG_STOP_TOL            = 0.07      # deadzone – no correction below this
TAG_BRAKE_S             = 1.5       # seconds of braking on tag detection
TAG_YAW_KP              = 0.8
TAG_YAW_RATE_MAX        = 0.25
TAG_YAW_TOL_RAD         = 0.1
TAG_EY_TO_VX_SIGN       = -1.0
TAG_REACQ_MAX_S         = 1.2       # keep using last known position this long
TAG_REACQ_GAIN          = 0.3
TAG_PITCH_HOLD_RAD      = math.radians(3.5)
TAG_PITCH_RATE_HOLD_RAD_S = math.radians(8.0)
TAG_STABLE_VXY_TOL      = 0.08
TAG_STABLE_VZ_TOL       = 0.06
TAG_STABLE_RATE_TOL_RAD_S = math.radians(8.0)
TAG_STABLE_NEED         = 14

# How long to hold centred above tag 2 before landing
TAG2_HOVER_CONFIRM_S    = 3.0
tag2_centered_at        = None      # timestamp when tag2 first confirmed centred

tag_vx_slew = controls.SlewRateLimiter(rate_per_s=0.2)
tag_vy_slew = controls.SlewRateLimiter(rate_per_s=0.2)
tag_yr_slew = controls.SlewRateLimiter(rate_per_s=0.8)
tag_stability = controls.AxisStabilityChecker(
    vxy_tol=TAG_STABLE_VXY_TOL,
    vz_tol=TAG_STABLE_VZ_TOL,
    rate_tol=TAG_STABLE_RATE_TOL_RAD_S,
    stable_needed=TAG_STABLE_NEED,
)

# ─── Attitude / velocity telemetry ────────────────────────────────────────────
last_att_pitch     = None
last_att_pitch_rate = None
last_att_yaw       = None
last_att_roll_rate = None
last_att_yaw_rate  = None
last_local_vx      = None
last_local_vy      = None
last_local_vz      = None

# ─── Tag tracking state ───────────────────────────────────────────────────────
last_seen_tag      = None
last_seen_tag_t    = 0.0
last_tag_t         = 0.0
tag_lock_yaw       = None
tag_lock_started_at = None

# ─── Turn state ───────────────────────────────────────────────────────────────
turn_started    = False
turn_start_yaw  = None
turn_target_yaw = None


# ─── Reusable tag-centering helper ────────────────────────────────────────────
def run_scan_apriltag(tag, img, now, dt_tag):
    """
    One frame of AprilTag centering control.
    Reads/writes globals for tag tracking state.
    Sends velocity commands.
    Returns tag_hover_good (int).
    """
    global tag_hover_good, last_seen_tag, last_seen_tag_t
    global last_vx_cmd, last_vy_cmd

    # ── Pitch stability gate ──────────────────────────────────────────────────
    pitch_unstable = False
    if last_att_pitch is not None and abs(last_att_pitch) > TAG_PITCH_HOLD_RAD:
        pitch_unstable = True
    if last_att_pitch_rate is not None and abs(last_att_pitch_rate) > TAG_PITCH_RATE_HOLD_RAD_S:
        pitch_unstable = True

    # ── Yaw hold ─────────────────────────────────────────────────────────────
    yaw_err = 0.0
    yaw_rate_hold = 0.0
    if tag_lock_yaw is not None and last_att_yaw is not None:
        yaw_err = wrap_pi(tag_lock_yaw - last_att_yaw)
        yaw_rate_hold = float(np.clip(TAG_YAW_KP * yaw_err, -TAG_YAW_RATE_MAX, TAG_YAW_RATE_MAX))

    # ── Tag not visible ───────────────────────────────────────────────────────
    if not tag.get("found", False):
        if last_seen_tag is not None and (now - last_seen_tag_t) <= TAG_REACQ_MAX_S:
            h, w = img.shape[:2]
            cx, cy = last_seen_tag["center"]
            ex_mem = (cx - (w / 2.0)) / (w / 2.0)
            ey_mem = (cy - (h / 2.0)) / (h / 2.0)
            vy_cmd = np.clip(TAG_REACQ_GAIN * ex_mem,
                             -TAG_MAX_LAT_V * 0.5, TAG_MAX_LAT_V * 0.5)
            vx_cmd = np.clip(TAG_REACQ_GAIN * TAG_EY_TO_VX_SIGN * ey_mem,
                             -TAG_MAX_FWD_V * 0.5, TAG_MAX_FWD_V * 0.5)
            if pitch_unstable:
                vx_cmd = 0.0
            vx_cmd = tag_vx_slew.update(vx_cmd, dt_tag)
            vy_cmd = tag_vy_slew.update(vy_cmd, dt_tag)
            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_hold)
            last_vx_cmd = vx_cmd
            last_vy_cmd = vy_cmd
        else:
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_hold)
            last_vx_cmd = 0.0
            last_vy_cmd = 0.0
        tag_hover_good = 0
        return tag_hover_good

    # ── Tag visible ───────────────────────────────────────────────────────────
    last_seen_tag   = tag
    last_seen_tag_t = now
    h, w = img.shape[:2]
    cx, cy = tag["center"]

    ex = (cx - (w / 2.0)) / (w / 2.0)   # left–, right+
    ey = (cy - (h / 2.0)) / (h / 2.0)   # up–,   down+

    ex_f = tag_x_lpf.update(ex)
    ey_f = tag_y_lpf.update(ey)

    vy_cmd = 0.0 if abs(ex_f) < TAG_STOP_TOL else tag_x_pid.update(ex_f, dt_tag)

    # Pitch-proportional suppression of forward velocity
    pitch_abs      = abs(last_att_pitch)      if last_att_pitch      is not None else 0.0
    pitch_rate_abs = abs(last_att_pitch_rate) if last_att_pitch_rate is not None else 0.0
    pitch_suppress = max(0.0, 1.0 - (pitch_abs / TAG_PITCH_HOLD_RAD) * 1.5)
    pitch_suppress = min(1.0, pitch_suppress)
    if pitch_rate_abs > math.radians(5.0):
        pitch_suppress *= 0.3

    if pitch_unstable:
        vx_cmd = 0.0
        tag_y_pid.reset()
    else:
        raw_vx = 0.0 if abs(ey_f) < TAG_STOP_TOL else (TAG_EY_TO_VX_SIGN * tag_y_pid.update(ey_f, dt_tag))
        vx_cmd = raw_vx * pitch_suppress

    vx_cmd = tag_vx_slew.update(vx_cmd, dt_tag)
    vy_cmd = tag_vy_slew.update(vy_cmd, dt_tag)

    controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_hold)
    last_vx_cmd = vx_cmd
    last_vy_cmd = vy_cmd

    # ── Centred check ─────────────────────────────────────────────────────────
    yaw_ok = abs(yaw_err) < TAG_YAW_TOL_RAD
    if abs(ex_f) < TAG_CENTER_TOL and abs(ey_f) < TAG_CENTER_TOL and yaw_ok:
        tag_hover_good += 1
    else:
        tag_hover_good = 0

    return tag_hover_good


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
while True:

    # ── Camera frame ──────────────────────────────────────────────────────────
    try:
        header  = recv_exact(cam_sock, 4)
        width, height = struct.unpack("<HH", header)
        payload = recv_exact(cam_sock, width * height)
        img     = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
        last_img = img
        last_frame_t = time.time()
    except TimeoutError:
        img = last_img
    except ConnectionError as e:
        print(f"Camera error: {e}")
        break
    if img is None:
        continue

    # ── Attitude telemetry ────────────────────────────────────────────────────
    att_msg = master.recv_match(type='ATTITUDE', blocking=False)
    while att_msg is not None:
        last_att_yaw        = float(att_msg.yaw)
        last_att_pitch      = float(att_msg.pitch)
        last_att_pitch_rate = float(att_msg.pitchspeed)
        last_att_roll_rate  = float(att_msg.rollspeed)
        last_att_yaw_rate   = float(att_msg.yawspeed)
        att_msg = master.recv_match(type='ATTITUDE', blocking=False)

    # ── AprilTag detection (every frame) ─────────────────────────────────────
    tag = vision.detect_apriltag(img)

    # ── Line detection (only when actively following a line) ─────────────────
    line_states = (DroneState.FOLLOW_LINE_01, DroneState.FOLLOW_LINE_02)
    if current_state in line_states:
        m = vision.detect_line(
            img,
            thresh_val=LINE_THRESH_DEBUG,
            roi_top_ratio=line_roi_top_ratio_active,
        )
        vis = vision.draw_debug(img, m)
    else:
        m = {"found": False, "cx": None, "cy": None,
             "angle_rad": None, "angle_error_rad": None, "roi": None}
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if tag.get("found", False):
            corners = tag.get("corners")
            if corners is not None:
                pts = corners.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            tcx, tcy = tag["center"]
            cv2.circle(vis, (int(tcx), int(tcy)), 4, (0, 255, 255), -1)
            cv2.putText(vis, f"TAG id={tag['id']}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "TAG SEARCH", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)

    # ── State label overlay ───────────────────────────────────────────────────
    cv2.putText(vis, current_state.name, (20, vis.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

    cv2.imshow("Webots Camera", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        vision.dump_gray_values(img, LINE_THRESH_DEBUG)
        _, dbg_mask = cv2.threshold(img, LINE_THRESH_DEBUG, 255, cv2.THRESH_BINARY)
        print(f"mask_white_ratio={(dbg_mask > 0).mean():.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # STATE MACHINE
    # ══════════════════════════════════════════════════════════════════════════

    # ── INITIALIZING ──────────────────────────────────────────────────────────
    if current_state == DroneState.INITIALIZING:
        set_mode('GUIDED')
        if arm_requested_at is None:
            controls.force_arm(master)
            arm_requested_at = time.time()
            print("Arm requested...")
        if controls.is_armed(master):
            print("Armed ✅")
            current_state = DroneState.ARMED
        else:
            if (time.time() - arm_requested_at) > ARM_TIMEOUT_S:
                print("Arming FAILED (timeout)")
                arm_requested_at = None

    # ── ARMED ─────────────────────────────────────────────────────────────────
    if current_state == DroneState.ARMED:
        controls.set_param(master, "WPNAV_SPEED_UP", TAKEOFF_CLIMB_SPEED_MPS * 100.0)
        controls.set_param(master, "WPNAV_ACCEL_Z",  TAKEOFF_ACCEL_Z_MPS2   * 100.0)
        time.sleep(0.2)
        controls.takeoff(master, TARGET_ALT)
        alt_tracker.stable_count = 0
        alt_tracker.last_alt     = None
        takeoff_started_at       = time.time()
        current_state            = DroneState.TAKEOFF

    # ── TAKEOFF ───────────────────────────────────────────────────────────────
    if current_state == DroneState.TAKEOFF:
        reached, current_alt = alt_tracker.update(master, TARGET_ALT, ALT_TOL)
        now_t = time.time()
        if current_alt is not None and (now_t - last_alt_log_t) >= ALT_LOG_DT:
            print(f"Altitude: {current_alt:.2f} m")
            last_alt_log_t = now_t
        if reached:
            print("Reached target altitude — entering HOVER")
            current_state    = DroneState.HOVER
            hover_started_at = time.time()
        elif takeoff_started_at and (time.time() - takeoff_started_at) > TAKEOFF_TIMEOUT_S:
            if current_alt is not None and current_alt >= TAKEOFF_MIN_ACCEPT_ALT:
                print("TAKEOFF timeout but altitude acceptable — entering HOVER")
                current_state    = DroneState.HOVER
                hover_started_at = time.time()
            else:
                print("TAKEOFF timeout — switching to LAND")
                set_mode('LAND')
                current_state = DroneState.LANDING

    # ── HOVER ─────────────────────────────────────────────────────────────────
    if current_state == DroneState.HOVER:
        if (time.time() - hover_started_at) >= HOVER_STABILIZE_S:
            print("Hover stabilised — starting FOLLOW_LINE_01")
            line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FOLLOW
            follow_started_at = time.time()
            vx_slew.y = 0.0; vx_slew.inited = True
            vy_slew.y = 0.0; vy_slew.inited = True
            yr_slew.y = 0.0; yr_slew.inited = True
            off_lpf.reset(0.0); ang_lpf.reset(0.0)
            lat_pid.reset();    yaw_pid.reset()
            current_state = DroneState.FOLLOW_LINE_01

    # ── FOLLOW_LINE_01 ────────────────────────────────────────────────────────
    if current_state == DroneState.FOLLOW_LINE_01:
        # Transition immediately if tag appears inside the line ROI
        if tag_inside_roi(tag, m.get("roi")):
            print(f"[LINE01] AprilTag seen: id={tag['id']} → APRILTAG_01_DETECTED")
            last_seen_tag            = tag
            line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FULL
            tag_lock_yaw             = last_att_yaw
            tag_lock_started_at      = time.time()
            tag_hover_good           = 0
            tag_x_pid.reset();  tag_y_pid.reset()
            tag_x_lpf.reset(0.0); tag_y_lpf.reset(0.0)
            tag_vx_slew.y = last_vx_cmd; tag_vx_slew.inited = True
            tag_vy_slew.y = last_vy_cmd; tag_vy_slew.inited = True
            last_tag_t               = time.time()
            last_seen_tag_t          = time.time()
            current_state            = DroneState.APRILTAG_01_DETECTED
            continue

        # Line-follow at CMD_HZ
        now      = time.time()
        dt       = (now - last_cmd_t) if last_cmd_t > 0 else CMD_DT
        send_now = (now - last_cmd_t) >= CMD_DT
        if send_now:
            last_cmd_t    = now
            h, w          = img.shape[:2]
            follow_elapsed = (now - follow_started_at) if follow_started_at else 0.0
            ramp_v_max     = min(FOLLOW_FWD_V,
                                 FOLLOW_START_V + FOLLOW_ACCEL_MPS2 * max(0.0, follow_elapsed))

            if not m.get("found", False):
                vx_cmd       = min(0.2, ramp_v_max)
                vy_cmd       = 0.0
                yaw_rate_cmd = 0.0 if ramp_v_max < FOLLOW_YAW_ENABLE_V else 0.2
            else:
                offset_px  = m["cx"] - (w / 2.0)
                offset_n   = offset_px / (w / 2.0)
                ang_err    = float(m["angle_error_rad"])
                offset_n_f = off_lpf.update(offset_n)
                ang_err_f  = ang_lpf.update(ang_err)

                vy_cmd = lat_pid.update(offset_n_f, dt)
                if abs(ang_err_f) < FOLLOW_ANGLE_DEADBAND_RAD:
                    ang_err_f = 0.0
                if ramp_v_max < FOLLOW_YAW_ENABLE_V:
                    yaw_rate_cmd = 0.0
                    yaw_pid.reset()
                else:
                    yaw_rate_cmd = -yaw_pid.update(ang_err_f, dt)

                err_mag = min(1.0, abs(offset_n_f) + 0.6 * abs(ang_err_f))
                vx_cmd  = max(0.25, FOLLOW_FWD_V * (1.0 - 0.5 * err_mag))
                vx_cmd  = min(vx_cmd, ramp_v_max)

            if last_att_pitch is not None:
                pitch_abs = abs(last_att_pitch)
                if pitch_abs >= PITCH_HARD_LIMIT_RAD:
                    vx_cmd = min(vx_cmd, 0.10)
                elif pitch_abs > PITCH_SOFT_LIMIT_RAD:
                    p = ((pitch_abs - PITCH_SOFT_LIMIT_RAD)
                         / (PITCH_HARD_LIMIT_RAD - PITCH_SOFT_LIMIT_RAD))
                    vx_cmd *= max(0.2, 1.0 - p)
            if last_att_pitch_rate is not None and abs(last_att_pitch_rate) > PITCH_RATE_SOFT_RAD_S:
                vx_cmd *= 0.7

            vx_cmd       = vx_slew.update(vx_cmd, dt)
            vy_cmd       = vy_slew.update(vy_cmd, dt)
            yaw_rate_cmd = yr_slew.update(yaw_rate_cmd, dt)

            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_cmd)
            last_vx_cmd = vx_cmd
            last_vy_cmd = vy_cmd

    # ── APRILTAG_01_DETECTED ──────────────────────────────────────────────────
    if current_state == DroneState.APRILTAG_01_DETECTED:
        current_state = DroneState.SCAN_APRILTAG_01

    # ── SCAN_APRILTAG_01 ──────────────────────────────────────────────────────
    if current_state == DroneState.SCAN_APRILTAG_01:
        now    = time.time()
        dt_tag = (now - last_tag_t) if last_tag_t > 0 else CMD_DT
        last_tag_t    = now
        lock_elapsed  = (now - tag_lock_started_at) if tag_lock_started_at else 0.0

        # Ensure yaw lock is captured
        if tag_lock_yaw is None and last_att_yaw is not None:
            tag_lock_yaw = last_att_yaw

        # Braking phase – bleed off forward momentum
        if lock_elapsed < TAG_BRAKE_S:
            vx_cmd = tag_vx_slew.update(0.0, dt_tag)
            vy_cmd = tag_vy_slew.update(0.0, dt_tag)
            yaw_err      = wrap_pi(tag_lock_yaw - last_att_yaw) if tag_lock_yaw and last_att_yaw else 0.0
            yaw_rate_hold = float(np.clip(TAG_YAW_KP * yaw_err, -TAG_YAW_RATE_MAX, TAG_YAW_RATE_MAX))
            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_hold)
            last_vx_cmd = vx_cmd
            last_vy_cmd = vy_cmd
            tag_hover_good = 0
            continue

        good = run_scan_apriltag(tag, img, now, dt_tag)

        if good >= TAG_HOVER_NEED:
            print(f"AprilTag 01 centred ✅ id={tag['id']} → TURN_RIGHT_90")
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
            turn_started = False
            current_state = DroneState.TURN_RIGHT_90

    # ── TURN_RIGHT_90 ─────────────────────────────────────────────────────────
    if current_state == DroneState.TURN_RIGHT_90:
        # Use cached attitude; poll fresh if available
        att_now = master.recv_match(type='ATTITUDE', blocking=False)
        if att_now:
            last_att_yaw = float(att_now.yaw)

        if last_att_yaw is None:
            # No attitude yet – nudge right until we get telemetry
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.3)
        else:
            if not turn_started:
                turn_started    = True
                turn_start_yaw  = last_att_yaw
                # ArduPilot NED: positive yaw_rate = clockwise (right)
                # So target = start + π/2  (wrapped)
                turn_target_yaw = wrap_pi(turn_start_yaw + math.pi / 2.0)
                print(f"Turn start: {math.degrees(turn_start_yaw):.1f}° → "
                      f"target: {math.degrees(turn_target_yaw):.1f}°")

            err          = wrap_pi(turn_target_yaw - last_att_yaw)
            yaw_rate_cmd = float(np.clip(1.2 * err, -0.6, 0.6))
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_cmd)

            if abs(err) < math.radians(3.5):
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                print("Turn 90° right ✅ → FOLLOW_LINE_02")
                # Reset line-following state for the second leg
                line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FOLLOW
                follow_started_at = time.time()
                vx_slew.y = 0.0; vx_slew.inited = True
                vy_slew.y = 0.0; vy_slew.inited = True
                yr_slew.y = 0.0; yr_slew.inited = True
                off_lpf.reset(0.0); ang_lpf.reset(0.0)
                lat_pid.reset();    yaw_pid.reset()
                # Reset tag state so we don't carry over tag-01 data
                last_seen_tag   = None
                last_seen_tag_t = 0.0
                tag_hover_good  = 0
                current_state   = DroneState.FOLLOW_LINE_02

    # ── FOLLOW_LINE_02 ────────────────────────────────────────────────────────
    if current_state == DroneState.FOLLOW_LINE_02:
        # Transition immediately if tag appears inside the line ROI
        if tag_inside_roi(tag, m.get("roi")):
            print(f"[LINE02] AprilTag seen: id={tag['id']} → APRILTAG_02_DETECTED")
            last_seen_tag             = tag
            line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FULL
            tag_lock_yaw              = last_att_yaw
            tag_lock_started_at       = time.time()
            tag_hover_good            = 0
            tag2_centered_at          = None
            tag_x_pid.reset();  tag_y_pid.reset()
            tag_x_lpf.reset(0.0); tag_y_lpf.reset(0.0)
            tag_vx_slew.y = last_vx_cmd; tag_vx_slew.inited = True
            tag_vy_slew.y = last_vy_cmd; tag_vy_slew.inited = True
            last_tag_t                = time.time()
            last_seen_tag_t           = time.time()
            current_state             = DroneState.APRILTAG_02_DETECTED
            continue

        # Same line-follow logic as LINE_01
        now      = time.time()
        dt       = (now - last_cmd_t) if last_cmd_t > 0 else CMD_DT
        send_now = (now - last_cmd_t) >= CMD_DT
        if send_now:
            last_cmd_t     = now
            h, w           = img.shape[:2]
            follow_elapsed = (now - follow_started_at) if follow_started_at else 0.0
            ramp_v_max     = min(FOLLOW_FWD_V,
                                 FOLLOW_START_V + FOLLOW_ACCEL_MPS2 * max(0.0, follow_elapsed))

            if not m.get("found", False):
                vx_cmd       = min(0.2, ramp_v_max)
                vy_cmd       = 0.0
                yaw_rate_cmd = 0.0 if ramp_v_max < FOLLOW_YAW_ENABLE_V else 0.2
            else:
                offset_px  = m["cx"] - (w / 2.0)
                offset_n   = offset_px / (w / 2.0)
                ang_err    = float(m["angle_error_rad"])
                offset_n_f = off_lpf.update(offset_n)
                ang_err_f  = ang_lpf.update(ang_err)

                vy_cmd = lat_pid.update(offset_n_f, dt)
                if abs(ang_err_f) < FOLLOW_ANGLE_DEADBAND_RAD:
                    ang_err_f = 0.0
                if ramp_v_max < FOLLOW_YAW_ENABLE_V:
                    yaw_rate_cmd = 0.0
                    yaw_pid.reset()
                else:
                    yaw_rate_cmd = -yaw_pid.update(ang_err_f, dt)

                err_mag = min(1.0, abs(offset_n_f) + 0.6 * abs(ang_err_f))
                vx_cmd  = max(0.25, FOLLOW_FWD_V * (1.0 - 0.5 * err_mag))
                vx_cmd  = min(vx_cmd, ramp_v_max)

            if last_att_pitch is not None:
                pitch_abs = abs(last_att_pitch)
                if pitch_abs >= PITCH_HARD_LIMIT_RAD:
                    vx_cmd = min(vx_cmd, 0.10)
                elif pitch_abs > PITCH_SOFT_LIMIT_RAD:
                    p = ((pitch_abs - PITCH_SOFT_LIMIT_RAD)
                         / (PITCH_HARD_LIMIT_RAD - PITCH_SOFT_LIMIT_RAD))
                    vx_cmd *= max(0.2, 1.0 - p)
            if last_att_pitch_rate is not None and abs(last_att_pitch_rate) > PITCH_RATE_SOFT_RAD_S:
                vx_cmd *= 0.7

            vx_cmd       = vx_slew.update(vx_cmd, dt)
            vy_cmd       = vy_slew.update(vy_cmd, dt)
            yaw_rate_cmd = yr_slew.update(yaw_rate_cmd, dt)

            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_cmd)
            last_vx_cmd = vx_cmd
            last_vy_cmd = vy_cmd

    # ── APRILTAG_02_DETECTED ──────────────────────────────────────────────────
    if current_state == DroneState.APRILTAG_02_DETECTED:
        current_state = DroneState.SCAN_APRILTAG_02

    # ── SCAN_APRILTAG_02 ──────────────────────────────────────────────────────
    if current_state == DroneState.SCAN_APRILTAG_02:
        now    = time.time()
        dt_tag = (now - last_tag_t) if last_tag_t > 0 else CMD_DT
        last_tag_t   = now
        lock_elapsed = (now - tag_lock_started_at) if tag_lock_started_at else 0.0

        if tag_lock_yaw is None and last_att_yaw is not None:
            tag_lock_yaw = last_att_yaw

        # Braking phase
        if lock_elapsed < TAG_BRAKE_S:
            vx_cmd = tag_vx_slew.update(0.0, dt_tag)
            vy_cmd = tag_vy_slew.update(0.0, dt_tag)
            yaw_err       = wrap_pi(tag_lock_yaw - last_att_yaw) if tag_lock_yaw and last_att_yaw else 0.0
            yaw_rate_hold = float(np.clip(TAG_YAW_KP * yaw_err, -TAG_YAW_RATE_MAX, TAG_YAW_RATE_MAX))
            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_hold)
            last_vx_cmd = vx_cmd
            last_vy_cmd = vy_cmd
            tag_hover_good  = 0
            tag2_centered_at = None
            continue

        good = run_scan_apriltag(tag, img, now, dt_tag)

        if good >= TAG_HOVER_NEED:
            # Start (or maintain) the confirmation hover timer
            if tag2_centered_at is None:
                tag2_centered_at = now
                print(f"AprilTag 02 centred ✅ id={tag['id']} — holding for {TAG2_HOVER_CONFIRM_S:.1f}s…")
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)

            elif (now - tag2_centered_at) >= TAG2_HOVER_CONFIRM_S:
                print("Hover confirmed ✅ → LANDING")
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                set_mode('LAND')
                current_state = DroneState.LANDING
        else:
            # Lost centre — reset the confirmation timer
            tag2_centered_at = None

    # ── LANDING ───────────────────────────────────────────────────────────────
    if current_state == DroneState.LANDING:
        # ArduPilot LAND mode handles the descent autonomously.
        # Poll for disarm to know when it's done.
        if not controls.is_armed(master):
            print("Landed and disarmed ✅ — DONE")
            current_state = DroneState.DONE

    # ── DONE ──────────────────────────────────────────────────────────────────
    if current_state == DroneState.DONE:
        break

cv2.destroyAllWindows()
cam_sock.close()
print("Mission complete.")