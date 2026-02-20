import time
import math
import socket
import struct
import numpy as np
import cv2
from pymavlink import mavutil
from states import DroneState
import controls
import vision

# Establish connections
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
# master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')

CAM_HOST = "127.0.0.1"
CAM_PORT = 5599

cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cam_sock.connect((CAM_HOST, CAM_PORT))
cam_sock.settimeout(0.2)
print("Connected to camera stream")

# Wait for heartbeat to confirm connection
master.wait_heartbeat()
print(f"Connected to system {master.target_system}, component {master.target_component}")

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

# Since TCP does not guarantee all bytes in one recv()
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

def wrap_pi(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a


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

current_state = DroneState.INITIALIZING

ARM_TIMEOUT_S = 5.0
arm_requested_at = None
last_img = None
last_frame_t = 0.0
TARGET_ALT = 2.5 # meters
ALT_TOL = 0.2 # meters
TAKEOFF_TIMEOUT_S = 35.0
TAKEOFF_CLIMB_SPEED_MPS = 0.6
TAKEOFF_ACCEL_Z_MPS2 = 0.8
TAKEOFF_MIN_ACCEPT_ALT = TARGET_ALT - 0.35
alt_tracker = controls.AltitudeTracker()
takeoff_started_at = None
HOVER_STABILIZE_S = 2.0
hover_started_at = None
ALT_LOG_HZ = 2.0
ALT_LOG_DT = 1.0 / ALT_LOG_HZ
last_alt_log_t = 0.0
LINE_THRESH_DEBUG = 155
LINE_ROI_TOP_RATIO_FOLLOW = 0.30
LINE_ROI_TOP_RATIO_FULL = 1.00
line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FOLLOW

# --- Line following ---
FOLLOW_FWD_V = 0.8        # m/s forward cruise
MAX_LAT_V = 0.6           # m/s
MAX_YAW_RATE = 0.8        # rad/s
CMD_HZ = 20.0             # command rate
CMD_DT = 1.0 / CMD_HZ
FOLLOW_START_V = 0.05     # m/s
FOLLOW_ACCEL_MPS2 = 0.25  # m/s^2 ramp-up
FOLLOW_YAW_ENABLE_V = 0.45
FOLLOW_ANGLE_DEADBAND_RAD = math.radians(4.0)

# PID: offset -> lateral velocity, angle -> yaw rate
lat_pid = controls.PID(kp=0.9, ki=0.0, kd=0.15, out_limit=MAX_LAT_V, i_limit=0.5)
yaw_pid = controls.PID(kp=1.8, ki=0.0, kd=0.25, out_limit=MAX_YAW_RATE, i_limit=0.8)

# Smoothers
off_lpf = controls.LowPass(alpha=0.25)
ang_lpf = controls.LowPass(alpha=0.25)

vx_slew = controls.SlewRateLimiter(rate_per_s=1.5)   # m/s per second
vy_slew = controls.SlewRateLimiter(rate_per_s=1.2)
yr_slew = controls.SlewRateLimiter(rate_per_s=2.0)   # rad/s per second

last_cmd_t = 0.0
follow_started_at = None

# --- AprilTag hover ---
TAG_MAX_LAT_V = 0.4
TAG_MAX_FWD_V = 0.4
TAG_MAX_YAW_RATE = 0.6

tag_x_pid = controls.PID(kp=0.8, ki=0.0, kd=0.12, out_limit=TAG_MAX_LAT_V, i_limit=0.4)  
tag_y_pid = controls.PID(kp=0.8, ki=0.0, kd=0.12, out_limit=TAG_MAX_FWD_V, i_limit=0.4) 

tag_x_lpf = controls.LowPass(alpha=0.25)
tag_y_lpf = controls.LowPass(alpha=0.25)

tag_hover_good = 0
TAG_HOVER_NEED = 15         
TAG_CENTER_TOL = 0.06      
TAG_BRAKE_S = 0.8
TAG_YAW_KP = 1.4
TAG_YAW_RATE_MAX = 0.35
TAG_YAW_TOL_RAD = 0.08

last_seen_tag = None
last_tag_t = 0.0
last_turn_t = 0.0
last_att_yaw = None
tag_lock_yaw = None
tag_lock_started_at = None

tag_vx_slew = controls.SlewRateLimiter(rate_per_s=0.8)
tag_vy_slew = controls.SlewRateLimiter(rate_per_s=0.8)
tag_yr_slew = controls.SlewRateLimiter(rate_per_s=1.5)

# --- Turn tracking ---
turn_started = False
turn_start_yaw = None
turn_target_yaw = None

while True:
    try:
        header = recv_exact(cam_sock, 4)
        width, height = struct.unpack("<HH", header)
        payload = recv_exact(cam_sock, width * height)
        img = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
        last_img = img
        last_frame_t = time.time()
    except TimeoutError:
        img = last_img
    except ConnectionError as e:
        print(f"Camera error: {e}")
        break
    if img is None:
        continue

    att_msg = master.recv_match(type='ATTITUDE', blocking=False)
    while att_msg is not None:
        last_att_yaw = float(att_msg.yaw)
        att_msg = master.recv_match(type='ATTITUDE', blocking=False)

    m = vision.detect_line(
        img,
        thresh_val=LINE_THRESH_DEBUG,
        roi_top_ratio=line_roi_top_ratio_active,
    )
    tag = vision.detect_apriltag(img)
    vis = vision.draw_debug(img, m)

    cv2.imshow("Webots Camera", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('p'):
        vision.dump_gray_values(img, LINE_THRESH_DEBUG)
        _, dbg_mask = cv2.threshold(img, LINE_THRESH_DEBUG, 255, cv2.THRESH_BINARY)
        print(f"mask_white_ratio={(dbg_mask > 0).mean():.4f}")

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
            if (time.time() - arm_requested_at > ARM_TIMEOUT_S):
                print("Arming FAILED (timeout)")
                arm_requested_at = None

    if current_state == DroneState.ARMED:
        controls.set_param(master, "WPNAV_SPEED_UP", TAKEOFF_CLIMB_SPEED_MPS * 100.0)
        controls.set_param(master, "WPNAV_ACCEL_Z", TAKEOFF_ACCEL_Z_MPS2 * 100.0)
        time.sleep(0.2)
        controls.takeoff(master, TARGET_ALT)

        alt_tracker.stable_count = 0
        alt_tracker.last_alt = None
        takeoff_started_at = time.time()

        current_state = DroneState.TAKEOFF

    if current_state == DroneState.TAKEOFF:
        reached, current_alt = alt_tracker.update(master, TARGET_ALT, ALT_TOL)
        now_takeoff = time.time()
        if current_alt is not None and (now_takeoff - last_alt_log_t) >= ALT_LOG_DT:
            print(f"Altitude: {current_alt:.2f} m")
            last_alt_log_t = now_takeoff

        if reached:
            print("Reached target altitude — entering HOVER")
            current_state = DroneState.HOVER
            hover_started_at = time.time()
        elif takeoff_started_at and (time.time() - takeoff_started_at) > TAKEOFF_TIMEOUT_S:
            if current_alt is not None and current_alt >= TAKEOFF_MIN_ACCEPT_ALT:
                print("TAKEOFF timeout but altitude acceptable — entering HOVER")
                current_state = DroneState.HOVER
                hover_started_at = time.time()
            else:
                print("TAKEOFF timeout — switching to LAND for safety")
                set_mode('LAND')
                current_state = DroneState.LANDING

    if current_state == DroneState.HOVER:
        if (time.time() - hover_started_at >= HOVER_STABILIZE_S):
            print("Hover stabilized — ready for FOLLOW_LINE_01")
            line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FOLLOW
            follow_started_at = time.time()
            vx_slew.y = 0.0
            vx_slew.inited = True
            yr_slew.y = 0.0
            yr_slew.inited = True
            yaw_pid.reset()
            current_state = DroneState.FOLLOW_LINE_01

    if current_state == DroneState.FOLLOW_LINE_01:
        # 1) Transition ASAP if tag appears (no 20Hz gate)
        if tag_inside_roi(tag, m.get("roi")):
            last_seen_tag = tag
            print(f"AprilTag seen while following line: id={tag['id']}")
            line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FULL
            tag_lock_yaw = last_att_yaw
            tag_lock_started_at = time.time()
            current_state = DroneState.APRILTAG_01_DETECTED
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)

            tag_hover_good = 0
            tag_x_pid.reset(); tag_y_pid.reset()
            tag_x_lpf.reset(0.0); tag_y_lpf.reset(0.0)

            tag_vx_slew.reset(0.0); tag_vy_slew.reset(0.0); tag_yr_slew.reset(0.0)
            last_tag_t = time.time()
            continue

        # 2) Otherwise, line-follow command at 20Hz (no 'continue' gate)
        now = time.time()
        send_now = (now - last_cmd_t) >= CMD_DT
        dt = (now - last_cmd_t) if last_cmd_t > 0 else CMD_DT

        if send_now:
            last_cmd_t = now
            h, w = img.shape[:2]
            follow_elapsed = (now - follow_started_at) if follow_started_at else 0.0
            ramp_v_max = min(FOLLOW_FWD_V, FOLLOW_START_V + FOLLOW_ACCEL_MPS2 * max(0.0, follow_elapsed))

            if not m.get("found", False):
                vx_cmd = min(0.2, ramp_v_max)
                vy_cmd = 0.0
                yaw_rate_cmd = 0.0 if ramp_v_max < FOLLOW_YAW_ENABLE_V else 0.2
            else:
                offset_px = (m["cx"] - (w / 2.0))
                offset_n = offset_px / (w / 2.0)
                ang_err = float(m["angle_error_rad"])

                offset_n_f = off_lpf.update(offset_n)
                ang_err_f = ang_lpf.update(ang_err)

                vy_cmd = lat_pid.update(offset_n_f, dt)
                if abs(ang_err_f) < FOLLOW_ANGLE_DEADBAND_RAD:
                    ang_err_f = 0.0
                if ramp_v_max < FOLLOW_YAW_ENABLE_V:
                    yaw_rate_cmd = 0.0
                    yaw_pid.reset()
                else:
                    yaw_rate_cmd = -yaw_pid.update(ang_err_f, dt)

                err_mag = min(1.0, abs(offset_n_f) + 0.6 * abs(ang_err_f))
                vx_cmd = FOLLOW_FWD_V * (1.0 - 0.5 * err_mag)
                if vx_cmd < 0.25:
                    vx_cmd = 0.25
                vx_cmd = min(vx_cmd, ramp_v_max)

            vx_cmd = vx_slew.update(vx_cmd, dt)
            vy_cmd = vy_slew.update(vy_cmd, dt)
            yaw_rate_cmd = yr_slew.update(yaw_rate_cmd, dt)

            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_cmd)

    if current_state == DroneState.APRILTAG_01_DETECTED:
        current_state = DroneState.SCAN_APRILTAG_01

    if current_state == DroneState.SCAN_APRILTAG_01:
        now = time.time()
        dt_tag = (now - last_tag_t) if last_tag_t > 0 else CMD_DT
        last_tag_t = now
        lock_elapsed = (now - tag_lock_started_at) if tag_lock_started_at else 0.0

        if tag_lock_yaw is None and last_att_yaw is not None:
            tag_lock_yaw = last_att_yaw

        yaw_err = 0.0
        yaw_rate_hold = 0.0
        if tag_lock_yaw is not None and last_att_yaw is not None:
            yaw_err = wrap_pi(tag_lock_yaw - last_att_yaw)
            yaw_rate_hold = float(np.clip(TAG_YAW_KP * yaw_err, -TAG_YAW_RATE_MAX, TAG_YAW_RATE_MAX))

        if lock_elapsed < TAG_BRAKE_S:
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_hold)
            tag_hover_good = 0
            continue

        # If tag not visible, hold position and keep locked heading
        if not tag.get("found", False):
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_hold)
            tag_hover_good = 0
        else:
            last_seen_tag = tag
            h, w = img.shape[:2]
            cx, cy = tag["center"]

            # normalized errors: left=-, right=+, up=-, down=+
            ex = (cx - (w / 2.0)) / (w / 2.0)
            ey = (cy - (h / 2.0)) / (h / 2.0)

            ex_f = tag_x_lpf.update(ex)
            ey_f = tag_y_lpf.update(ey)

            # Map image errors -> body velocities:
            # ex (tag to the right) => move right (vy +)
            vy_cmd = tag_x_pid.update(ex_f, dt_tag)

            # ey (tag lower in image) usually means it's "behind" depending on camera projection,
            # but for a downward camera in many sims:
            # - if tag appears "down" (cy > center), you need to move forward a bit (vx +)
            vx_cmd = tag_y_pid.update(ey_f, dt_tag)

            vx_cmd = tag_vx_slew.update(vx_cmd, dt_tag)
            vy_cmd = tag_vy_slew.update(vy_cmd, dt_tag)

            controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_hold)

            # Check if centered stably
            yaw_ok = abs(yaw_err) < TAG_YAW_TOL_RAD
            if abs(ex_f) < TAG_CENTER_TOL and abs(ey_f) < TAG_CENTER_TOL and yaw_ok:
                tag_hover_good += 1
            else:
                tag_hover_good = 0

            if tag_hover_good >= TAG_HOVER_NEED:
                print(f"AprilTag01 READ ✅ id={tag['id']}")

                # Lock a stop before turning
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)

                # Prepare turn
                turn_started = False
                current_state = DroneState.TURN_RIGHT_90

    if current_state == DroneState.TURN_RIGHT_90:
        msg = master.recv_match(type='ATTITUDE', blocking=False)
        if not msg:
            # keep sending a small yaw until we get attitude
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.3)
        else:
            yaw = float(msg.yaw)  # radians, typically [-pi,pi]

            if not turn_started:
                turn_started = True
                turn_start_yaw = yaw
                turn_target_yaw = wrap_pi(turn_start_yaw - (np.pi / 2.0))  # right turn (sign may need flip)

            err = wrap_pi(turn_target_yaw - yaw)

            # simple proportional yaw-rate with clamp
            yaw_rate_cmd = np.clip(1.2 * err, -0.6, 0.6)
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_cmd)

            if abs(err) < 0.06:  # ~3.5 degrees
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                print("Turn 90° right ✅")
                line_roi_top_ratio_active = LINE_ROI_TOP_RATIO_FOLLOW
                follow_started_at = time.time()
                vx_slew.y = 0.0
                vx_slew.inited = True
                yr_slew.y = 0.0
                yr_slew.inited = True
                yaw_pid.reset()
                current_state = DroneState.FOLLOW_LINE_01
