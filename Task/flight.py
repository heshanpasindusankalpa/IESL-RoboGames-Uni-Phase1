import time
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
        mavutil.mavlink.
        MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
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


current_state = DroneState.INITIALIZING

ARM_TIMEOUT_S = 5.0
arm_requested_at = None
last_img = None
last_frame_t = 0.0
TARGET_ALT = 2.0 # meters
ALT_TOL = 0.2 # meters
TAKEOFF_TIMEOUT_S = 20.0
alt_tracker = controls.AltitudeTracker()
takeoff_started_at = None
HOVER_STABILIZE_S = 2.0
hover_started_at = None

# --- Line following ---
FOLLOW_FWD_V = 0.8        # m/s forward cruise
MAX_LAT_V = 0.6           # m/s
MAX_YAW_RATE = 0.8        # rad/s
CMD_HZ = 20.0             # command rate
CMD_DT = 1.0 / CMD_HZ

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

    m = vision.detect_line(img)
    vis = vision.draw_debug(img, m)

    cv2.imshow("Webots Camera", vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
        controls.takeoff(master, TARGET_ALT)

        alt_tracker.stable_count = 0
        alt_tracker.last_alt = None
        takeoff_started_at = time.time()

        current_state = DroneState.TAKEOFF

    if current_state == DroneState.TAKEOFF:
        if takeoff_started_at and (time.time() - takeoff_started_at) > TAKEOFF_TIMEOUT_S:
            print("TAKEOFF timeout — switching to LAND for safety")
            set_mode('LAND')
            current_state = DroneState.LANDING

        else:
            reached, current_alt = alt_tracker.update(master, TARGET_ALT, ALT_TOL)
            if current_alt is not None:
                print(f"Altitude: {current_alt:.2f} m")

            if reached:
                print("Reached target altitude — entering HOVER")
                current_state = DroneState.HOVER
                hover_started_at = time.time()

    if current_state == DroneState.HOVER:
        if (time.time() - hover_started_at >= HOVER_STABILIZE_S):
            print("Hover stabilized — ready for FOLLOW_LINE_01")
            current_state = DroneState.FOLLOW_LINE_01

    if current_state == DroneState.FOLLOW_LINE_01:
        now = time.time()
        if (now - last_cmd_t) < CMD_DT:
            continue
        dt = (now - last_cmd_t) if last_cmd_t > 0 else CMD_DT
        last_cmd_t = now

        h, w = img.shape[:2]

        if not m.get("found", False):
            # If line lost: slow forward, gently yaw to search 
            vx_cmd = 0.2
            vy_cmd = 0.0
            yaw_rate_cmd = 0.35   # gentle spin to reacquire
        else:
            # Normalize offset: -1..+1 (left..right)
            offset_px = (m["cx"] - (w / 2.0))
            offset_n = offset_px / (w / 2.0)

            ang_err = float(m["angle_error_rad"])

            # Filter measurements
            offset_n_f = off_lpf.update(offset_n)
            ang_err_f = ang_lpf.update(ang_err)

            # Controllers
            vy_cmd = lat_pid.update(offset_n_f, dt)

            yaw_rate_cmd = -yaw_pid.update(ang_err_f, dt)

            # Forward speed scheduling (slow down when large error, for stability)
            err_mag = min(1.0, abs(offset_n_f) + 0.6 * abs(ang_err_f))
            vx_cmd = FOLLOW_FWD_V * (1.0 - 0.5 * err_mag)
            if vx_cmd < 0.25:
                vx_cmd = 0.25

        # Slew-limit commands 
        vx_cmd = vx_slew.update(vx_cmd, dt)
        vy_cmd = vy_slew.update(vy_cmd, dt)
        yaw_rate_cmd = yr_slew.update(yaw_rate_cmd, dt)

        # Keep vz ~ 0 so autopilot holds altitude
        controls.send_body_velocity(master, vx_cmd, vy_cmd, 0.0, yaw_rate_cmd)
