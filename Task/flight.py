import time
import socket
import struct
from states import DroneState
import numpy as np
import cv2
from pymavlink import mavutil
import controls

# Establish connections
master = mavutil.mavlink_connection('udp:0.0.0.0:14550')

CAM_HOST = "127.0.0.1"
CAM_PORT = 5599

cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cam_sock.connect((CAM_HOST, CAM_PORT))
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
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("Camera stream closed")
        data += chunk
    return data


current_state = DroneState.INITIALIZING

TARGET_ALT = 2.0 # meters
ALT_TOL = 0.2 # meters
TAKEOFF_TIMEOUT_S = 20.0
alt_tracker = controls.AltitudeTracker()
takeoff_started_at = None

while True:
    header = recv_exact(cam_sock, 4)
    width, height = struct.unpack("<HH", header)

    payload = recv_exact(cam_sock, width * height)
    img = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))

    cv2.imshow("Webots Camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if current_state == DroneState.INITIALIZING:
        set_mode('GUIDED')
        if controls.force_arm(master):
            current_state = DroneState.ARMED
        else:
            current_state = DroneState.INITIALIZING

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

    if current_state == DroneState.HOVER:
        time.sleep(5)
        current_state = DroneState.LANDING
        set_mode('LAND')
        break