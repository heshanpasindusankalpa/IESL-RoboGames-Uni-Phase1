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

master = mavutil.mavlink_connection('udp:0.0.0.0:14550')
# master = mavutil.mavlink_connection('tcp:127.0.0.1:5762')

CAM_HOST = "127.0.0.1"
CAM_PORT = 5599

cam_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cam_sock.connect((CAM_HOST, CAM_PORT))
cam_sock.settimeout(0.2)
print("Connected to camera stream")

master.wait_heartbeat()
print(f"Connected: system {master.target_system}, component {master.target_component}")

def set_mode(mode_name: str):
    modes   = master.mode_mapping()
    mode_id = modes.get(mode_name)
    if mode_id is None:
        raise RuntimeError(f"Mode '{mode_name}' not found. Available: {list(modes)}")
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id,
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


def tag_inside_roi(tag_m: dict, roi) -> bool:
    if not tag_m.get("found", False) or roi is None:
        return False
    cx, cy = tag_m["center"]
    x0, x1, y0, y1 = roi
    return (x0 <= cx < x1) and (y0 <= cy < y1)

TARGET_ALT              = 2.5   # m
ALT_TOL                 = 0.2   # m
TAKEOFF_TIMEOUT_S       = 35.0
TAKEOFF_CLIMB_SPEED_MPS = 0.6
TAKEOFF_ACCEL_Z_MPS2    = 0.8
TAKEOFF_MIN_ACCEPT_ALT  = TARGET_ALT - 0.35
HOVER_STABILIZE_S       = 2.0
ARM_TIMEOUT_S           = 5.0

LINE_THRESH             = 155
LINE_ROI_FOLLOW         = 0.30   # top-30 % of image when following
LINE_ROI_FULL           = 1.00   # full image when scanning for tag

TAG2_HOVER_CONFIRM_S    = 3.0   
POST_TURN_STABILIZE_S   = 2.5   

HEADING_ALIGN_TOL_RAD   = math.radians(3.0)

alt_tracker = controls.AltitudeTracker()

line_follower_1 = controls.LineFollower(yaw_sign=+1.0)

LINE2_TUNING = dict(
    lat_kp      = 0.55,  
    lat_kd      = 0.20,   
    yaw_kp      = 0.90,  
    yaw_kd      = 0.40,  
    off_alpha   = 0.15,   
    ang_alpha   = 0.15,   
    yr_slew     = 0.8,    
    vy_slew     = 0.7,    
    angle_deadband = math.radians(7.0),  
)
line_follower_2 = controls.LineFollower(yaw_sign=-1.0, tuning=LINE2_TUNING)

tag_ctrl = controls.TagHoverController()

current_state      = DroneState.INITIALIZING

arm_requested_at   = None
takeoff_started_at = None
hover_started_at   = None
last_alt_log_t     = 0.0
ALT_LOG_DT         = 0.5

line_roi_active    = LINE_ROI_FOLLOW

turn_base_yaw   = None
turn_target_yaw = None
turn_phase      = None

tag2_centered_at = None


scan_samples: list[dict] = []  


prelanding_target_yaw = None

post_turn_hover_started_at = None

LINE2_TAG_IGNORE_S = 3.0
line2_started_at   = None

# Telemetry cache
att_yaw        = None
att_pitch      = None
att_pitch_rate = None

# Last camera frame (used if stream times out)
last_img = None

while True:
    try:
        header        = recv_exact(cam_sock, 4)
        width, height = struct.unpack("<HH", header)
        payload       = recv_exact(cam_sock, width * height)
        img           = np.frombuffer(payload, dtype=np.uint8).reshape((height, width))
        last_img      = img
    except TimeoutError:
        img = last_img
    except ConnectionError as e:
        print(f"Camera error: {e}")
        break
    if img is None:
        continue

    img_h, img_w = img.shape[:2]

    msg = master.recv_match(type="ATTITUDE", blocking=False)
    while msg is not None:
        att_yaw        = float(msg.yaw)
        att_pitch      = float(msg.pitch)
        att_pitch_rate = float(msg.pitchspeed)
        msg = master.recv_match(type="ATTITUDE", blocking=False)

    tag = vision.detect_apriltag(img)

    line_states = (DroneState.FOLLOW_LINE_01, DroneState.FOLLOW_LINE_02)
    if current_state in line_states:
        line_m = vision.detect_line(img, thresh_val=LINE_THRESH,
                                    roi_top_ratio=line_roi_active)
        vis    = vision.draw_debug(img, line_m)
    else:
        line_m = {"found": False, "cx": None, "cy": None,
                  "angle_rad": None, "angle_error_rad": None,
                  "roi": (0, img_w, 0, img_h)}
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        vision.draw_tag_overlay(vis, tag)

    cv2.putText(vis, current_state.name, (20, img_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

    cv2.imshow("Webots Camera", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("p"):
        vision.dump_gray_values(img, LINE_THRESH)

    # STATE MACHINE
    # ── INITIALIZING ──────────────────────────────────────────────────────────
    if current_state == DroneState.INITIALIZING:
        set_mode("GUIDED")
        if arm_requested_at is None:
            controls.force_arm(master)
            arm_requested_at = time.time()
            print("Arm requested…")
        if controls.is_armed(master):
            print("Armed ✅")
            current_state = DroneState.ARMED
        elif (time.time() - arm_requested_at) > ARM_TIMEOUT_S:
            print("Arming timeout – retrying")
            arm_requested_at = None

    # ── ARMED ─────────────────────────────────────────────────────────────────
    if current_state == DroneState.ARMED:
        controls.set_param(master, "WPNAV_SPEED_UP",
                           TAKEOFF_CLIMB_SPEED_MPS * 100.0)
        controls.set_param(master, "WPNAV_ACCEL_Z",
                           TAKEOFF_ACCEL_Z_MPS2 * 100.0)
        time.sleep(0.2)
        controls.takeoff(master, TARGET_ALT)
        alt_tracker.reset()
        takeoff_started_at = time.time()
        current_state      = DroneState.TAKEOFF

    # ── TAKEOFF ───────────────────────────────────────────────────────────────
    if current_state == DroneState.TAKEOFF:
        reached, current_alt = alt_tracker.update(master, TARGET_ALT, ALT_TOL)
        now = time.time()
        if current_alt is not None and (now - last_alt_log_t) >= ALT_LOG_DT:
            print(f"  alt: {current_alt:.2f} m")
            last_alt_log_t = now

        if reached:
            print("Target altitude reached → HOVER")
            hover_started_at = time.time()
            current_state    = DroneState.HOVER
        elif (now - takeoff_started_at) > TAKEOFF_TIMEOUT_S:
            if current_alt is not None and current_alt >= TAKEOFF_MIN_ACCEPT_ALT:
                print("Takeoff timeout – altitude acceptable → HOVER")
                hover_started_at = time.time()
                current_state    = DroneState.HOVER
            else:
                print("Takeoff timeout – altitude too low → LAND")
                set_mode("LAND")
                current_state = DroneState.LANDING

    # ── HOVER ─────────────────────────────────────────────────────────────────
    if current_state == DroneState.HOVER:
        if (time.time() - hover_started_at) >= HOVER_STABILIZE_S:
            print("Hover stable → FOLLOW_LINE_01")
            line_roi_active = LINE_ROI_FOLLOW
            line_follower_1.reset()
            current_state = DroneState.FOLLOW_LINE_01

    # ── FOLLOW_LINE_01 ────────────────────────────────────────────────────────
    if current_state == DroneState.FOLLOW_LINE_01:
        roi = line_m.get("roi")
        if tag_inside_roi(tag, roi):
            print(f"[LINE01] Tag seen id={tag['id']} → APRILTAG_01_DETECTED")
            line_roi_active = LINE_ROI_FULL
            tag_ctrl.enter(lock_yaw=att_yaw,
                           entry_vx=line_follower_1.last_vx,
                           entry_vy=line_follower_1.last_vy)
            current_state = DroneState.APRILTAG_01_DETECTED
            continue

        line_follower_1.update(master, line_m, img_w,
                               att_pitch=att_pitch,
                               att_pitch_rate=att_pitch_rate)

    # ── APRILTAG_01_DETECTED ──────────────────────────────────────────────────
    if current_state == DroneState.APRILTAG_01_DETECTED:
        scan_samples.clear()
        current_state = DroneState.SCAN_APRILTAG_01

    # ── SCAN_APRILTAG_01 ──────────────────────────────────────────────────────
    if current_state == DroneState.SCAN_APRILTAG_01:
        good = tag_ctrl.update(master, tag, img,
                               att_yaw=att_yaw,
                               att_pitch=att_pitch,
                               att_pitch_rate=att_pitch_rate)
        # Accumulate readings – only print when centred
        if tag.get("found"):
            cx, cy = tag["center"]
            h, w   = img.shape[:2]
            scan_samples.append({
                "ex": (cx - w / 2.0) / (w / 2.0),
                "ey": (cy - h / 2.0) / (h / 2.0),
                "cx": cx, "cy": cy,
            })
        if good >= controls.TagHoverController.HOVER_NEED:
            if scan_samples:
                avg_cx = sum(s["cx"] for s in scan_samples) / len(scan_samples)
                avg_cy = sum(s["cy"] for s in scan_samples) / len(scan_samples)
                avg_ex = sum(s["ex"] for s in scan_samples) / len(scan_samples)
                avg_ey = sum(s["ey"] for s in scan_samples) / len(scan_samples)
                print()
                print(f"[TAG 01 SCAN RESULT]  id={tag['id']}")
                print()
            scan_samples.clear()
            print(f"Tag 01 centred ✅ id={tag['id']} → TURN_RIGHT_90")
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)

            turn_base_yaw   = tag_ctrl.lock_yaw
            turn_target_yaw = wrap_pi(turn_base_yaw + math.pi / 2.0)

            heading_err = (abs(wrap_pi(turn_base_yaw - att_yaw))
                           if att_yaw is not None else 0.0)

            if heading_err > HEADING_ALIGN_TOL_RAD:
                turn_phase = "align"
                print(f"Heading drift {math.degrees(heading_err):.1f}° — "
                      f"realigning to {math.degrees(turn_base_yaw):.1f}° first")
            else:
                turn_phase = "turn"
                print(f"Heading OK ({math.degrees(heading_err):.1f}° drift) — "
                      f"turning right 90° → target {math.degrees(turn_target_yaw):.1f}°")

            current_state = DroneState.TURN_RIGHT_90

    # ── TURN_RIGHT_90 ─────────────────────────────────────────────────────────
    if current_state == DroneState.TURN_RIGHT_90:
        fresh = master.recv_match(type="ATTITUDE", blocking=False)
        if fresh:
            att_yaw = float(fresh.yaw)

        if att_yaw is None:
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)

        elif turn_phase == "align":
            # Rotate back to the original line-following heading
            err          = wrap_pi(turn_base_yaw - att_yaw)
            yaw_rate_cmd = float(np.clip(1.2 * err, -0.3, 0.3))
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_cmd)

            if abs(err) < HEADING_ALIGN_TOL_RAD:
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                turn_phase = "turn"
                print(f"Heading realigned — turning right 90° → "
                      f"target {math.degrees(turn_target_yaw):.1f}°")

        elif turn_phase == "turn":
            # Rotate 90° clockwise from the corrected heading
            err          = wrap_pi(turn_target_yaw - att_yaw)
            yaw_rate_cmd = float(np.clip(1.2 * err, -0.6, 0.6))
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_cmd)

            if abs(err) < math.radians(3.5):
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                print(f"Turn complete... → POST_TURN_HOVER "
                      f"(stabilizing {POST_TURN_STABILIZE_S:.1f}s)")
                post_turn_hover_started_at = time.time()
                current_state = DroneState.POST_TURN_HOVER
                continue

    # ── POST_TURN_HOVER ───────────────────────────────────────────────────────
    if current_state == DroneState.POST_TURN_HOVER:
        controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
        if (time.time() - post_turn_hover_started_at) >= POST_TURN_STABILIZE_S:
            print("Post-turn stable ✅ → FOLLOW_LINE_02")
            line_roi_active  = LINE_ROI_FOLLOW
            line2_started_at = time.time()
            line_follower_2.reset()
            current_state = DroneState.FOLLOW_LINE_02
            continue   

    # ── FOLLOW_LINE_02 ────────────────────────────────────────────────────────
    if current_state == DroneState.FOLLOW_LINE_02:
        roi = line_m.get("roi")
        line2_elapsed = (time.time() - line2_started_at) if line2_started_at else LINE2_TAG_IGNORE_S
        if line2_elapsed >= LINE2_TAG_IGNORE_S and tag_inside_roi(tag, roi):
            print(f"[LINE02] Tag seen id={tag['id']} → APRILTAG_02_DETECTED")
            line_roi_active  = LINE_ROI_FULL
            tag2_centered_at = None
            tag_ctrl.enter(lock_yaw=att_yaw,
                           entry_vx=line_follower_2.last_vx,
                           entry_vy=line_follower_2.last_vy)
            current_state = DroneState.APRILTAG_02_DETECTED
            continue

        line_follower_2.update(master, line_m, img_w,
                               att_pitch=att_pitch,
                               att_pitch_rate=att_pitch_rate)

    # ── APRILTAG_02_DETECTED ──────────────────────────────────────────────────
    if current_state == DroneState.APRILTAG_02_DETECTED:
        scan_samples.clear()
        current_state = DroneState.SCAN_APRILTAG_02

    # ── SCAN_APRILTAG_02 ──────────────────────────────────────────────────────
    if current_state == DroneState.SCAN_APRILTAG_02:
        good = tag_ctrl.update(master, tag, img,
                               att_yaw=att_yaw,
                               att_pitch=att_pitch,
                               att_pitch_rate=att_pitch_rate)
        now = time.time()

        # Accumulate readings – only print when centred
        if tag.get("found"):
            cx, cy = tag["center"]
            h, w   = img.shape[:2]
            scan_samples.append({
                "ex": (cx - w / 2.0) / (w / 2.0),
                "ey": (cy - h / 2.0) / (h / 2.0),
                "cx": cx, "cy": cy,
            })

        if good >= controls.TagHoverController.HOVER_NEED:
            if tag2_centered_at is None:
                tag2_centered_at = now
                # ── Print single summary of the scan ──────────────────────────
                if scan_samples:
                    avg_cx = sum(s["cx"] for s in scan_samples) / len(scan_samples)
                    avg_cy = sum(s["cy"] for s in scan_samples) / len(scan_samples)
                    avg_ex = sum(s["ex"] for s in scan_samples) / len(scan_samples)
                    avg_ey = sum(s["ey"] for s in scan_samples) / len(scan_samples)
                    print()
                    print(f"[TAG 02 SCAN RESULT]  id={tag['id']}")
                    print()
                scan_samples.clear()
                print(f"Tag 02 centred ✅ id={tag['id']} "
                      f"— holding {TAG2_HOVER_CONFIRM_S:.0f}s before realign…")
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
            elif (now - tag2_centered_at) >= TAG2_HOVER_CONFIRM_S:
                prelanding_target_yaw = turn_target_yaw
                print(f"Hover confirmed → REALIGN_BEFORE_LAND "
                      f"(target {math.degrees(prelanding_target_yaw):.1f}° — line-2 heading)")
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                current_state = DroneState.REALIGN_BEFORE_LAND
        else:
            tag2_centered_at = None   # lost centre; restart the timer

    # ── REALIGN_BEFORE_LAND ───────────────────────────────────────────────────
    if current_state == DroneState.REALIGN_BEFORE_LAND:
        fresh = master.recv_match(type="ATTITUDE", blocking=False)
        if fresh:
            att_yaw = float(fresh.yaw)

        if att_yaw is None or prelanding_target_yaw is None:
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
        else:
            err          = wrap_pi(prelanding_target_yaw - att_yaw)
            yaw_rate_cmd = float(np.clip(1.0 * err, -0.5, 0.5))
            controls.send_body_velocity(master, 0.0, 0.0, 0.0, yaw_rate_cmd)

            if abs(err) < HEADING_ALIGN_TOL_RAD:
                controls.send_body_velocity(master, 0.0, 0.0, 0.0, 0.0)
                # ── Print final tag values before landing ──────────────────────
                if tag.get("found"):
                    cx, cy = tag["center"]
                    h, w   = img.shape[:2]
                    ex = (cx - w / 2.0) / (w / 2.0)
                    ey = (cy - h / 2.0) / (h / 2.0)
                    print(f"[PRE-LAND TAG]  id={tag['id']}  "
                          f"center=({cx:.1f},{cy:.1f})  "
                          f"ex={ex:+.3f}  ey={ey:+.3f}  "
                          f"heading={math.degrees(att_yaw):.1f}°")
                else:
                    print(f"[PRE-LAND TAG]  not visible — "
                          f"heading={math.degrees(att_yaw):.1f}°")
                print("Heading realigned ✅ → LANDING")
                set_mode("LAND")
                current_state = DroneState.LANDING

    # ── LANDING ───────────────────────────────────────────────────────────────
    if current_state == DroneState.LANDING:
        if not controls.is_armed(master):
            print("Landed and disarmed ✅ → DONE")
            current_state = DroneState.DONE

    # ── DONE ──────────────────────────────────────────────────────────────────
    if current_state == DroneState.DONE:
        break


cv2.destroyAllWindows()
cam_sock.close()
print("Mission complete.")