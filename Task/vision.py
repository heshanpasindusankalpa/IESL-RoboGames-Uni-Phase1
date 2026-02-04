import cv2
import numpy as np
import math

def detect_line(gray_img, thresh_val = 100, min_pixels = 200, band_width_ratio=0.35):
    h, w = gray_img.shape
    half_bw = int((w * band_width_ratio) / 2)
    cx_img = w // 2

    x0 = max(cx_img - half_bw, 0)
    x1 = min(cx_img + half_bw, w)

    roi = gray_img[:, x0:x1]

    _, mask = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphology cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return {
            "found": False,
            "cx": None,
            "cy": None,
            "angle_rad": None,
            "angle_error_rad": None,
            "roi": (x0, x1)
        }

    # Largest contour = line
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < min_pixels:
        return {
            "found": False,
            "cx": None,
            "cy": None,
            "angle_rad": None,
            "angle_error_rad": None,
            "roi": (x0, x1)
        }
    
    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return {"found": False, "roi": (x0, x1)}

    cx_roi = M["m10"] / M["m00"]
    cy_roi = M["m01"] / M["m00"]

    # Convert to full-image coords
    cx = cx_roi + x0
    cy = cy_roi

    # PCA
    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts[:, 0] -= cx_roi
    pts[:, 1] -= cy_roi

    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]

    dx, dy = float(direction[0]), float(direction[1])

    if dy > 0:
        dx, dy = -dx, -dy

    angle_rad = math.atan2(dy, dx)

    forward_angle = -math.pi / 2
    angle_error = wrap_angle(angle_rad - forward_angle)

    return {
        "found": True,
        "cx": cx,
        "cy": cy,
        "angle_rad": angle_rad,
        "angle_error_rad": angle_error,
        "roi": (x0, x1)
    }

def wrap_angle(a):
    # Wrap angle to [-pi, +pi]
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def draw_debug(gray_img, measurement, ref_color=(0, 0, 255), line_color=(0, 255, 0)):
    if gray_img.ndim != 2:
        raise ValueError("draw_debug expects a single-channel greyscale image")
    
    h, w = gray_img.shape
    p0 = (w // 2, h - 1)
    p1 = (w // 2, 0)

    vis_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.line(vis_img, p0, p1, ref_color, 1)

    if not measurement["found"]:
        cv2.putText(vis_img, "LINE NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
        return vis_img
    
    cx = int(measurement["cx"])
    cy = int(measurement["cy"])

    cv2.circle(vis_img, (cx, cy), 2, (255, 0, 0), -1)
    ang = measurement["angle_rad"]
    length = int(min(h, w) * 0.35)
    
    dx = math.cos(ang)
    dy = math.sin(ang)

    if dy > 0:
        dx, dy = dx * -1, dy * -1

    pA = (int(cx - dx * length), int(cy - dy * length))
    pB = (int(cx + dx * length), int(cy + dy * length))
    cv2.line(vis_img, pA, pB, line_color, 1)

    offset_px = cx - (w // 2)
    angle_deg = measurement["angle_error_rad"] * 180.0 / math.pi
    cv2.putText(vis_img, f"offset_px: {offset_px:+d}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(vis_img, f"angle_err: {angle_deg:+.1f} deg", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
    
    return vis_img
        