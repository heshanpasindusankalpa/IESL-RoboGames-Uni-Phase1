import cv2
import numpy as np
import math

def _wrap(a: float) -> float:
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def detect_line(gray_img, thresh_val=100, min_pixels=200,
                band_width_ratio=0.35, roi_top_ratio=0.30) -> dict:
    h, w = gray_img.shape
    half_bw = int((w * band_width_ratio) / 2)
    cx_img  = w // 2
    roi_top_ratio = max(0.01, min(float(roi_top_ratio), 1.0))
    y0 = 0
    y1 = max(1, int(h * roi_top_ratio))
    x0 = max(cx_img - half_bw, 0)
    x1 = min(cx_img + half_bw, w)

    roi = gray_img[y0:y1, x0:x1]
    _, mask = cv2.threshold(roi, thresh_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    no_find = {"found": False, "cx": None, "cy": None,
               "angle_rad": None, "angle_error_rad": None,
               "roi": (x0, x1, y0, y1)}

    if not contours:
        return no_find

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_pixels:
        return no_find

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return no_find

    cx_roi = M["m10"] / M["m00"]
    cy_roi = M["m01"] / M["m00"]
    cx     = cx_roi + x0
    cy     = cy_roi + y0

    # PCA to find line orientation
    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts[:, 0] -= cx_roi
    pts[:, 1] -= cy_roi
    cov      = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    direction = eigvecs[:, np.argmax(eigvals)]
    dx, dy = float(direction[0]), float(direction[1])
    if dy > 0:
        dx, dy = -dx, -dy

    angle_rad   = math.atan2(dy, dx)
    angle_error = _wrap(angle_rad - (-math.pi / 2))   # deviation from vertical

    return {
        "found": True,
        "cx": cx, "cy": cy,
        "angle_rad": angle_rad,
        "angle_error_rad": angle_error,
        "roi": (x0, x1, y0, y1),
    }


def detect_apriltag(gray_img) -> dict:
    try:
        aruco = cv2.aruco
    except AttributeError:
        return {"found": False, "id": None, "center": None, "corners": None}

    dict_id = getattr(aruco, "DICT_APRILTAG_36h11", None)
    if dict_id is None:
        return {"found": False, "id": None, "center": None, "corners": None}

    dictionary = aruco.getPredefinedDictionary(dict_id)
    params     = aruco.DetectorParameters()

    if hasattr(aruco, "ArucoDetector"):
        detector         = aruco.ArucoDetector(dictionary, params)
        corners, ids, _  = detector.detectMarkers(gray_img)
    else:
        corners, ids, _  = aruco.detectMarkers(gray_img, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return {"found": False, "id": None, "center": None, "corners": None}

    tag_id = int(ids[0][0])
    c      = corners[0].reshape(4, 2)
    cx     = float(c[:, 0].mean())
    cy     = float(c[:, 1].mean())
    return {"found": True, "id": tag_id, "center": (cx, cy), "corners": c}


def draw_debug(gray_img, line_m: dict,
               ref_color=(0, 0, 255), line_color=(0, 255, 0)):
    if gray_img.ndim != 2:
        raise ValueError("draw_debug expects a single-channel greyscale image")

    h, w   = gray_img.shape
    vis    = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    # Centre reference line
    cv2.line(vis, (w // 2, h - 1), (w // 2, 0), ref_color, 1)

    roi = line_m.get("roi")
    if roi is not None:
        x0, x1, y0, y1 = roi
        cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1) - 1), (80, 80, 80), 1)

    if not line_m["found"]:
        cv2.putText(vis, "LINE NOT FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
        return vis

    cx  = int(line_m["cx"])
    cy  = int(line_m["cy"])
    ang = line_m["angle_rad"]
    cv2.circle(vis, (cx, cy), 2, (255, 0, 0), -1)

    length = int(min(h, w) * 0.35)
    dx, dy = math.cos(ang), math.sin(ang)
    if dy > 0:
        dx, dy = -dx, -dy
    pA = (int(cx - dx * length), int(cy - dy * length))
    pB = (int(cx + dx * length), int(cy + dy * length))
    cv2.line(vis, pA, pB, line_color, 1)

    offset_px = cx - (w // 2)
    angle_deg = line_m["angle_error_rad"] * 180.0 / math.pi
    cv2.putText(vis, f"offset_px: {offset_px:+d}",      (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(vis, f"angle_err: {angle_deg:+.1f} deg", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return vis


def draw_tag_overlay(bgr_img, tag: dict) -> None:
    if not tag.get("found", False):
        cv2.putText(bgr_img, "TAG SEARCH", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        return
    corners = tag.get("corners")
    if corners is not None:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(bgr_img, [pts], True, (0, 255, 0), 2)
    cx, cy = tag["center"]
    cv2.circle(bgr_img, (int(cx), int(cy)), 4, (0, 255, 255), -1)
    cv2.putText(bgr_img, f"TAG id={tag['id']}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


def dump_gray_values(gray_img, thresh_val: int):
    img = gray_img.astype(np.uint8)
    print("\n=== GRAYSCALE FRAME DUMP ===")
    print(f"shape={img.shape}  min={img.min()}  max={img.max()}  mean={img.mean():.2f}")
    print(f"below {thresh_val}: {(img < thresh_val).sum()}  "
          f"above: {(img >= thresh_val).sum()}")
    old = np.get_printoptions()
    np.set_printoptions(threshold=np.inf, linewidth=100000)
    print(img)
    np.set_printoptions(**old)