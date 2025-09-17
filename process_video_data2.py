# process_video_data2.py
# -----------------------------------
# Per-video calibration + measurement pipeline with bending-angle tracking.
# - Tip tracked via a BLACK fiducial found INSIDE the filled contour.
# - Base center is locked from the first valid frame using contour geometry.
# Usage:
#   python process_video_data2.py <video.mp4> --out <output_dir>
# -----------------------------------

import os
import cv2
import csv
import math
import argparse
import numpy as np

# ---------------- CONFIG ----------------
# Chessboard for mm/px scale
BOARD_SQUARES       = (5, 5)                   # physical squares on printed board
BOARD_INNER         = (BOARD_SQUARES[0]-1, BOARD_SQUARES[1]-1)
SQUARE_MM           = 10.0                     # edge length of one square, in mm

DETECTIONS_REQ      = 3
FRAME_STRIDE        = 2
MAX_CALIB_FRAMES    = 100
SCALE_DEF           = 3.7
UNDISTORT           = False
DURATION_S          = 3.5
TRIM_THRESHOLD_PX   = 5

# Body color (magenta) mask in HSV
COLOUR_RANGE_LO     = np.array([150, 180, 180],  dtype=np.uint8)
COLOUR_RANGE_HI     = np.array([180, 255, 255], dtype=np.uint8)
KERNEL              = np.ones((5,5), np.uint8)

# Tip dot (BLACK) detection inside contour
TIP_BLACK_THRESH    = 80            # grayscale threshold; <= this is "black"
TIP_TOP_FRAC        = 0.45          # search only in top 45% of the contour height
TIP_MIN_AREA_PX     = 4             # absolute min area to reject specks
TIP_MAX_AREA_FRAC   = 0.1           # max tip area as fraction of contour area
TIP_OPEN_K          = 3             # small open to denoise the dot mask

# Angle smoothing (EMA)
ANGLE_EMA_ALPHA     = 0.2

# Video codecs to try (in order)
CODECS = [("avc1",".mp4"), ("mp4v",".mp4"), ("MJPG",".avi")]

# ---------------- Helpers ----------------
def undistort(img, mtx, dist):
    if mtx is None or dist is None:
        return img
    h, w = img.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, mtx, dist, None, new_mtx)

def _robust_mm_per_px_from_corners(corners, square_mm: float) -> float | None:
    if corners is None or len(corners) == 0:
        return None
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])
    pts = corners.reshape(-1, 2)
    dists = []
    for r in range(rows_inner):
        base = r * cols_inner
        for c in range(cols_inner - 1):
            p0 = pts[base + c]
            p1 = pts[base + c + 1]
            dists.append(np.hypot(*(p1 - p0)))
    for c in range(cols_inner):
        for r in range(rows_inner - 1):
            p0 = pts[r * cols_inner + c]
            p1 = pts[(r + 1) * cols_inner + c]
            dists.append(np.hypot(*(p1 - p0)))
    if not dists:
        return None
    px_mean = float(np.median(dists))
    return square_mm / px_mean if px_mean > 1e-6 else None

def calibrate_from_video(cap):
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])
    objp = np.zeros((rows_inner * cols_inner, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols_inner, 0:rows_inner].T.reshape(-1, 2) * SQUARE_MM

    objpoints, imgpoints = [], []
    scale_estimates = []

    origin = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or MAX_CALIB_FRAMES

    detections = 0
    gray = None

    for fidx in range(0, min(total_frames, MAX_CALIB_FRAMES), FRAME_STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols_inner, rows_inner))
        if not found:
            continue
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        detections += 1

        scale_i = _robust_mm_per_px_from_corners(corners, SQUARE_MM)
        if scale_i is not None:
            scale_estimates.append(scale_i)
        if detections >= DETECTIONS_REQ:
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, origin)
    if not objpoints or gray is None:
        return False, None, None, None

    h, w = gray.shape[:2]
    ok, mtx, dist, *_ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    scale = float(np.median(scale_estimates)) if scale_estimates else None
    return bool(ok), mtx, dist, scale

def open_writer(sample_frame, base_out):
    h, w = sample_frame.shape[:2]
    for fourcc, ext in CODECS:
        out_path = base_out + ext
        four = cv2.VideoWriter_fourcc(*fourcc)
        vw = cv2.VideoWriter(out_path, four, 30, (w, h))
        if vw.isOpened():
            print(f"[INFO] Using codec {fourcc} → {out_path}")
            return vw, out_path
        print(f"[WARN] Codec {fourcc} failed, trying next...")
    raise RuntimeError("Could not open any VideoWriter")

# ---- Tip detection INSIDE the contour (black dot) ----
def find_black_tip_in_contour(frame_bgr, contour, tip_black_thresh, top_frac,
                              min_area_px, max_area_frac):
    """
    Returns (tip_x, tip_y) in full-frame coords or None.
    Works on the FULL FRAME but masked to the FILLED contour (no bbox dependence).
    Optionally restricts to the top fraction of the contour height.
    """
    H, W = frame_bgr.shape[:2]
    mask_full = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask_full, [contour], -1, 255, cv2.FILLED)

    # Optional: keep only top frac of contour
    pts = contour.reshape(-1, 2)
    ymin, ymax = float(pts[:,1].min()), float(pts[:,1].max())
    height = ymax - ymin
    y_cut = int(ymin + top_frac * height)
    top_mask = np.zeros_like(mask_full)
    top_mask[:max(y_cut, 0), :] = 255
    mask_top_contour = cv2.bitwise_and(mask_full, top_mask)

    # Grayscale & mask
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_in = cv2.bitwise_and(gray, gray, mask=mask_top_contour)

    # Threshold: black → white
    _, bin_inv = cv2.threshold(gray_in, tip_black_thresh, 255, cv2.THRESH_BINARY_INV)

    # Clean small specks
    if TIP_OPEN_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (TIP_OPEN_K, TIP_OPEN_K))
        bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    body_area = cv2.contourArea(contour)
    max_area_px = max_area_frac * body_area

    # Filter by area and choose the *highest* (smallest y) candidate
    candidates = []
    for cc in cnts:
        a = cv2.contourArea(cc)
        if a < min_area_px or a > max_area_px:
            continue
        M = cv2.moments(cc)
        if M["m00"] <= 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        candidates.append((cx, cy, a))

    if not candidates:
        return None

    # pick the one closest to the tip (smallest y)
    cx, cy, _ = min(candidates, key=lambda t: t[1])
    return float(cx), float(cy)

# ---------------- Main processing ----------------
def process(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps

    # --- calibration from this video ---
    ok, mtx, dist, mm_per_px_video = calibrate_from_video(cap)
    if not ok:
        print(f"[WARN] Calibration failed; using fallback scale {SCALE_DEF:.6f} mm/px.")
        mtx, dist = None, None
        scale = float(SCALE_DEF)
    else:
        scale = float(mm_per_px_video) if mm_per_px_video else float(SCALE_DEF)
        print(f"[INFO] Calibration OK. mm/px = {scale:.6f}")

    # --- measurement loop ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vw = None

    # extrema & history
    found_any = False
    min_w = min_h = math.inf
    max_w = max_h = 0.0
    min_area = math.inf
    max_area = 0.0

    prev_lin_mm = 0.0
    prev_rad_mm = 0.0
    prev_area_mm = 0.0

    # base lock (contour-based)
    base_cx_locked = None
    base_cy_locked = None

    per_frame = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if UNDISTORT and (mtx is not None) and (dist is not None):
            frame = undistort(frame, mtx, dist)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, COLOUR_RANGE_LO, COLOUR_RANGE_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lin_mm = prev_lin_mm
        rad_mm = prev_rad_mm
        area_mm = prev_area_mm

        if cnts:
            # Largest body contour
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)  # still used for legacy width/height metrics
            area = float(cv2.contourArea(c))
            pts = c.reshape(-1, 2)

            # ---- Lock base center on first valid frame (contour-only) ----
            if base_cx_locked is None:
                ymin, ymax = float(pts[:,1].min()), float(pts[:,1].max())
                height = ymax - ymin
                band_h = max(6.0, 0.12 * height)            # bottom 12% (min 6 px)
                base_band = pts[pts[:,1] >= (ymax - band_h)]
                if len(base_band) >= 5:
                    base_cx_locked = float(np.mean(base_band[:,0]))
                    base_cy_locked = float(np.mean(base_band[:,1]))
                else:
                    # Fallback without bbox: use bottom-most slice
                    tol = 2.0
                    bottom_pts = pts[np.abs(pts[:,1] - ymax) <= tol]
                    if len(bottom_pts) >= 1:
                        base_cx_locked = float(np.mean(bottom_pts[:,0]))
                        base_cy_locked = float(np.mean(bottom_pts[:,1]))
                    else:
                        # final fallback: contour centroid projected to base y
                        M = cv2.moments(c)
                        cx = (M["m10"]/M["m00"]) if M["m00"] else (x + w/2.0)
                        base_cx_locked = float(cx)
                        base_cy_locked = float(ymax)

            base_cx = base_cx_locked
            base_cy = base_cy_locked

            # ---- Tip: BLACK dot inside contour; fallback to topmost contour point ----
            tip_xy = find_black_tip_in_contour(
                frame, c, TIP_BLACK_THRESH, TIP_TOP_FRAC, TIP_MIN_AREA_PX, TIP_MAX_AREA_FRAC
            )
            if tip_xy is not None:
                tip = np.array(tip_xy, dtype=float)
            else:
                tip_idx = np.argmin(pts[:, 1])
                tip = pts[tip_idx].astype(float)

            # ---- Bending angle (base -> tip) ----
            dx = tip[0] - base_cx
            dy = base_cy - tip[1]           # screen y downward → invert
            bend_rad = math.atan2(dx, dy)   # 0 = vertical; sign: + right, - left
            bend_deg = math.degrees(bend_rad)

            if frame_idx == 0:
                bend_deg_smooth = bend_deg
            else:
                bend_deg_prev = per_frame[-1].get("bend_deg_smooth", bend_deg)
                bend_deg_smooth = ANGLE_EMA_ALPHA * bend_deg + (1 - ANGLE_EMA_ALPHA) * bend_deg_prev

            # initialize extrema on first detection
            if not found_any:
                min_w = max_w = float(w)
                min_h = max_h = float(h)
                min_area = max_area = float(area)
                found_any = True

            # update extrema
            min_w, max_w = min(min_w, w), max(max_w, w)
            min_h, max_h = min(min_h, h), max(max_h, h)
            min_area, max_area = min(min_area, area), max(max_area, area)

            # deltas relative to current minima
            lin_px  = float(h - min_h)
            rad_px  = float(w - min_w)
            area_px = float(area - min_area)

            # convert to mm / mm^2
            lin_mm  = lin_px  * scale
            rad_mm  = rad_px  * scale
            area_mm = area_px * (scale ** 2)

            # velocities
            lin_vel_mm  = (lin_mm  - prev_lin_mm)  / dt if frame_idx else 0.0
            rad_vel_mm  = (rad_mm  - prev_rad_mm)  / dt if frame_idx else 0.0
            area_vel_mm = (area_mm - prev_area_mm) / dt if frame_idx else 0.0

            # store record
            per_frame.append({
                "frame": frame_idx,
                "x": x, "y": y, "w": w, "h": h,             # kept for legacy metrics/plots
                "area_px": area_px,
                "lin_px": lin_px,
                "rad_px": rad_px,
                "lin_mm": lin_mm,
                "rad_mm": rad_mm,
                "area_mm": area_mm,
                "lin_vel_mm": lin_vel_mm,
                "rad_vel_mm": rad_vel_mm,
                "area_vel_mm": area_vel_mm,
                "bend_deg": bend_deg,
                "bend_deg_smooth": bend_deg_smooth,
                "tip_x": float(tip[0]),
                "tip_y": float(tip[1]),
                "base_cx": base_cx,
                "base_cy": base_cy
            })

            # ---- overlay viz (draw contour mask, NOT bbox-based tip search) ----
            overlay = frame.copy()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.drawContours(overlay, [c], -1, (0,0,255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.circle(frame, (int(base_cx), int(base_cy)), 5, (0, 255, 255), -1)
            cv2.circle(frame, (int(tip[0]), int(tip[1])), 5, (255, 255, 0), -1)
            cv2.line(frame, (int(base_cx), int(base_cy)), (int(tip[0]), int(tip[1])), (255, 255, 255), 2)
            cv2.putText(frame, f"bend: {bend_deg_smooth:+.1f} deg",
                        (int(min(pts[:,0])), max(0, int(min(pts[:,1])) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # lazy-open writer once we know frame dims
        if vw is None:
            vw, out_path = open_writer(frame, os.path.join(out_dir, f"{name}__annot"))
        vw.write(frame)

        # update previous for next iteration
        prev_lin_mm  = lin_mm
        prev_rad_mm  = rad_mm
        prev_area_mm = area_mm
        frame_idx += 1

    cap.release()
    if vw:
        vw.release()

    if not found_any or not per_frame:
        print("[WARN] No valid contours found; nothing to write.")
        return

    # maxima in mm (based on extrema we tracked)
    max_lin_mm  = float((max_h - min_h) * scale) if max_h >= min_h else 0.0
    max_rad_mm  = float((max_w - min_w) * scale) if max_w >= min_w else 0.0
    max_area_mm = float((max_area - min_area) * (scale ** 2)) if max_area >= min_area else 0.0

    # find first index where the normalized value is maximal (tolerance for float)
    def _argmax_with_tol(seq):
        if not seq:
            return None
        m = max(seq)
        for idx, v in enumerate(seq):
            if abs(v - m) <= 1e-9:
                return idx
        return seq.index(m)

    # ---- trim per_frame to "last rest frame" + DURATION_S seconds
    baseline_idx = 0
    for i, rec in enumerate(per_frame):
        if abs(rec.get("lin_px", 0.0)) <= TRIM_THRESHOLD_PX:
            baseline_idx = i

    first_frame = per_frame[baseline_idx]["frame"]
    frames_window = int(round(fps * DURATION_S))
    per_frame_trimmed = per_frame[baseline_idx : baseline_idx + frames_window]

    # build normalized values and write the trimmed CSV
    csv_frames = os.path.join(out_dir, f"{name}__frames.csv")
    fieldnames = [
        "frame","time_s","x","y","w","h",
        "area_px","lin_px","rad_px",
        "lin_mm","rad_mm","area_mm",
        "lin_vel_mm","rad_vel_mm","area_vel_mm",
        "lin_norm","rad_norm","area_norm",
        "bend_deg","bend_deg_smooth","tip_x","tip_y","base_cx","base_cy"
    ]

    max_lin_frame = None
    max_rad_frame = None
    max_area_frame = None

    with open(csv_frames, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        lin_norms, rad_norms, area_norms = [], [], []
        for rec in per_frame_trimmed:
            new_frame = rec["frame"] - first_frame
            rec_out = dict(rec)
            rec_out["frame"] = new_frame
            rec_out["time_s"] = new_frame * dt
            rec_out["lin_norm"]  = (rec["lin_mm"]  / max_lin_mm)  if max_lin_mm  > 0 else 0.0
            rec_out["rad_norm"]  = (rec["rad_mm"]  / max_rad_mm)  if max_rad_mm  > 0 else 0.0
            rec_out["area_norm"] = (rec["area_mm"] / max_area_mm) if max_area_mm > 0 else 0.0

            lin_norms.append(rec_out["lin_norm"])
            rad_norms.append(rec_out["rad_norm"])
            area_norms.append(rec_out["area_norm"])

            wr.writerow(rec_out)

        i_lin  = _argmax_with_tol(lin_norms)
        i_rad  = _argmax_with_tol(rad_norms)
        i_area = _argmax_with_tol(area_norms)

        max_lin_frame  = per_frame_trimmed[i_lin]["frame"]  - first_frame if i_lin  is not None else 0
        max_rad_frame  = per_frame_trimmed[i_rad]["frame"]  - first_frame if i_rad  is not None else 0
        max_area_frame = per_frame_trimmed[i_area]["frame"] - first_frame if i_area is not None else 0

    # ---- summary CSV (append/create)
    csv_summary = os.path.join(out_dir, "summary_results.csv")
    header = [
        "video","inflation_period_s",
        "max_lin_mm","max_lin_time",
        "max_rad_mm","max_rad_time",
        "max_area_mm","max_area_time",
        "mm_per_px"
    ]
    need_hdr = not os.path.exists(csv_summary)

    with open(csv_summary, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if need_hdr:
            wr.writeheader()
        wr.writerow(dict(
            video=name,
            inflation_period_s=DURATION_S,
            max_lin_mm=max_lin_mm, max_lin_time=max_lin_frame * dt,
            max_rad_mm=max_rad_mm, max_rad_time=max_rad_frame * dt,
            max_area_mm=max_area_mm, max_area_time=max_area_frame * dt,
            mm_per_px=scale
        ))

    print("✓ Finished", video_path, "\n  → annotated:", out_path, "\n  → frames CSV:", csv_frames, "\n  → summary:", csv_summary)

# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to .mp4/.avi")
    ap.add_argument("--out", default="./processed_video_data/", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    process(args.video, args.out)
