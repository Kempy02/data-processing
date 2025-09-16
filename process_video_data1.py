# process_video_data1.py
# -----------------------------------
# Per-video calibration + measurement pipeline.
# Usage:
#   python process_video_data1.py <video.mp4> --out <output_dir>
# Example:
#   python process_video_data1.py raw_video_data/test1.mp4 --out ./processed_video_data/
# -----------------------------------

import os
import cv2
import csv
import math
import argparse
import numpy as np

# ---------------- CONFIG ----------------
# If the PHYSICAL board has Sx x Sy squares, OpenCV wants *inner* corners:
# inner = (Sx-1, Sy-1)
# For a 5x5 squares board → inner corners = (4,4).
BOARD_SQUARES       = (5, 5)                   # physical squares on printed board
BOARD_INNER         = (BOARD_SQUARES[0]-1, BOARD_SQUARES[1]-1)
SQUARE_MM           = 10.0                     # edge length of one square, in mm

DETECTIONS_REQ      = 10                       # frames needed for calibration
FRAME_STRIDE        = 5                        # sample every Nth frame for calibration
MAX_CALIB_FRAMES    = 2000                     # upper cap for scanning
SCALE_DEF           = 0.43348014               # fallback mm/px if calibration fails
UNDISTORT           = True                     # undistort frames during measurement
DURATION_S          = 3.5                      # trimmed window after baseline (~rest) frame
TRIM_THRESHOLD_PX   = 5                        # |lin_px| <= threshold → baseline frame

# Target colour (mask) — magenta example (HSV)
COLOUR_RANGE_LO     = np.array([150, 70, 70],  dtype=np.uint8)
COLOUR_RANGE_HI     = np.array([180, 255, 255], dtype=np.uint8)
KERNEL              = np.ones((5,5), np.uint8)

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
    """
    Given subpixel chessboard corners (N x 1 x 2), compute mm/px for that frame by
    averaging adjacent-corner pixel spacings in both directions.
    """
    if corners is None or len(corners) == 0:
        return None

    cols_inner, rows_inner = BOARD_INNER  # inner corners along X (cols) and Y (rows)
    cols_inner, rows_inner = int(cols_inner), int(rows_inner)
    pts = corners.reshape(-1, 2)  # (rows*cols, 2)

    dists = []
    # horizontal neighbors
    for r in range(rows_inner):
        base = r * cols_inner
        for c in range(cols_inner - 1):
            p0 = pts[base + c]
            p1 = pts[base + c + 1]
            dists.append(np.hypot(*(p1 - p0)))
    # vertical neighbors
    for c in range(cols_inner):
        for r in range(rows_inner - 1):
            p0 = pts[r * cols_inner + c]
            p1 = pts[(r + 1) * cols_inner + c]
            dists.append(np.hypot(*(p1 - p0)))

    if not dists:
        return None
    px_mean = float(np.median(dists))  # robust to outliers
    return square_mm / px_mean if px_mean > 1e-6 else None


def calibrate_from_video(cap) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None]:
    """
    Collect chessboard detections from this video, calibrate intrinsics, and return
    a robust mm/px for THIS video by averaging adjacent-corner spacings.
    Returns: (ok, camera_matrix, dist_coeffs, mm_per_px)
    """
    # OpenCV expects pattern as (cols, rows) for findChessboardCorners
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])

    # world coordinates (z=0 plane), in millimetres
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

        # refine to subpixel
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        detections += 1

        # per-frame mm/px
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
    """Open a VideoWriter with the first working codec."""
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

    # track extrema & history
    found_any = False
    min_w = min_h = math.inf
    max_w = max_h = 0.0
    min_area = math.inf
    max_area = 0.0

    prev_lin_mm = 0.0
    prev_rad_mm = 0.0
    prev_area_mm = 0.0

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
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            area = float(cv2.contourArea(c))

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
                "x": x, "y": y, "w": w, "h": h,
                "area_px": area_px,
                "lin_px": lin_px,
                "rad_px": rad_px,
                "lin_mm": lin_mm,
                "rad_mm": rad_mm,
                "area_mm": area_mm,
                "lin_vel_mm": lin_vel_mm,
                "rad_vel_mm": rad_vel_mm,
                "area_vel_mm": area_vel_mm,
            })

            # overlay viz
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            overlay = frame.copy()
            cv2.drawContours(overlay, [c], -1, (0,0,255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

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
    # baseline: last idx where |lin_px| <= threshold
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
            rec_out = dict(rec)  # copy
            rec_out["frame"] = new_frame
            rec_out["time_s"] = new_frame * dt
            rec_out["lin_norm"]  = (rec["lin_mm"]  / max_lin_mm)  if max_lin_mm  > 0 else 0.0
            rec_out["rad_norm"]  = (rec["rad_mm"]  / max_rad_mm)  if max_rad_mm  > 0 else 0.0
            rec_out["area_norm"] = (rec["area_mm"] / max_area_mm) if max_area_mm > 0 else 0.0

            lin_norms.append(rec_out["lin_norm"])
            rad_norms.append(rec_out["rad_norm"])
            area_norms.append(rec_out["area_norm"])

            wr.writerow(rec_out)

        # find first max frames within the trimmed window
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
