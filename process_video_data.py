# process_video_data.py
# -----------------------------------
# Per-video calibration + measurement pipeline with linear-extension and bending-angle tracking.
#
# Bending tip selection rule:
#   * Base point (locked once): bbox bottom mid-point (x + w/2, y + h)
#   * Tip point:
#       - Use HIGHEST contour point by default
#       - If the highest point's x is OUTSIDE the original bbox width range
#         (captured on the first valid frame), use the bbox top corner (left or right)
#         with the larger |x - base_x|.
#
# Summary CSV tweak:
#   * "max_bend_deg", "max_bend_mm", "max_bend_time" now correspond to the frame where
#     the BOUNDING-BOX AREA (w*h) is maximal (not the absolute max bend across the run).
#
# Usage:
#   Set INPUT_DIR and OUTPUT_DIR below, then run:
#       python process_video_data.py
# -----------------------------------

import os
import glob
import cv2
import csv
import math
import numpy as np

# ---------------- I/O ----------------
INPUT_DIR  = "raw_video_data/ExRunBending1 - MP4"   # <- SET
OUTPUT_DIR = "./processed_video_data/ExRunBending"  # <- SET

SPEC_VIDEO = None  # for single-file debug runs; set to None to batch process all videos in INPUT_DIR

# ---- Toggle bending calculations/outputs here ----
BENDING_ENABLED = True   # set False to skip bending angle & horizontal displacement entirely

# ---------------- CONFIG ----------------
# Chessboard for mm/px scale
BOARD_SQUARES       = (5, 5)                   # physical squares on printed board
BOARD_INNER         = (BOARD_SQUARES[0]-1, BOARD_SQUARES[1]-1)
SQUARE_MM           = 10.0                     # edge length of one square, in mm

DETECTIONS_REQ      = 3
FRAME_STRIDE        = 2
MAX_CALIB_FRAMES    = 20
SCALE_DEF           = 0.38  # fallback mm/px if calibration fails
UNDISTORT           = False
DURATION_S          = 3.0
TRIM_THRESHOLD_PX   = 5

# Body color (magenta) mask in HSV (tune as needed)
COLOUR_RANGE_LO     = np.array([140,  50, 175], dtype=np.uint8)
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
    if corners is None or len(corners) == 0:
        return None
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])
    pts = corners.reshape(-1, 2)
    dists = []
    for r in range(rows_inner):
        base = r * cols_inner
        for c in range(cols_inner - 1):
            p0 = pts[base + c]; p1 = pts[base + c + 1]
            dists.append(np.hypot(*(p1 - p0)))
    for c in range(cols_inner):
        for r in range(rows_inner - 1):
            p0 = pts[r * cols_inner + c]; p1 = pts[(r + 1) * cols_inner + c]
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
        if not ok: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols_inner, rows_inner))
        if not found: continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp); imgpoints.append(corners); detections += 1

        scale_i = _robust_mm_per_px_from_corners(corners, SQUARE_MM)
        if scale_i is not None: scale_estimates.append(scale_i)
        if detections >= DETECTIONS_REQ: break

    cap.set(cv2.CAP_PROP_POS_FRAMES, origin)
    if not objpoints or gray is None: return False, None, None, None

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

# ---------------- Per-video processing ----------------
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
        print(f"[INFO] Calibration COMPLETE. mm/px = {scale:.6f}")

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

    # base & original bbox width range (from first valid detection)
    base_cx_locked = None
    base_cy_locked = None
    baseline_bbox_left = None
    baseline_bbox_right = None

    per_frame = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if UNDISTORT and (mtx is not None) and (dist is not None):
            frame = undistort(frame, mtx, dist)

        hsv_body = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_body, COLOUR_RANGE_LO, COLOUR_RANGE_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lin_mm = prev_lin_mm
        rad_mm = prev_rad_mm
        area_mm = prev_area_mm

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)  # still used for legacy width/height metrics
            area = float(cv2.contourArea(c))
            pts = c.reshape(-1, 2)

            # ---- initialize extrema & baseline bbox range on first detection
            if not found_any:
                min_w = max_w = float(w)
                min_h = max_h = float(h)
                min_area = max_area = float(area)
                baseline_bbox_left  = float(x)
                baseline_bbox_right = float(x + w)
                found_any = True

            # ---- bending path (optional)
            bend_fields = {}
            if BENDING_ENABLED:
                # Lock base center ONCE: bbox bottom mid-point
                if base_cx_locked is None:
                    base_cx_locked = float(x + w/2.0)
                    base_cy_locked = float(y + h)   # bottom edge
                base_cx = base_cx_locked
                base_cy = base_cy_locked

                # Tip DEFAULT: highest contour point
                tip_highest = pts[np.argmin(pts[:, 1])].astype(float)
                tip_x = float(tip_highest[0])

                # Condition: if highest tip x is OUTSIDE initial bbox width range → use corner
                use_corner = (
                    baseline_bbox_left is not None and baseline_bbox_right is not None and
                    (tip_x < baseline_bbox_left or tip_x > baseline_bbox_right)
                )

                if use_corner:
                    left_top  = np.array([float(x),       float(y)], dtype=float)
                    right_top = np.array([float(x + w),   float(y)], dtype=float)
                    dx_left   = left_top[0]  - base_cx
                    dx_right  = right_top[0] - base_cx
                    tip = right_top if abs(dx_right) >= abs(dx_left) else left_top
                else:
                    tip = tip_highest

                # Bending angle (base -> tip) + horizontal displacement
                dx_px = tip[0] - base_cx
                dy    = base_cy - tip[1]            # screen y downward → invert
                bend_rad = math.atan2(dx_px, dy)    # 0 = vertical; sign: + right, - left
                bend_deg = abs(math.degrees(bend_rad))
                horiz_disp_px = dx_px
                horiz_disp_mm = horiz_disp_px * scale

                bend_fields.update({
                    "bend_deg": bend_deg,
                    "tip_x": float(tip[0]),
                    "tip_y": float(tip[1]),
                    "base_cx": base_cx,
                    "base_cy": base_cy,
                    "horiz_disp_px": float(horiz_disp_px),
                    "horiz_disp_mm": float(horiz_disp_mm)
                })

            # ---- update extrema
            min_w, max_w = min(min_w, w), max(max_w, w)
            min_h, max_h = min(min_h, h), max(max_h, h)
            min_area, max_area = min(min_area, area), max(max_area, area)

            # ---- deltas relative to current minima
            lin_px  = float(h - min_h)
            rad_px  = float(w - min_w)
            area_px = float(area - min_area)

            # ---- convert to mm / mm^2
            lin_mm  = lin_px  * scale
            rad_mm  = rad_px  * scale
            area_mm = area_px * (scale ** 2)

            # ---- velocities
            lin_vel_mm  = (lin_mm  - prev_lin_mm)  / dt if frame_idx else 0.0
            rad_vel_mm  = (rad_mm  - prev_rad_mm)  / dt if frame_idx else 0.0
            area_vel_mm = (area_mm - prev_area_mm) / dt if frame_idx else 0.0

            # ---- store record
            rec = {
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
            }
            if BENDING_ENABLED:
                rec.update(bend_fields)
            per_frame.append(rec)

            # ---- overlay viz ----
            overlay = frame.copy()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.drawContours(overlay, [c], -1, (0,0,255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            label_y = max(0, int(min(pts[:,1])) - 60)
            cv2.putText(frame, f"Lin ext: {lin_mm:+.1f} mm",
                        (int(min(pts[:,0])), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            if BENDING_ENABLED and bend_fields:
                cv2.circle(frame, (int(bend_fields["base_cx"]), int(bend_fields["base_cy"])), 5, (0, 255, 255), -1)
                cv2.circle(frame, (int(bend_fields["tip_x"]), int(bend_fields["tip_y"])), 5, (255, 255, 0), -1)
                cv2.line(frame,
                         (int(bend_fields["base_cx"]), int(bend_fields["base_cy"])),
                         (int(bend_fields["tip_x"]), int(bend_fields["tip_y"])),
                         (255, 255, 255), 2)
                cv2.putText(frame, f"Bend ang: {bend_fields['bend_deg']:+.1f} deg",
                            (int(min(pts[:,0])), label_y + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Bend dx: {bend_fields['horiz_disp_mm']:+.1f} mm",
                            (int(min(pts[:,0])), label_y + 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        # lazy-open writer once we know frame dims
        if vw is None:
            os.makedirs(out_dir, exist_ok=True)
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
        print("[WARN] No valid contours found; nothing to write for", name)
        return

    # maxima (based on extrema we tracked)
    max_lin_mm  = float((max_h - min_h) * scale) if max_h >= min_h else 0.0
    max_rad_mm  = float((max_w - min_w) * scale) if max_w >= min_w else 0.0
    max_area_mm = float((max_area - min_area) * (scale ** 2)) if max_area >= min_area else 0.0

    # ---- trim per_frame to "last rest frame" + DURATION_S seconds
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    baseline_idx = 0
    for i, rec in enumerate(per_frame):
        if abs(rec.get("lin_px", 0.0)) <= TRIM_THRESHOLD_PX:
            baseline_idx = i

    first_frame = per_frame[baseline_idx]["frame"]
    frames_window = int(round(fps * DURATION_S))
    per_frame_trimmed = per_frame[baseline_idx : baseline_idx + frames_window]

    # build normalized values and write the trimmed CSV
    csv_frames = os.path.join(out_dir, f"{name}__frames.csv")
    base_fields = [
        "frame","time_s","x","y","w","h",
        "area_px","lin_px","rad_px",
        "lin_mm","rad_mm","area_mm",
        "lin_vel_mm","rad_vel_mm","area_vel_mm",
        "lin_norm","rad_norm","area_norm",
    ]
    bend_fields = [
        "bend_deg","tip_x","tip_y","base_cx","base_cy",
        "horiz_disp_px","horiz_disp_mm"
    ]
    fieldnames = base_fields + (bend_fields if BENDING_ENABLED else [])

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

            if rec_out["lin_norm"] == 1.0 and max_lin_frame is None:
                max_lin_frame = new_frame
            if rec_out["rad_norm"] == 1.0 and max_rad_frame is None:
                max_rad_frame = new_frame
            if rec_out["area_norm"] == 1.0 and max_area_frame is None:
                max_area_frame = new_frame

            if not BENDING_ENABLED:
                for k in bend_fields:
                    rec_out.pop(k, None)

            wr.writerow(rec_out)

    # ---- NEW: bending-at-max-bbox-area ----
    if BENDING_ENABLED:
        if per_frame:
            # Find the frame with maximum bounding-box area (w*h) over the FULL run
            bbox_areas = [float(r["w"]) * float(r["h"]) for r in per_frame]
            idx_bbox_max = int(np.argmax(bbox_areas))
            rec_bbox_max = per_frame[idx_bbox_max]
            bend_at_bbox_max_deg = float(rec_bbox_max.get("bend_deg", 0.0))
            disp_at_bbox_max_mm  = float(rec_bbox_max.get("horiz_disp_mm", 0.0))
            time_at_bbox_max     = float(rec_bbox_max["frame"]) * dt
        else:
            bend_at_bbox_max_deg = 0.0
            disp_at_bbox_max_mm  = 0.0
            time_at_bbox_max     = 0.0

    # ---- summary CSV (append/create) ----
    csv_summary = os.path.join(out_dir, "summary_results.csv")
    if BENDING_ENABLED:
        header = [
            "video","inflation_period_s",
            "max_lin_mm","max_lin_time",
            "max_rad_mm","max_rad_time",
            "max_area_mm","max_area_time",
            # These are now the bend metrics AT max bbox area:
            "max_bend_deg","max_bend_mm","max_bend_time",
            "mm_per_px",
        ]
    else:
        header = [
            "video","inflation_period_s",
            "max_lin_mm","max_lin_time",
            "max_rad_mm","max_rad_time",
            "max_area_mm","max_area_time",
            "mm_per_px",
        ]
    need_hdr = not os.path.exists(csv_summary)

    with open(csv_summary, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if need_hdr:
            wr.writeheader()
        base_row = dict(
            video=name,
            inflation_period_s=DURATION_S,
            max_lin_mm=max_lin_mm,
            max_lin_time=(0.0 if max_lin_frame is None else max_lin_frame * dt),
            max_rad_mm=max_rad_mm,
            max_rad_time=(0.0 if max_rad_frame is None else max_rad_frame * dt),
            max_area_mm=max_area_mm,
            max_area_time=(0.0 if max_area_frame is None else max_area_frame * dt),
            mm_per_px=scale
        )
        if BENDING_ENABLED:
            base_row.update(dict(
                max_bend_deg=bend_at_bbox_max_deg,
                max_bend_mm=disp_at_bbox_max_mm,
                max_bend_time=time_at_bbox_max,
            ))
        wr.writerow(base_row)

    print("✓ Finished", video_path,
          "\n  → annotated:", os.path.join(out_dir, f"{name}__annot.*"),
          "\n  → frames CSV:", csv_frames,
          "\n  → summary:", csv_summary)

# ---------------- Batch runner ----------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv")
    videos = []
    if SPEC_VIDEO:
        sp = os.path.join(INPUT_DIR, SPEC_VIDEO)
        if os.path.exists(sp):
            videos = [sp]
        else:
            print(f"[ERROR] SPEC_VIDEO set to {SPEC_VIDEO} but that file does not exist in {INPUT_DIR}")
            exit(1)
    else:
        for ext in exts:
            videos.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    if not videos:
        print(f"[WARN] No videos found in {INPUT_DIR} with extensions {exts}")
    for vp in sorted(videos):
        print(f"\n[RUN] Processing: {vp}")
        process(vp, OUTPUT_DIR)
