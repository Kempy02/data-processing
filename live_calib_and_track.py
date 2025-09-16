# live_calib_and_track.py
# ---------------------------------------------------------
# Live camera calibration + actuator tracking.
#
# Keys:
#   c  : start/restart calibration
#   t  : switch to tracking (after calibration)
#   u  : toggle undistortion during tracking
#   r  : reset tracking baseline (min width/height/area)
#   s  : save current annotated frame (PNG)
#   q/ESC : quit
#
# Run:
#   python live_calib_and_track.py --cam 0
# ---------------------------------------------------------

import os
import cv2
import time
import math
import argparse
import numpy as np
from datetime import datetime

# ---------- Config ----------
# Physical board: 5 x 5 squares => inner corners = 4 x 4 for OpenCV
BOARD_SQUARES = (5, 5)                        # physical squares on the printed board
BOARD_INNER   = (BOARD_SQUARES[0]-1, BOARD_SQUARES[1]-1)  # inner corners
SQUARE_MM     = 10.0                          # square size (mm)
DETECTIONS_REQ = 10                           # frames to collect for calibration
SCALE_DEF      = 0.43348014                   # fallback mm/px if calibration fails
UNDISTORT_DEFAULT = True

# sample chessboard every N frames while calibrating
CALIB_STRIDE_FRAMES = 2

# HSV range for magenta mask (same as your batch script)
COLOUR_RANGE_LO = np.array([150, 70, 70], dtype=np.uint8)
COLOUR_RANGE_HI = np.array([180, 255, 255], dtype=np.uint8)
KERNEL = np.ones((5,5), np.uint8)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------- Helpers ----------
def draw_text(img, text, org, scale=0.5, color=(255,255,255), thickness=1, bg=True):
    if bg:
        (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x-2, y-h-2), (x+w+2, y+baseline+2), (0,0,0), -1)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

def undistort(img, mtx, dist):
    if mtx is None or dist is None:
        return img
    h, w = img.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, mtx, dist, None, new_mtx)

def robust_mm_per_px_from_corners(corners, square_mm: float) -> float | None:
    """
    Per-frame mm/px estimate from subpixel chessboard corners by averaging
    adjacent-corner distances (horizontal + vertical).
    """
    if corners is None or len(corners) == 0:
        return None

    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])
    pts = corners.reshape(-1, 2)

    dists = []
    # horizontal neighbors
    for r in range(rows_inner):
        base = r * cols_inner
        for c in range(cols_inner - 1):
            p0, p1 = pts[base + c], pts[base + c + 1]
            dists.append(float(np.hypot(*(p1 - p0))))
    # vertical neighbors
    for c in range(cols_inner):
        for r in range(rows_inner - 1):
            p0, p1 = pts[r * cols_inner + c], pts[(r + 1) * cols_inner + c]
            dists.append(float(np.hypot(*(p1 - p0))))

    if not dists:
        return None
    px = float(np.median(dists))
    return square_mm / px if px > 1e-6 else None

def calibrate_incremental(frame, objpoints, imgpoints, scales) -> tuple[bool, np.ndarray | None, np.ndarray | None, float | None, np.ndarray | None]:
    """
    Try to detect chessboard on given frame and append a detection if found.
    If we have enough detections, run calibrateCamera and return intrinsics + mm/px.
    Returns: (done, mtx, dist, mm_per_px, corners_drawn)
    """
    cols_inner, rows_inner = int(BOARD_INNER[0]), int(BOARD_INNER[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, (cols_inner, rows_inner))
    corners_drawn = None

    if found:
        # refine to subpixel
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        # append
        objp = np.zeros((rows_inner * cols_inner, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols_inner, 0:rows_inner].T.reshape(-1, 2) * SQUARE_MM
        objpoints.append(objp)
        imgpoints.append(corners)

        # per-frame mm/px
        s = robust_mm_per_px_from_corners(corners, SQUARE_MM)
        if s is not None:
            scales.append(float(s))

        corners_drawn = frame.copy()
        cv2.drawChessboardCorners(corners_drawn, (cols_inner, rows_inner), corners, found)

    if len(objpoints) >= DETECTIONS_REQ:
        h, w = gray.shape[:2]
        ok, mtx, dist, *_ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
        scale = float(np.median(scales)) if scales else None
        return True, (mtx if ok else None), (dist if ok else None), scale, corners_drawn

    return False, None, None, None, corners_drawn

def save_frame(img, prefix="frame"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{prefix}_{ts}.png"
    cv2.imwrite(path, img)
    print(f"[INFO] saved {path}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="camera index (default 0)")
    ap.add_argument("--no-undistort", action="store_true", help="disable undistortion during tracking")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] Could not open camera", args.cam)
        return

    ok, sample = cap.read()
    if not ok:
        print("[ERROR] Could not read from camera")
        return

    # state
    state = "CALIB"        # CALIB or TRACK
    undistort_on = not args.no_undistort

    # calibration accumulators
    objpoints, imgpoints, scales = [], [], []
    last_calibration = dict(mtx=None, dist=None, mm_per_px=None)

    # tracking baseline
    min_w = min_h = math.inf
    min_area = math.inf
    max_w = max_h = 0.0
    max_area = 0.0
    prev_lin_mm = prev_rad_mm = prev_area_mm = 0.0

    # timing
    fps_tick = time.time()
    frames_count = 0
    fps = 0.0

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        display = frame.copy()
        h, w = display.shape[:2]

        # crude FPS
        frames_count += 1
        if time.time() - fps_tick >= 0.5:
            fps = frames_count / (time.time() - fps_tick)
            fps_tick = time.time()
            frames_count = 0

        # ---- CALIBRATION MODE ----
        if state == "CALIB":
            # sample detection every CALIB_STRIDE_FRAMES to avoid near-duplicates
            corners_overlay = None
            if (frame_id % CALIB_STRIDE_FRAMES) == 0:
                done, mtx, dist, scale, corners_overlay = calibrate_incremental(display, objpoints, imgpoints, scales)
                if corners_overlay is not None:
                    display = corners_overlay

                draw_text(display, f"CALIBRATION: {len(objpoints)}/{DETECTIONS_REQ} detections", (10, 20))
                if scales:
                    draw_text(display, f"mm/px (median of seen): {np.median(scales):.5f}", (10, 45))
                if done:
                    if (mtx is not None) and (dist is not None):
                        last_calibration["mtx"] = mtx
                        last_calibration["dist"] = dist
                        last_calibration["mm_per_px"] = float(scale) if scale else float(SCALE_DEF)
                        draw_text(display, "Calibration SUCCESS", (10, 70), color=(0,255,0))
                        draw_text(display, f"mm/px: {last_calibration['mm_per_px']:.5f}", (10, 95), color=(0,255,0))
                        # print intrinsics to console
                        print("[CALIB] mm/px =", last_calibration["mm_per_px"])
                        print("[CALIB] Camera matrix:\n", mtx)
                        print("[CALIB] Distortion:", dist.ravel())
                    else:
                        last_calibration["mtx"] = None
                        last_calibration["dist"] = None
                        last_calibration["mm_per_px"] = float(SCALE_DEF)
                        draw_text(display, "Calibration FAILED → using fallback scale", (10, 70), color=(0,0,255))
                    # switch to tracking automatically
                    state = "TRACK"

            draw_text(display, "Press 'c' to restart calibration, 't' to force TRACK", (10, h-30))

        # ---- TRACKING MODE ----
        else:
            mtx = last_calibration["mtx"]
            dist = last_calibration["dist"]
            mm_per_px = last_calibration["mm_per_px"] or float(SCALE_DEF)

            if undistort_on and (mtx is not None) and (dist is not None):
                display = undistort(display, mtx, dist)

            hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, COLOUR_RANGE_LO, COLOUR_RANGE_HI)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            lin_mm = prev_lin_mm
            rad_mm = prev_rad_mm
            area_mm = prev_area_mm

            if cnts:
                c = max(cnts, key=cv2.contourArea)
                x, y, ww, hh = cv2.boundingRect(c)
                area = float(cv2.contourArea(c))

                # init baselines on first detection
                if math.isinf(min_w):
                    min_w = max_w = float(ww)
                    min_h = max_h = float(hh)
                    min_area = max_area = float(area)

                # update extrema
                min_w, max_w = min(min_w, ww), max(max_w, ww)
                min_h, max_h = min(min_h, hh), max(max_h, hh)
                min_area, max_area = min(min_area, area), max(max_area, area)

                # deltas
                lin_px  = float(hh - min_h)
                rad_px  = float(ww - min_w)
                area_px = float(area - min_area)

                lin_mm  = lin_px  * mm_per_px
                rad_mm  = rad_px  * mm_per_px
                area_mm = area_px * (mm_per_px ** 2)

                # draw overlay
                cv2.rectangle(display, (x, y), (x+ww, y+hh), (0,255,0), 2)
                ov = display.copy()
                cv2.drawContours(ov, [c], -1, (0,0,255), cv2.FILLED)
                cv2.addWeighted(ov, 0.5, display, 0.5, 0, display)

                draw_text(display, f"lin: {lin_mm:.2f} mm", (10, 70))
                draw_text(display, f"rad: {rad_mm:.2f} mm", (10, 95))
                draw_text(display, f"area: {area_mm:.2f} mm^2", (10, 120))

            # maxima (based on seen extrema)
            max_lin_mm  = float((max_h - min_h) * mm_per_px) if (max_h >= min_h and not math.isinf(min_h)) else 0.0
            max_rad_mm  = float((max_w - min_w) * mm_per_px) if (max_w >= min_w and not math.isinf(min_w)) else 0.0
            max_area_mm = float((max_area - min_area) * (mm_per_px ** 2)) if (max_area >= min_area and not math.isinf(min_area)) else 0.0

            draw_text(display, f"mm/px: {mm_per_px:.5f}", (10, 20))
            draw_text(display, f"max lin: {max_lin_mm:.2f} | max rad: {max_rad_mm:.2f}", (10, 145))
            draw_text(display, f"undistort: {'ON' if undistort_on else 'OFF'}", (10, 170), color=(0,255,255))
            draw_text(display, "Keys: c=calib  t=track  u=undistort  r=reset  s=save  q=quit", (10, h-30))

            prev_lin_mm, prev_rad_mm, prev_area_mm = lin_mm, rad_mm, area_mm

        # common HUD
        draw_text(display, f"FPS: {fps:.1f}", (display.shape[1]-110, 20))

        cv2.imshow("Live Calib + Tracking", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('c'):
            # restart calibration
            state = "CALIB"
            objpoints.clear(); imgpoints.clear(); scales.clear()
            draw_text(display, "Restarting calibration...", (10, 200), color=(0,0,255))
        elif key == ord('t'):
            state = "TRACK"
        elif key == ord('u'):
            undistort_on = not undistort_on
        elif key == ord('r'):
            # reset baselines/extrema
            min_w = min_h = min_area = math.inf
            max_w = max_h = max_area = 0.0
            prev_lin_mm = prev_rad_mm = prev_area_mm = 0.0
            print("[INFO] tracking baseline reset")
        elif key == ord('s'):
            save_frame(display, prefix="live")

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
