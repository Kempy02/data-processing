# process_video_data.py
import cv2, json, csv, os, argparse, numpy as np


# ---------------------------------------------------------------------- #
# terminal code:
#   python process_video_data.py <video.mp4> --out <output_dir>
#   e.g. :
#   python process_video_data.py video_raw_data_mp4/AMP1_test1.mp4 --out ./processed_video_data/
# ----------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# CONFIG
# ---------------------------------------------------------------------- #
BOARD_SIZE          = (6, 9)          # inner corners (rows, cols)
SQUARE_MM           = 10.0
DETECTIONS_REQ      = 3              # chessboard finds before calib
COLOUR_RANGE_LO     = np.array([150, 70, 70],  dtype=np.uint8)   # magenta
COLOUR_RANGE_HI     = np.array([180, 255, 255],  dtype=np.uint8)
KERNEL              = np.ones((5,5), np.uint8)

SCALE_DEF           = 0.43348014           # fallback scale if calibration fails (mm per pixel)

# video codecs
CODECS = [("avc1",".mp4"), ("mp4v",".mp4"), ("MJPG",".avi")]

# ---------------------------------------------------------------------- #
# camera calibration function to find the chessboard corners and calculate the camera matrix
def calibrate(cap):
    rows, cols = BOARD_SIZE
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2) * SQUARE_MM

    objpoints, imgpoints = [], []
    origin = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    frames = DETECTIONS_REQ
    
    global calibration_frame
    calibration_frame = None

    global scale

    while len(objpoints) < DETECTIONS_REQ:
        ok, frame = cap.read()
        if not ok: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
            objpoints.append(objp)
            imgpoints.append(corners)

            calibration_frame = frames
            print(f"Calibration frame: {calibration_frame}")

        frames = frames + 1
        scale = mm_per_px(frame)

    # # Display the calibration process
    # while len(objpoints) < DETECTIONS_REQ:
    #     ok, frame = cap.read()
    #     if not ok: break
    #     display = frame.copy()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     found, corners = cv2.findChessboardCorners(gray, (cols, rows))
    #     if found:
    #         cv2.drawChessboardCorners(display, (cols, rows), corners, found)
    #         corners = cv2.cornerSubPix(
    #             gray, corners, (11,11), (-1,-1),
    #             (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
    #         objpoints.append(objp)
    #         imgpoints.append(corners)
    #     cv2.imshow("calib", display)         
    #     if cv2.waitKey(1) & 0xFF == 27: break
    # cv2.destroyWindow("calib")

    cap.set(cv2.CAP_PROP_POS_FRAMES, origin)
    if not objpoints:
        return False, None, None
    ok, mtx, dist, *_ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return ok, mtx, dist

# ---------------------------------------------------------------------- #
# Undistort the image using the camera matrix and distortion coefficients
def undistort(img, mtx, dist):
    h,w = img.shape[:2]
    new_mtx,_ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return cv2.undistort(img, mtx, dist, None, new_mtx)

# ---------------------------------------------------------------------- #
# Calculate the scale in mm per pixel based on the chessboard corners
def mm_per_px(frame):
    rows, cols = BOARD_SIZE
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows))
    if not ok: return None
    p0, p1 = corners[0][0], corners[1][0]
    dist = np.hypot(*(p1-p0))
    return SQUARE_MM / dist if dist>1e-3 else None

# ---------------------------------------------------------------------- #
# Open a VideoWriter with the first working codec
def open_writer(sample_frame, base_out):
    h,w = sample_frame.shape[:2]
    for fourcc,ext in CODECS:
        out_path = base_out + ext
        four = cv2.VideoWriter_fourcc(*fourcc)
        vw = cv2.VideoWriter(out_path, four, 30, (w,h))
        if vw.isOpened():
            print(f"[INFO] Using codec {fourcc} → {out_path}")
            return vw, out_path
        print(f"[WARN] Codec {fourcc} failed, trying next...")
    raise RuntimeError("Could not open any VideoWriter")

# ---------------------------------------------------------------------- #
# Main processing function - processes a single video file
# This function calibrates the camera, measures the deformation metrics of the actuator in the video,
# and writes the results to a CSV file.
def process(video_path, out_dir):

    # define globals
    global area_px
    global scale

    # --- 0) setup
    name = os.path.splitext(os.path.basename(video_path))[0]
    cap  = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open", video_path); return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30   # fallback if metadata missing

    # --- 1) calibrate
    ok, mtx, dist = calibrate(cap)
    # ok = None
    if not ok:
        scale_def = SCALE_DEF
        print("Calibration failed, scale =", scale_def)
        return
    else:
        scale_def = 0.35#scale
        print("Calibration successful, scale =", scale_def)
    
    scale = scale_def

    # --- 2) measurement loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vw = None
    min_w = min_h = 1e9
    max_w = max_h = 0
    min_area = max_area = 0
    per_frame = []

    dt = 1.0 / fps

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, COLOUR_RANGE_LO, COLOUR_RANGE_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)

        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            min_w, max_w = min(min_w,w), max(max_w,w)
            min_h, max_h = min(min_h,h), max(max_h,h)

            if frame_idx == 1:
                max_area = min_area = cv2.contourArea(c)
            elif frame_idx > 1:
                min_area, max_area = min(min_area,area), max(max_area,area)

            # --- store per-frame record -------------------------
            lin_px = h - min_h
            rad_px = w - min_w
            area_px = area - min_area

            max_lin_px = max_h - min_h
            max_rad_px = max_w - min_w
            # area_px = cv2.contourArea(c)
            max_area_px = max_area - min_area

            lin_mm = lin_px * scale if scale else ""
            rad_mm = rad_px * scale if scale else ""
            area_mm = area_px * (scale**2) if scale else ""

            max_lin_mm = max_lin_px * scale if scale else ""
            max_rad_mm = max_rad_px * scale if scale else ""
            max_area_mm = max_area_px * (scale**2) if scale else ""

            # ---- additional dependant data
            # dt = 1.0 / fps

            # lin_mm_final = lin_mm
            # rad_mm_final = rad_mm
            # area_mm_final = area_mm

            lin_vel_mm = (lin_mm - prev_lin_mm) / dt if frame_idx else 0.0
            rad_vel_mm = (rad_mm - prev_rad_mm) / dt if frame_idx else 0.0
            area_vel_mm = (area_mm - prev_area_mm) / dt if frame_idx else 0.0

            per_frame.append({
                "frame"   : frame_idx,
                # "time_s"  : dt*frame_idx,
                "x"       : x,  "y": y,  "w": w,  "h": h,
                # "area_px" : cv2.contourArea(c),
                "area_px" : area_px,
                "lin_px"  : lin_px,
                "rad_px"  : rad_px,
                "lin_mm"  : lin_mm,
                "rad_mm"  : rad_mm,
                "area_mm" : area_mm,
                "lin_vel_mm" : lin_vel_mm,
                "rad_vel_mm" : rad_vel_mm,
                "area_vel_mm" : area_vel_mm,
            })

            # overlay
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            overlay = frame.copy()
            cv2.drawContours(overlay,[c],-1,(0,0,255),cv2.FILLED)
            cv2.addWeighted(overlay,0.5,frame,0.5,0,frame)

            # NEW ➜ thin black contour line
            # cv2.drawContours(frame, [c], -1, (0,0,0), 2)

        # lazy-open the VideoWriter once we know frame size
        if vw is None:
            vw, out_path = open_writer(frame,
                         os.path.join(out_dir, f"{name}__annot"))

        vw.write(frame)
        
        # counters for the next frame
        prev_lin_mm = lin_mm
        prev_rad_mm = rad_mm
        prev_area_mm = area_mm
        frame_idx += 1 
        

    cap.release()
    if vw: vw.release()

    # a-priori/dependent values
    lin_mm_final_max = max_lin_mm
    rad_mm_final_max = max_rad_mm
    area_mm_final_max = max_area_mm

    max_lin_frame = None
    max_rad_frame = None
    max_area_frame = None

    # ---- trim per_frame to "last rest frame" + defined time (s) ------------------------
    TRIM_THRESHOLD = 5 
    frames_4s      = int(fps * 3.5)   # how many frames = 3.5 seconds

    # find the last index where lin_px is ~0
    baseline_idx = None
    for i, rec in enumerate(per_frame):
        if abs(rec["lin_px"]) <= TRIM_THRESHOLD:
            baseline_idx = i
    first_frame = baseline_idx - 1
    print("First frame =", first_frame)

    if baseline_idx is not None:
        per_frame_trimmed = per_frame[baseline_idx : baseline_idx + frames_4s]
    else:
        # fallback: keep the full list (shouldn’t normally happen)
        per_frame_trimmed = per_frame

    # ---- write the trimmed CSV -------------------------------------------
    if per_frame_trimmed:
        csv_frames = os.path.join(out_dir, f"{name}__frames.csv")
        with open(csv_frames, "w", newline="") as f:
            fld = ["frame","time_s","x","y","w","h",
                "area_px","lin_px","rad_px",
                "lin_mm","rad_mm","area_mm",
                "lin_vel_mm","rad_vel_mm","area_vel_mm",
                "lin_norm","rad_norm","area_norm"]
            wr = csv.DictWriter(f, fieldnames=fld)
            wr.writeheader()
            for rec in per_frame_trimmed:
                new_frame = rec["frame"] - first_frame

                rec["frame"] = new_frame
                rec["time_s"] = new_frame * dt
                rec["lin_norm"] = rec["lin_mm"] / lin_mm_final_max if lin_mm_final_max else ""
                rec["rad_norm"] = rec["rad_mm"] / rad_mm_final_max if rad_mm_final_max else ""
                rec["area_norm"] = rec["area_mm"] / area_mm_final_max if area_mm_final_max else ""
                wr.writerow(rec)

                # collect max_value frame data
                if rec["lin_norm"] == 1.0 and max_lin_frame is None:
                    max_lin_frame = new_frame
                if rec["rad_norm"] == 1.0 and max_rad_frame is None:
                    max_rad_frame = new_frame
                if rec["area_norm"] == 1.0 and max_area_frame is None:
                    max_area_frame = new_frame
            wr.writerows(per_frame_trimmed)

    # ---- summary CSV (append / create)
    csv_path = os.path.join(out_dir,"summary_results.csv")
    header   = ["video","inflation_period_s","max_lin_px","max_lin_mm","max_lin_time","max_rad_px","max_rad_mm","max_rad_time","max_area_px","max_area_mm","max_area_time","mm_per_px"]
    need_hdr = not os.path.exists(csv_path)

    with open(csv_path,"a",newline="") as f:
        wr = csv.DictWriter(f, fieldnames=header)
        if need_hdr: wr.writeheader()

        # area_mm = lin_mm * rad_mm
        wr.writerow(dict(video=name, inflation_period_s=3.0,
                         max_lin_px=max_lin_px, max_lin_mm=max_lin_mm, max_lin_time=(max_lin_frame*dt), 
                         max_rad_px=max_rad_px, max_rad_mm=max_rad_mm, max_rad_time=(max_rad_frame*dt),
                         max_area_px=max_area_px, max_area_mm=max_area_mm, max_area_time=(max_area_frame*dt),
                         mm_per_px=scale or ""))

    print("✓ Finished", video_path,"\n  →", out_path)

# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="single .mp4 to process")
    ap.add_argument("--out", default="./processed_video_data/", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    process(args.video, args.out)
