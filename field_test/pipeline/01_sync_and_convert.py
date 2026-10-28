import cv2
import numpy as np
import pandas as pd
import os
import glob
import re
import sys
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import DATA  # noqa: E402

# --- CONFIGURATION ---
CALIB_FILE = "field_test/CALIBRATION.md"
BOOT_OFFSET = 1774262813.4894

def get_offset():
    if os.path.exists(CALIB_FILE):
        with open(CALIB_FILE, 'r') as f:
            content = f.read()
            match = re.search(r'Value: `([\d\.]+)`', content)
            if match:
                return float(match.group(1))
    return BOOT_OFFSET

def find_anchor(video_path, image_paths):
    print(f"  - Finding visual anchor using {len(image_paths)} captures...")
    cap = cv2.VideoCapture(video_path)
    ref_idx = len(image_paths) // 2
    ref_img_path = image_paths[ref_idx]
    ref_nano = int(re.search(r'(\d+)\.jpg', os.path.basename(ref_img_path)).group(1))
    
    target_small = cv2.resize(cv2.imread(ref_img_path), (160, 90))
    best_frame = 0
    min_mse = float('inf')
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % 2 == 0:
            frame_small = cv2.resize(frame, (160, 90))
            mse = np.mean((target_small.astype("float") - frame_small.astype("float")) ** 2)
            if mse < min_mse:
                min_mse = mse
                best_frame = frame_idx
        frame_idx += 1
    cap.release()
    print(f"  - Found anchor: Video Frame {best_frame} = {ref_nano} ns")
    return best_frame, ref_nano

def process_run(run_dir, offset):
    print(f"\n--- Processing: {os.path.basename(run_dir)} ---")
    raw_dir = os.path.join(run_dir, "raw")
    proc_dir = os.path.join(run_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)
    
    video_files = glob.glob(os.path.join(raw_dir, "*.mp4"))
    if not video_files:
        print(f"  ! No video found in {raw_dir}. Skipping."); return
    video = video_files[0]
    
    jpgs = sorted(glob.glob(os.path.join(raw_dir, "*/*.jpg")))
    if not jpgs: jpgs = sorted(glob.glob(os.path.join(raw_dir, "*.jpg")))
    if not jpgs:
        print(f"  ! No JPG captures found. Skipping."); return

    v_frame, p_nano = find_anchor(video, jpgs)
    
    ts_files = glob.glob(os.path.join(raw_dir, "*/*_imu_timestamps.csv"))
    if ts_files:
        ts_df = pd.read_csv(ts_files[0], header=None, names=['nano'])
        ts_df['Unix_Time'] = (ts_df['nano'] / 1e9) + offset
        ts_df.to_csv(os.path.join(proc_dir, "01_frame_times_unix.csv"), index=True, index_label='Frame')

    sensors = ['accel', 'gyro', 'magnetic']
    for s in sensors:
        s_file = glob.glob(os.path.join(raw_dir, f"*/*{s}.csv"))
        if s_file:
            print(f"  - Converting {s} to Unix time...")
            df = pd.read_csv(s_file[0], header=None)
            time_col = df.columns[-1]
            df.insert(0, 'Unix_Time', (df[time_col] / 1e9) + offset)
            df.to_csv(os.path.join(proc_dir, f"01_{s}_unix.csv"), index=False)
    print(f"Done.")

def main():
    base_data_path = str(DATA)
    
    # 1. Select Session (Day)
    sessions = sorted([d for d in os.listdir(base_data_path) if glob.glob(os.path.join(base_data_path, d, "run_*", "raw")) and d != 'apriltag_recording' and d != 'old'])
    
    if not sessions:
        print("No session folders (data/<session>/run_*/raw) found in data/")
        return

    print("\nAvailable Sessions:")
    for i, s in enumerate(sessions):
        print(f"  [{i+1}] {s}")
    
    s_choice = input("\nSelect Session Number: ").strip()
    try:
        session_name = sessions[int(s_choice)-1]
    except:
        print("Invalid selection."); return

    # 2. Select Run
    session_path = os.path.join(base_data_path, session_name)
    runs = sorted([d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))])
    
    print(f"\nRuns in {session_name}:")
    for i, r in enumerate(runs):
        print(f"  [{i+1}] {r}")
    
    r_choice = input("\nEnter Run Number (or 'all'): ").strip().lower()
    offset = get_offset()
    
    if r_choice == 'all':
        for r in runs: process_run(os.path.join(session_path, r), offset)
    else:
        try:
            process_run(os.path.join(session_path, runs[int(r_choice)-1]), offset)
        except:
            print("Invalid selection.")

if __name__ == "__main__":
    main()
