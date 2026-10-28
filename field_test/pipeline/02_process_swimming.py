import pandas as pd
import numpy as np
import os
import glob
import subprocess
import argparse
import cv2
import json
import sys
import tempfile
from dt_apriltags import Detector
from scipy.spatial.transform import Rotation as R
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import DATA  # noqa: E402

# Config path
CONFIG_FILE = "field_test/vision_settings.json"

def load_settings(run_name, camera_override=None):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            settings = data.get("default", {}).copy()
            settings.update(data.get(run_name, {}))
            camera_name = camera_override or 'overhead'
            cam = data.get('cameras', {}).get(camera_name, {})
            settings['camera_params'] = cam.get('camera_params', [1310.0, 1310.0, 960.0, 540.0])
            settings['dist_coeffs'] = cam.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
            return settings
    return {}

def rot_around_z(v, deg):
    """Rotate vector v around the Z axis (optical axis) by deg degrees."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ v

def get_rotation_matrix(R_raw, g_norm):
    # 1. World Y is Gravity (DOWN) — IMU reports gravity vector, not reaction force
    new_y = g_norm / np.linalg.norm(g_norm) 
    
    # 2. Use the Tag's Y-axis (the 'Top' of the printed square) as 'North'
    # This is much more stable than Z because it's perpendicular to gravity
    tag_top = R_raw[:, 1] 
    
    # 3. Create orthonormal basis
    new_x = np.cross(tag_top, new_y)
    new_x /= np.linalg.norm(new_x)
    new_z = np.cross(new_x, new_y)
    
    return np.stack([new_x, new_y, new_z], axis=1)

def process_swimming(run_dir, tag_size_mm=100.0, camera=None):
    run_name = os.path.basename(run_dir)
    print(f"\n--- Master Processing (Full 6DOF): {run_name} ---")
    raw_dir, proc_dir = os.path.join(run_dir, "raw"), os.path.join(run_dir, "processed")
    os.makedirs(proc_dir, exist_ok=True)

    settings = load_settings(run_name, camera_override=camera)
    alpha, beta = settings.get('alpha', 1.0), settings.get('beta', 0)
    blur, decimate = settings.get('blur', 1), settings.get('decimate', 2.0)
    cp = settings.get('camera_params', [1310.0, 1310.0, 960.0, 540.0])
    dc = settings.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
    
    mtx = np.array([[cp[0], 0, cp[2]], [0, cp[1], cp[3]], [0, 0, 1]])
    dist = np.array(dc)

    # 1. Load Sync & Logs
    sync_file = os.path.join(proc_dir, "01_frame_times_unix.csv")
    if not os.path.exists(sync_file): return
    sync_df = pd.read_csv(sync_file)
    fish_files = glob.glob(os.path.join(raw_dir, "20260401_*.csv"))
    if not fish_files: return
    fish_df = pd.read_csv(fish_files[0], comment='#')
    accel_df = pd.read_csv(os.path.join(proc_dir, "01_accel_unix.csv"))
    
    start_unix = fish_df['Unix_Time'].iloc[0]
    start_frame = sync_df[sync_df['Unix_Time'] >= start_unix].index[0]

    # 2. Setup Video
    video_path = glob.glob(os.path.join(raw_dir, "*.mp4"))[0]
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    temp_video = os.path.join(proc_dir, "temp_presentation.mp4")
    out = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    detector = Detector(families='tag36h11', nthreads=1, quad_decimate=decimate)
    avg_g = rot_around_z(accel_df[['0', '1', '2']].mean().values, +90.0)  # landscape CCW fix
    raw_data, fish_trail = [], []
    last_fish_unix = start_unix

    print(f"  - Extracting frames (Full 6-DOF Path)...")
    
    tfile = tempfile.TemporaryFile(mode='w+')
    old_stderr = os.dup(sys.stderr.fileno())
    
    try:
        os.dup2(tfile.fileno(), sys.stderr.fileno())
        for i in range(start_frame, len(sync_df)):
            ret, frame = cap.read()
            if not ret: break
            
            # --- MATH PATH ---
            undistorted = cv2.undistort(frame, mtx, dist)
            proc_detect = cv2.convertScaleAbs(undistorted, alpha=alpha, beta=beta)
            if blur > 1: proc_detect = cv2.GaussianBlur(proc_detect, (blur, blur), 0)
            gray = cv2.cvtColor(proc_detect, cv2.COLOR_BGR2GRAY)
            tags = detector.detect(gray, estimate_tag_pose=True, camera_params=cp, tag_size=tag_size_mm/1000.0)
            
            frame_unix = sync_df.loc[i, 'Unix_Time']
            f_data = {'Frame': i, 'Unix_Time': frame_unix, 'ID0_Seen': False, 'ID1_Seen': False}
            for t in tags:
                # Back-projection for visuals
                s = (tag_size_mm / 1000.0) / 2
                obj_pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=np.float32)
                rvec, _ = cv2.Rodrigues(t.pose_R)
                img_pts_warped, _ = cv2.projectPoints(obj_pts, rvec, t.pose_t, mtx, dist)
                warped_corners = img_pts_warped.reshape(-1, 2).astype(int)

                color = (255, 0, 0) if t.tag_id == 0 else (0, 255, 0)
                cv2.polylines(frame, [warped_corners], True, color, 3)
                cv2.putText(frame, f"ID{t.tag_id}", tuple(warped_corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                
                pref = f"ID{t.tag_id}"
                f_data[f'{pref}_Seen'] = True
                f_data[f'{pref}_t'] = t.pose_t.flatten()
                f_data[f'{pref}_R'] = t.pose_R
                f_data[f'{pref}_err'] = t.decision_margin # Detection quality
                
                if t.tag_id == 1:
                    last_fish_unix = f_data['Unix_Time']
                    warped_center = np.mean(warped_corners, axis=0).astype(int)
                    fish_trail.append(tuple(warped_center))

            if len(fish_trail) > 1:
                for k in range(1, len(fish_trail)): cv2.line(frame, fish_trail[k-1], fish_trail[k], (0, 255, 255), 2)

            out.write(frame)
            raw_data.append(f_data)
            if len(raw_data) % 100 == 0:
                os.dup2(old_stderr, sys.stderr.fileno())
                sys.stdout.write(f"    Frame {i}\r"); sys.stdout.flush()
                os.dup2(tfile.fileno(), sys.stderr.fileno())
    finally:
        os.dup2(old_stderr, sys.stderr.fileno()); tfile.seek(0)
        ambiguity_count = tfile.read().count("more than one new minimum found")
        tfile.close(); cap.release(); out.release()

    # 3. COORDINATE TRANSFORMATION (Center at Floor Tag)
    id0_logs = [d for d in raw_data if d['ID0_Seen']]
    if not id0_logs: print("\nError: ID 0 (Floor) not found."); return
    world_origin = np.mean([d['ID0_t'] for d in id0_logs], axis=0)
    R_cam_to_world_mat = get_rotation_matrix(np.mean([d['ID0_R'] for d in id0_logs], axis=0), avg_g).T
    
    processed_vision = []
    for d in raw_data:
        row = {'Unix_Time': d['Unix_Time'], 'Frame': d['Frame'], 'ID1_Seen': d['ID1_Seen'], 'ID0_Seen': d['ID0_Seen']}
        if d['ID1_Seen']:
            # Position
            p_world = R_cam_to_world_mat @ (d['ID1_t'] - world_origin)
            row['X'], row['Y'], row['Z'] = p_world
            # Orientation
            R_fish_to_world = R_cam_to_world_mat @ d['ID1_R']
            euler = R.from_matrix(R_fish_to_world).as_euler('xyz', degrees=True)
            row['Roll'], row['Pitch'] = euler[0], euler[2]
            # Yaw: extract heading in world XZ plane directly to avoid xyz Euler gimbal
            # lock singularity at 90deg (which the new world frame sits near).
            # Tag col-1 (tag Y = "top" of printed square) points along the swim axis.
            fwd = R_fish_to_world[:, 1]
            row['Yaw'] = np.degrees(np.arctan2(fwd[0], fwd[2]))
            row['Quality'] = d['ID1_err']
        else: row['X'] = row['Y'] = row['Z'] = row['Yaw'] = row['Pitch'] = row['Roll'] = row['Quality'] = np.nan
        processed_vision.append(row)

    vis_df = pd.DataFrame(processed_vision)
    
    # 4. TRIM & SAVE
    mask = (vis_df['Unix_Time'] <= last_fish_unix)
    swim_vis = vis_df[mask]
    swim_vis[['Unix_Time', 'X', 'Y', 'Z', 'Roll', 'Yaw', 'Pitch', 'Quality']].to_csv(os.path.join(proc_dir, "02_clean_path.csv"), index=False)
    trimmed_fish_df = fish_df[(fish_df['Unix_Time'] >= start_unix) & (fish_df['Unix_Time'] <= last_fish_unix)]
    trimmed_fish_df.to_csv(os.path.join(proc_dir, "02_fish_log_trimmed.csv"), index=False)

    vid_dur = last_fish_unix - start_unix
    subprocess.run(["ffmpeg", "-y", "-i", temp_video, "-t", str(vid_dur), "-c:v", "libx264", "-crf", "23", os.path.join(proc_dir, "02_swim_clip.mp4")], capture_output=True)
    os.remove(temp_video)
    
    # 5. QUALITY REPORT
    total_f = len(swim_vis); fish_missed = swim_vis[swim_vis['ID1_Seen'] == False]['Frame'].tolist(); floor_missed = swim_vis[swim_vis['ID0_Seen'] == False]['Frame'].tolist()
    with open(os.path.join(proc_dir, "02_quality_report.md"), 'w') as f:
        f.write(f"# Quality Report: {run_name}\n\n")
        f.write(f"## 🕒 Timing Boundaries (Unix)\n")
        f.write(f"- **Window Start:** {start_unix:.6f}\n")
        f.write(f"- **Window End:** {last_fish_unix:.6f}\n")
        f.write(f"- **Fish Log Range:** {trimmed_fish_df['Unix_Time'].iloc[0]:.6f} to {trimmed_fish_df['Unix_Time'].iloc[-1]:.6f}\n")
        f.write(f"- **Vision Path Range:** {swim_vis['Unix_Time'].iloc[0]:.6f} to {swim_vis['Unix_Time'].iloc[-1]:.6f}\n\n")
        f.write(f"## 📋 Summary\n- **Duration:** {vid_dur:.2f}s\n- **Total Frames:** {total_f}\n- **AprilTag Ambiguity Errors:** {ambiguity_count}\n\n")
        f.write(f"### 🐟 Fish Tag (ID 1)\n- **Success Rate:** {(total_f - len(fish_missed))/total_f*100:.1f}%\n- **Missing:** {len(fish_missed)}\n- **Frames:** {', '.join(map(str, fish_missed)) if fish_missed else 'None'}\n\n")
        f.write(f"### 🧱 Floor Tag (ID 0)\n- **Success Rate:** {(total_f - len(floor_missed))/total_f*100:.1f}%\n- **Missing:** {len(floor_missed)}\n")

    print(f"\n  - Done. Output with 6DOF (Yaw/Pitch/Roll) generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", nargs="?", help="Path to a run directory (optional)")
    parser.add_argument("--camera", default=None, help="Camera name from vision_settings.json (e.g. simonPhone, overhead)")
    args = parser.parse_args()

    if args.run_dir and os.path.isdir(args.run_dir):
        process_swimming(args.run_dir, camera=args.camera); sys.exit(0)

    base_data = str(DATA)
    sessions = sorted([d for d in os.listdir(base_data) if glob.glob(os.path.join(base_data, d, "run_*", "raw")) and d not in ['old', 'apriltag_recording']])
    print("\nAvailable Sessions:"); [print(f"  [{i+1}] {s}") for i, s in enumerate(sessions)]
    s_choice = input("\nSelect Session Number: ").strip()
    try:
        session_idx = int(s_choice) - 1
        if not (0 <= session_idx < len(sessions)): raise ValueError
        session_path = os.path.join(base_data, sessions[session_idx])
    except:
        print("Invalid session number."); sys.exit(1)

    runs = sorted([d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))])
    print(f"\nRuns:"); [print(f"  [{i+1}] {r}") for i, r in enumerate(runs)]
    r_choice = input("\nEnter Run Number (or 'all'): ").strip().lower()

    camera_args = ["--camera", args.camera] if args.camera else []
    if r_choice == 'all':
        for r in runs: subprocess.run([sys.executable, __file__, os.path.join(session_path, r)] + camera_args)
    else:
        try:
            run_idx = int(r_choice) - 1
            if not (0 <= run_idx < len(runs)): raise ValueError
            process_swimming(os.path.join(session_path, runs[run_idx]), camera=args.camera)
        except:
            print("Invalid run number."); sys.exit(1)
