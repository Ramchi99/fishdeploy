import cv2
import numpy as np
import argparse
import os
import sys
import shutil
from tqdm import tqdm

def calibrate_charuco(video_path, checker_size, marker_size):
    print(f"\n--- Professional ChArUco Calibration (Two-Pass Outlier Removal) ---")
    
    # Setup Paths
    evidence_dir = "field_test/pipeline/calibration_evidence"
    if os.path.exists(evidence_dir): shutil.rmtree(evidence_dir)
    os.makedirs(evidence_dir)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((11, 8), checker_size, marker_size, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)

    all_samples = [] 
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imsize = None

    print(f"Pass 1: Scanning {total_frames} frames for spatial coverage...")
    
    with tqdm(total=total_frames, desc="Scanning", unit="frame") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if imsize is None: imsize = gray.shape[::-1]
                charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
                if charuco_ids is not None and len(charuco_ids) >= 10:
                    all_samples.append({
                        'corners': charuco_corners,
                        'ids': charuco_ids,
                        'pos': np.mean(charuco_corners, axis=0).flatten(), 
                        'frame': frame.copy(),
                        'count': len(charuco_ids)
                    })
            frame_idx += 1
            pbar.update(1)
            pbar.set_postfix({"Found": len(all_samples)})
    cap.release()

    # --- PROFESSIONAL COVERAGE FILTERING ---
    # We divide the screen into a 3x3 grid and pick the best frames from each sector
    print(f"\nSector Filtering (Increasing budget to 100 per area)...")
    sector_samples = []
    w, h = imsize
    for sx in range(3):
        for sy in range(3):
            # Find samples in this sector
            in_sector = [s for s in all_samples if 
                         (sx*w/3 <= s['pos'][0] < (sx+1)*w/3) and 
                         (sy*h/3 <= s['pos'][1] < (sy+1)*h/3)]
            # Sort by corner count and take top 100 from each sector
            in_sector.sort(key=lambda x: x['count'], reverse=True)
            sector_samples.extend(in_sector[:100])
    
    def run_calib(samples):
        obj_points, img_points = [], []
        for s in samples:
            obj_pts, img_pts = board.matchImagePoints(s['corners'], s['ids'])
            if obj_pts is not None and len(obj_pts) > 0:
                obj_points.append(obj_pts)
                img_points.append(img_pts)
        return cv2.calibrateCamera(obj_points, img_points, imsize, None, None)

    print(f"Computing Pass 1 ({len(sector_samples)} frames)...")
    ret, mtx, dist, rvecs, tvecs = run_calib(sector_samples)
    print(f"  - Initial Reprojection Error: {ret:.6f}")

    # --- PASS 2: OUTLIER REMOVAL ---
    print(f"\nPass 2: Removing noisy frames...")
    frame_errors = []
    board_corners = board.getChessboardCorners()
    
    for i in range(len(sector_samples)):
        # Project 3D board points to 2D
        ids = sector_samples[i]['ids'].flatten()
        imgpts, _ = cv2.projectPoints(board_corners[ids], rvecs[i], tvecs[i], mtx, dist)
        
        # Ensure types and shapes match for cv2.norm
        p1 = sector_samples[i]['corners'].astype(np.float32)
        p2 = imgpts.reshape(-1, 1, 2).astype(np.float32)
        
        err = cv2.norm(p1, p2, cv2.NORM_L2) / len(imgpts)
        frame_errors.append(err)
    
    # Throw away frames with error > 1.5x the median
    limit = np.median(frame_errors) * 1.5
    final_samples = [sector_samples[i] for i in range(len(sector_samples)) if frame_errors[i] < limit]
    
    print(f"  - Removed {len(sector_samples) - len(final_samples)} outliers (Error limit: {limit:.4f})")
    
    # Save Final Evidence
    for i, s in enumerate(final_samples):
        draw = s['frame'].copy()
        cv2.aruco.drawDetectedCornersCharuco(draw, s['corners'], s['ids'])
        cv2.imwrite(os.path.join(evidence_dir, f"sample_{i:03d}.jpg"), draw)

    print(f"Computing Final Intrinsics from {len(final_samples)} clean frames...")
    ret, mtx, dist, rvecs, tvecs = run_calib(final_samples)

    print("\n==========================================")
    print("   MASTER CAMERA INTRINSICS (CERTIFIED)   ")
    print("==========================================")
    print(f" Final Reprojection Error: {ret:.6f}")
    print(f" Clean Data Points:        {len(final_samples)}")
    print(f"\n fx: {mtx[0,0]:.4f}")
    print(f" fy: {mtx[1,1]:.4f}")
    print(f" cx: {mtx[0,2]:.4f}")
    print(f" cy: {mtx[1,2]:.4f}")
    print(f"\n Distortion Coefficients:")
    print(f" {dist.flatten().tolist()}")
    print("==========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--checker_size", type=float, required=True)
    parser.add_argument("--marker_size", type=float, required=True)
    args = parser.parse_args()
    calibrate_charuco(args.video, args.checker_size, args.marker_size)
