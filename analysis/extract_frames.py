#!/usr/bin/env python3
import os
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.abspath(os.path.join(_HERE, "..", "deploy", "runs"))
OUT_DIR = os.path.join(_HERE, "insane_run_8_frames")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    run_name = "insane_run_8"
    folders = [f for f in os.listdir(RUNS_DIR) if run_name in f]
    if not folders:
        print("Run folder not found")
        return
    run_folder = folders[0]
    vid_path = os.path.join(RUNS_DIR, run_folder, "raw", "video.avi")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(RUNS_DIR, run_folder, "raw", "video.mp4")
        
    if not os.path.exists(vid_path):
        print("Video not found")
        return
        
    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract 20 equally spaced frames
    indices = np.linspace(0, total - 1, 20).astype(int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Draw frame number clearly
            cv2.putText(frame, f"Frame {idx}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 10, cv2.LINE_AA)
            cv2.imwrite(os.path.join(OUT_DIR, f"frame_{idx}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(f"Saved frame {idx}")

    print("Done generating frames")

if __name__ == "__main__":
    main()
