import cv2
import numpy as np
import argparse
import os
import json
from dt_apriltags import Detector

# Config path
CONFIG_FILE = "field_test/vision_settings.json"

def load_settings(run_name):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            # Start with defaults
            settings = data.get("default", {}).copy()
            # Overwrite with run-specific if they exist
            settings.update(data.get(run_name, {}))
            return settings
    return {}

def save_settings(run_name, settings):
    data = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
    data[run_name] = settings
    if "default" not in data: data["default"] = settings
    with open(CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n>> Settings saved to {CONFIG_FILE}")

def nothing(x): pass

def tune_vision(video_path):
    run_name = os.path.basename(os.path.dirname(os.path.dirname(video_path)))
    current = load_settings(run_name)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at: {video_path}")
        print("Check your path and ensure you are using forward slashes (/).")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Error: Video file has 0 frames or is corrupt.")
        return
    
    detector = Detector(families='tag36h11', nthreads=1)
    
    cv2.namedWindow('Sofi Vision Tuner', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Timeline', 'Sofi Vision Tuner', 0, total_frames-1, nothing)
    cv2.createTrackbar('Contrast', 'Sofi Vision Tuner', int(current.get('alpha', 1.0)*10), 30, nothing)
    cv2.createTrackbar('Brightness', 'Sofi Vision Tuner', int(current.get('beta', 0)+50), 100, nothing)
    cv2.createTrackbar('Blur', 'Sofi Vision Tuner', int((current.get('blur', 1)-1)/2), 10, nothing)
    cv2.createTrackbar('Decimate', 'Sofi Vision Tuner', int(current.get('decimate', 2)), 4, nothing)

    print(f"\n--- Tuning Run: {run_name} ---")
    print(" [CONTROLS]")
    print("  - SPACEBAR : Play / Pause")
    print("  - SLIDERS  : Tune Detection")
    print("  - 'q'      : Save and Exit")

    is_playing = False
    frame_idx = 0

    while True:
        # 1. Handle Playback logic
        if is_playing:
            frame_idx += 1
            if frame_idx >= total_frames: frame_idx = 0
            cv2.setTrackbarPos('Timeline', 'Sofi Vision Tuner', frame_idx)
        else:
            # If paused, get position from slider
            frame_idx = cv2.getTrackbarPos('Timeline', 'Sofi Vision Tuner')

        # 2. Get Settings
        alpha = cv2.getTrackbarPos('Contrast', 'Sofi Vision Tuner') / 10.0
        beta = cv2.getTrackbarPos('Brightness', 'Sofi Vision Tuner') - 50
        blur_val = cv2.getTrackbarPos('Blur', 'Sofi Vision Tuner') * 2 + 1
        decimate = float(cv2.getTrackbarPos('Decimate', 'Sofi Vision Tuner'))
        if decimate < 1.0: decimate = 1.0

        # 3. Seek and Read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # 4. Process & Detect
        proc = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        if blur_val > 1: proc = cv2.GaussianBlur(proc, (blur_val, blur_val), 0)
        
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        detector.quad_decimate = decimate
        tags = detector.detect(gray)

        # 5. Drawing
        display = proc.copy()
        # Add Status Overlay
        status_text = "PLAYING" if is_playing else "PAUSED"
        cv2.putText(display, f"STATUS: {status_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for t in tags:
            color = (255, 0, 0) if t.tag_id == 0 else (0, 255, 0)
            for i in range(4):
                cv2.line(display, tuple(t.corners[i].astype(int)), tuple(t.corners[(i+1)%4].astype(int)), color, 3)
            cv2.putText(display, f"ID:{t.tag_id}", tuple(t.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow('Sofi Vision Tuner', display)
        
        # Binary Threshold Preview
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow('Detector Binary View', thresh)

        # 6. Key Handlers
        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '): # Spacebar
            is_playing = not is_playing
        elif key == ord('q'):
            best_settings = {'alpha': alpha, 'beta': beta, 'blur': blur_val, 'decimate': decimate}
            save_settings(run_name, best_settings)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    tune_vision(args.video)
