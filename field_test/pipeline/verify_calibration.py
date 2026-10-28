import cv2
import numpy as np
import argparse
import os
import json

def verify(video_path, camera=None):
    CONFIG_FILE = "field_test/vision_settings.json"
    with open(CONFIG_FILE, 'r') as f:
        data = json.load(f)
        camera_name = camera or 'overhead'
        cam = data.get('cameras', {}).get(camera_name, {})
        cp = cam.get('camera_params', [1310.0, 1310.0, 960.0, 540.0])
        dc = cam.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
    print(f"Using camera: {camera_name}")

    mtx = np.array([[cp[0], 0, cp[2]], [0, cp[1], cp[3]], [0, 0, 1]])
    dist = np.array(dc)

    cap = cv2.VideoCapture(video_path)
    
    print("\n--- Visual Verification Mode ---")
    print(" 1. Look at the edges of the laptop screen.")
    print(" 2. They should be PERFECTLY STRAIGHT.")
    print(" 3. Press 'q' to exit.")

    cv2.namedWindow('Calibration Verification', cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Apply Undistortion
        h, w = frame.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(frame, mtx, dist, None, new_mtx)
        
        # Draw a help grid to check straightness
        for x in range(0, w, 100): cv2.line(dst, (x, 0), (x, h), (50, 50, 50), 1)
        for y in range(0, h, 100): cv2.line(dst, (0, y), (w, y), (50, 50, 50), 1)

        cv2.imshow('Calibration Verification', dst)
        if cv2.waitKey(30) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--camera", default=None, help="Camera name from vision_settings.json (e.g. simonPhone, overhead)")
    args = parser.parse_args()
    verify(args.video, camera=args.camera)
