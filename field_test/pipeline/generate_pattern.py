import cv2
import numpy as np
import argparse
import math

def generate(diag_inch, res_w, res_h):
    # 1. Calculate Pixel Pitch
    ratio = res_w / res_h
    h_inch = diag_inch / math.sqrt(ratio**2 + 1)
    w_inch = h_inch * ratio
    w_mm = w_inch * 25.4
    pixel_pitch = w_mm / res_w
    
    print(f"--- Screen Analysis ---")
    print(f"  Physical Size: {w_mm:.1f} x {h_inch*25.4:.1f} mm")
    print(f"  Pixel Pitch:   {pixel_pitch:.4f} mm/px")

    # 2. Define Large Landscape Pattern
    # 11 columns x 8 rows fits well on a 4K 16:9 screen
    cols, rows = 11, 8
    px_per_square = 250
    px_per_marker = 190 
    
    checker_size_mm = px_per_square * pixel_pitch
    marker_size_mm = px_per_marker * pixel_pitch
    
    # 3. Create ChArUco Board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((cols, rows), px_per_square, px_per_marker, aruco_dict)
    
    # 4. Generate Image
    # Total px: 2750 x 2000
    img = board.generateImage((cols * px_per_square, rows * px_per_square))
    
    output_file = "calibration_pattern.png"
    cv2.imwrite(output_file, img)
    
    print(f"\n--- Larger Pattern Generated ---")
    print(f"  File: {output_file} ({cols*px_per_square}x{rows*px_per_square} px)")
    print(f"  Checker Size: {checker_size_mm:.4f} mm")
    print(f"  Marker Size:  {marker_size_mm:.4f} mm")
    print(f"\n>> IMPORTANT: Open at 100% ZOOM (Ctrl+1 in Windows Photos).")
    print(f">> Use these exact numbers in calibrate_camera.py:")
    print(f"   --checker_size {checker_size_mm:.4f} --marker_size {marker_size_mm:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diag", type=float, required=True)
    parser.add_argument("--res_w", type=int, required=True)
    parser.add_argument("--res_h", type=int, required=True)
    args = parser.parse_args()
    generate(args.diag, args.res_w, args.res_h)
