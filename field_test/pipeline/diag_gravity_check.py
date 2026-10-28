#!/usr/bin/env python3
"""
Gravity Calibration Diagnostic
================================
Compares 4 candidate 'UP' vectors in camera frame:

  1. Accel Raw    : average accelerometer as-is (current approach in 02_process_swimming.py)
  2. Accel +90°   : accel rotated +90° around optical axis (Z) — landscape CCW hypothesis
  3. Accel -90°   : accel rotated -90° around optical axis (Z) — landscape CW hypothesis
  4. Floor Tag    : mean of ID0 Z-axis (tag normal) from AprilTag detections

The method whose accel variant best matches the floor tag normal tells you whether
the landscape-mode axis swap is real and in which direction.

Variance comparison reveals which source is more stable.

Usage:
  python3 diag_gravity_check.py <run_dir>
  python3 diag_gravity_check.py data/01_field_openloop_20260401/run_03_1.0Hz_9.0mm
"""

import sys
import os
import glob
import json
import tempfile
import numpy as np
import pandas as pd
import cv2
from dt_apriltags import Detector

CONFIG_FILE = "field_test/vision_settings.json"
MAX_ID0_DETECTIONS = 200   # stop early once we have enough floor tag samples


def load_settings(run_name, camera_override=None):
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            data = json.load(f)
            s = data.get("default", {}).copy()
            s.update(data.get(run_name, {}))
            camera_name = camera_override or 'overhead'
            cam = data.get('cameras', {}).get(camera_name, {})
            s['camera_params'] = cam.get('camera_params', [1310.0, 1310.0, 960.0, 540.0])
            s['dist_coeffs'] = cam.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
            return s
    return {}


def rot_around_z(v, deg):
    """Rotate 3-vector v around the Z axis by deg degrees (right-hand rule)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])
    return R @ v


def angle_deg(v1, v2):
    """Angle in degrees between two vectors (ignores sign / direction)."""
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)
    dot = np.clip(abs(np.dot(u1, u2)), 0.0, 1.0)   # abs → treats opposite dirs as same axis
    return np.degrees(np.arccos(dot))


def angular_std(vectors):
    """
    Angular std dev (degrees) of a set of unit-normalised vectors relative to their mean.
    Each vector is treated as a direction (sign ignored).
    """
    normed = np.array([v / np.linalg.norm(v) for v in vectors])
    mean_v = normed.mean(axis=0)
    mean_v /= np.linalg.norm(mean_v)
    # Use abs(dot) so opposite-pointing vectors still count as close
    angles = np.degrees(np.arccos(np.clip(np.abs(normed @ mean_v), 0, 1)))
    return float(np.std(angles))


def run_comparison(run_dir, camera=None):
    run_name = os.path.basename(run_dir)
    proc_dir = os.path.join(run_dir, "processed")
    raw_dir  = os.path.join(run_dir, "raw")

    print(f"\n{'='*60}")
    print(f"  Gravity Calibration Diagnostic: {run_name}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 1. Accelerometer                                                     #
    # ------------------------------------------------------------------ #
    accel_path = os.path.join(proc_dir, "01_accel_unix.csv")
    if not os.path.exists(accel_path):
        print("ERROR: 01_accel_unix.csv not found. Run 01_sync_and_convert.py first.")
        sys.exit(1)

    accel_df  = pd.read_csv(accel_path)
    g_samples = accel_df[['0', '1', '2']].values       # (N, 3) — raw IMU axes
    avg_g     = g_samples.mean(axis=0)
    g_mag     = np.linalg.norm(avg_g)

    # Variance: subsample every 5th row for speed
    accel_std = angular_std(g_samples[::5])

    print("─── Accelerometer ─────────────────────────────────────────")
    print(f"  Columns 0,1,2 raw mean : ({avg_g[0]:+.4f}, {avg_g[1]:+.4f}, {avg_g[2]:+.4f})")
    print(f"  Magnitude              : {g_mag:.4f} m/s²  (expect ~9.81)")
    print(f"  Samples                : {len(g_samples)}")
    print(f"  Angular std dev        : {accel_std:.4f}°\n")

    # ------------------------------------------------------------------ #
    # 2. Three accel candidate "UP" vectors                               #
    # ------------------------------------------------------------------ #
    g_raw = avg_g
    g_p90 = rot_around_z(avg_g, +90.0)
    g_m90 = rot_around_z(avg_g, -90.0)

    accel_candidates = {
        "Accel Raw  ": g_raw,
        "Accel +90° ": g_p90,
        "Accel -90° ": g_m90,
    }

    # ------------------------------------------------------------------ #
    # 3. Floor tag normals from video                                      #
    # ------------------------------------------------------------------ #
    video_files = glob.glob(os.path.join(raw_dir, "*.mp4"))
    if not video_files:
        print("ERROR: No .mp4 found in raw/"); sys.exit(1)

    settings = load_settings(run_name, camera_override=camera)
    alpha    = settings.get('alpha', 1.0)
    beta     = settings.get('beta', 0)
    blur     = settings.get('blur', 1)
    decimate = settings.get('decimate', 1.0)
    cp       = settings.get('camera_params', [1310.0, 1310.0, 960.0, 540.0])
    dc       = settings.get('dist_coeffs', [0.0, 0.0, 0.0, 0.0, 0.0])
    mtx      = np.array([[cp[0], 0, cp[2]], [0, cp[1], cp[3]], [0, 0, 1]])
    dist     = np.array(dc)

    print("─── Floor Tag (ID0) normal scan ───────────────────────────")
    print(f"  Scanning video for up to {MAX_ID0_DETECTIONS} ID0 detections...")

    cap      = cv2.VideoCapture(video_files[0])
    detector = Detector(families='tag36h11', nthreads=1, quad_decimate=decimate)
    id0_normals   = []   # R[:, 2] = tag Z-axis in camera frame (normal direction)
    id0_tag_tops  = []   # R[:, 1] = tag Y-axis (top of printed square)
    total_frames  = 0

    # Suppress dt_apriltags stderr chatter
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(sys.stderr.fileno())
    os.dup2(devnull_fd, sys.stderr.fileno())

    try:
        while cap.isOpened() and len(id0_normals) < MAX_ID0_DETECTIONS:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1

            if total_frames % 2 != 0:   # check every 2nd frame
                continue

            undist = cv2.undistort(frame, mtx, dist)
            proc   = cv2.convertScaleAbs(undist, alpha=alpha, beta=beta)
            if blur > 1:
                proc = cv2.GaussianBlur(proc, (blur, blur), 0)
            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

            tags = detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=cp,
                tag_size=0.1          # 100 mm floor tag
            )
            for t in tags:
                if t.tag_id == 0:
                    id0_normals.append(t.pose_R[:, 2].copy())
                    id0_tag_tops.append(t.pose_R[:, 1].copy())
    finally:
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(devnull_fd)
        os.close(old_stderr)
        cap.release()

    print(f"  Scanned {total_frames} frames → {len(id0_normals)} ID0 detections")

    if not id0_normals:
        print("  ERROR: No ID0 detections. Cannot compute floor tag reference.")
        sys.exit(1)

    normals_arr  = np.array(id0_normals)
    avg_normal   = normals_arr.mean(axis=0)

    # Make avg_normal point in the same hemisphere as avg_g
    # (both should point "upward" in camera frame; if one is flipped, dot product < 0)
    if np.dot(avg_normal, avg_g) < 0:
        avg_normal = -avg_normal
        normals_arr = -normals_arr
        print("  Note: flipped ID0 normals to match accelerometer hemisphere.")

    tag_std = angular_std(normals_arr)

    print(f"  avg normal (cam frame) : ({avg_normal[0]:+.4f}, {avg_normal[1]:+.4f}, {avg_normal[2]:+.4f})")
    print(f"  Angular std dev        : {tag_std:.4f}°\n")

    # ------------------------------------------------------------------ #
    # 4. Pairwise comparison table                                         #
    # ------------------------------------------------------------------ #
    all_candidates = {**accel_candidates, "Floor Tag  ": avg_normal}

    print("─── Angle between all candidates (degrees) ────────────────")
    col_w = 13
    header = f"{'':16}" + "".join(f"{n:{col_w}}" for n in all_candidates)
    print(header)
    names = list(all_candidates.keys())
    vecs  = list(all_candidates.values())
    for i, ni in enumerate(names):
        row = f"  {ni:14}"
        for j, nj in enumerate(names):
            if i == j:
                row += f"{'---':{col_w}}"
            elif j < i:
                row += f"{'':{ col_w}}"
            else:
                a = angle_deg(vecs[i], vecs[j])
                row += f"{a:.2f}°{' ':{col_w - 7}}"
        print(row)

    # ------------------------------------------------------------------ #
    # 5. Best accel match to floor tag                                     #
    # ------------------------------------------------------------------ #
    print("\n─── Best accel match to Floor Tag Normal ──────────────────")
    floor_v = avg_normal
    diffs   = {n: angle_deg(v, floor_v) for n, v in accel_candidates.items()}
    best    = min(diffs, key=diffs.get)

    for n, a in diffs.items():
        marker = "  ← BEST MATCH" if n == best else ""
        print(f"  {n:14}  {a:.3f}°{marker}")

    print()
    print(f"  Accel stability   : {accel_std:.4f}° std dev")
    print(f"  Floor tag stability: {tag_std:.4f}° std dev")

    winner = "Accel" if accel_std < tag_std else "Floor Tag"
    print(f"  → '{winner}' is more stable (lower variance).\n")

    # Interpretation
    print("─── Interpretation ─────────────────────────────────────────")
    best_angle = diffs[best]
    if best_angle < 1.0:
        print(f"  ✓ '{best.strip()}' matches Floor Tag within {best_angle:.2f}°.")
        if "Raw" in best:
            print("    Accel axes are already aligned with camera frame.")
            print("    Landscape rotation was NOT the cause of the tilt error.")
        elif "+90" in best:
            print("    Landscape rotation of +90° (CCW) accounts for the mismatch.")
            print("    Fix: rotate avg_g by +90° around Z before using in get_rotation_matrix().")
        elif "-90" in best:
            print("    Landscape rotation of -90° (CW) accounts for the mismatch.")
            print("    Fix: rotate avg_g by -90° around Z before using in get_rotation_matrix().")
    elif best_angle < 3.0:
        print(f"  ~ '{best.strip()}' is closest to Floor Tag ({best_angle:.2f}°), but not a perfect match.")
        print("    May need fine-tuning or a different rotation angle.")
    else:
        print(f"  ✗ No accel variant matches Floor Tag well (best: {best_angle:.2f}°).")
        print("    Possible causes: floor not flat, tag not flat, large accel bias,")
        print("    or a non-90° axis rotation (e.g. camera mounted at unusual angle).")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--camera", default=None, help="Camera name from vision_settings.json (e.g. simonPhone, overhead)")
    args = parser.parse_args()
    if not os.path.isdir(args.run_dir):
        print(__doc__); sys.exit(1)
    run_comparison(args.run_dir, camera=args.camera)
