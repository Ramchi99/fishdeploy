#!/usr/bin/env python3
"""
camera_stress_test.py — Measure maximum camera frame delivery rate.

Reads frames as fast as possible with no processing and prints per-second
and overall fps.  Use this to confirm the camera hardware ceiling before
adding AprilTag detection overhead.

Usage (Windows, from repo root):
    python deploy\\laptop\\camera_stress_test.py --device 1
    python deploy\\laptop\\camera_stress_test.py --device 1 --fps 90 --width 1920 --height 1080
    python deploy\\laptop\\camera_stress_test.py --device 1 --fps 60 --width 1280 --height 720

Usage (Linux / WSL2):
    python3 deploy/laptop/camera_stress_test.py --device 1
"""

import argparse
import sys
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Camera frame-rate stress test")
    parser.add_argument("--device",   type=int, default=0,    help="Camera device index")
    parser.add_argument("--fps",      type=int, default=90,   help="Requested fps (default 90; capped to 30 on Linux)")
    parser.add_argument("--width",    type=int, default=None, help="Frame width  (default: 1920 Windows / 1280 Linux)")
    parser.add_argument("--height",   type=int, default=None, help="Frame height (default: 1080 Windows / 720 Linux)")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration [s]")
    args = parser.parse_args()

    is_linux = sys.platform == "linux"

    width  = args.width  or (1280 if is_linux else 1920)
    height = args.height or (720  if is_linux else 1080)
    fps_req = min(args.fps, 30) if is_linux else args.fps

    if is_linux:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps_req)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        sys.exit(f"Cannot open camera device {args.device}")

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    backend    = "V4L2" if is_linux else "MSMF+MJPEG"

    print(f"Camera {args.device}  {actual_w}×{actual_h}  @ {actual_fps:.0f} fps reported  [{backend}]")
    print(f"Reading for {args.duration:.0f}s — no processing, no display\n")

    total_frames = 0
    failed_reads = 0
    interval_frames = 0
    interval_start = time.perf_counter()
    t_start = interval_start
    t_end   = t_start + args.duration

    fps_samples = []

    while True:
        ret, _ = cap.read()
        now = time.perf_counter()

        if ret:
            total_frames   += 1
            interval_frames += 1
        else:
            failed_reads += 1

        elapsed_interval = now - interval_start
        if elapsed_interval >= 1.0:
            fps_now = interval_frames / elapsed_interval
            fps_samples.append(fps_now)
            print(f"  {now - t_start:5.1f}s   {fps_now:5.1f} fps   ({interval_frames} frames)")
            interval_frames = 0
            interval_start  = now

        if now >= t_end:
            break

    cap.release()

    elapsed_total = time.perf_counter() - t_start
    fps_overall   = total_frames / elapsed_total
    fps_mean      = float(np.mean(fps_samples)) if fps_samples else fps_overall
    fps_min       = float(np.min(fps_samples))  if fps_samples else fps_overall
    fps_max       = float(np.max(fps_samples))  if fps_samples else fps_overall

    print(f"\n{'─'*50}")
    print(f"  Total frames  : {total_frames}  ({failed_reads} failed reads)")
    print(f"  Duration      : {elapsed_total:.2f}s")
    print(f"  Overall fps   : {fps_overall:.1f}")
    print(f"  Per-second    : mean={fps_mean:.1f}  min={fps_min:.1f}  max={fps_max:.1f}")
    print(f"  Requested fps : {fps_req}  (reported by driver: {actual_fps:.0f})")
    print(f"{'─'*50}")

    headroom = fps_overall - 51.6  # typical full-frame detection cost
    if headroom > 5:
        print(f"  Camera headroom above full-frame detection cost (~51.6 fps): +{headroom:.0f} fps")
    else:
        print(f"  Camera is the bottleneck — detection already faster than frame delivery.")


if __name__ == "__main__":
    main()
