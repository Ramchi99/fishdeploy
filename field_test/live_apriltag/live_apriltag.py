"""
Live AprilTag detection from USB camera on Windows.

Captures at full resolution, detects tags using calibrated intrinsics,
and shows a low-resolution preview with overlaid pose annotations.
Optionally streams pose data to the Pi over UDP.

Usage:
    python field_test/live_apriltag/live_apriltag.py
    python field_test/live_apriltag/live_apriltag.py --device 1 --camera overhead --tag_size 0.100 --preview_width 800
    python field_test/live_apriltag/live_apriltag.py --pi_ip raspberrypi.local --port 5005
"""

import os
import sys

# Must be set before cv2 is imported — OpenCV reads it at library load time
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import json
import time
import argparse
import socket
import struct

try:
    from dt_apriltags import Detector
except ModuleNotFoundError:
    from pupil_apriltags import Detector

# Packet format sent to Pi (108 bytes, big-endian):
#   d  timestamp      float64  monotonic seconds
#   i  tag_id         int32
#   d  tx             float64  metres, camera frame
#   d  ty             float64
#   d  tz             float64
#   9d R (row-major)  float64  rotation matrix
#   d  pose_err       float64  reprojection error
PACKET_FMT = "!didddddddddddd"  # network byte order

SETTINGS_FILE = "field_test/vision_settings.json"


def load_camera_params(settings_file, camera_name=None):
    with open(settings_file) as f:
        data = json.load(f)
    name = camera_name or "overhead"
    cam = data.get("cameras", {}).get(name, {})
    fx, fy, cx, cy = cam.get("camera_params", [1310.0, 1310.0, 960.0, 540.0])
    dist = np.array(cam.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"Camera profile: {name}  (fx={fx:.1f}, fy={fy:.1f})")
    return mtx, dist, [fx, fy, cx, cy]


def draw_tag(img, tag, scale, tag_size_m):
    corners = (tag.corners * scale).astype(int)
    center = (tag.center * scale).astype(int)

    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

    # Corner 0 = top-left of tag (red dot so orientation is visible)
    cv2.circle(img, tuple(corners[0]), 6, (0, 0, 255), -1)
    cv2.circle(img, tuple(center), 4, (255, 0, 255), -1)

    if tag.pose_t is not None:
        t = tag.pose_t.flatten()
        dist_m = np.linalg.norm(t)
        label = f"ID {tag.tag_id}  d={dist_m:.2f}m  x={t[0]:.3f} y={t[1]:.3f} z={t[2]:.3f}"
        cv2.putText(img, label, (center[0] + 8, center[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Live AprilTag detector")
    parser.add_argument("--device", type=int, default=0, help="USB camera device index (default 0)")
    parser.add_argument("--camera", default=None, help="Camera calibration profile from vision_settings.json (e.g. simonPhone, overhead)")
    parser.add_argument("--tag_size", type=float, default=0.080, help="Tag size in metres (default 0.080)")
    parser.add_argument("--preview_width", type=int, default=640, help="Preview window width in px (default 640)")
    parser.add_argument("--detect_width", type=int, default=0,
                        help="Resize to this width before detection (0 = full res, faster at lower values)")
    parser.add_argument("--decimate", type=float, default=2.0,
                        help="AprilTag quad_decimate (higher = faster, less accurate, default 2.0)")
    parser.add_argument("--pi_ip", default=None,
                        help="Pi IP address to stream pose over UDP (e.g. raspberrypi.local). Omit to disable.")
    parser.add_argument("--port", type=int, default=5005, help="UDP port (default 5005)")
    parser.add_argument("--capture_height", type=int, default=1080,
                        help="Capture resolution height (default 1080). Try 720 or 480 for higher fps)")
    parser.add_argument("--fps", type=int, default=90,
                        help="Requested camera fps (default 90). Camera must support it in MJPEG mode.")
    parser.add_argument("--no_preview", action="store_true",
                        help="Disable the preview window — faster, console output only")
    args = parser.parse_args()

    if not os.path.exists(SETTINGS_FILE):
        sys.exit(f"Cannot find {SETTINGS_FILE} — run from repo root.")

    mtx, dist, cam_params = load_camera_params(SETTINGS_FILE, camera_name=args.camera)

    # UDP socket — created unconditionally (cheap), only sent to if pi_ip is set
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_target = (args.pi_ip, args.port) if args.pi_ip else None
    if udp_target:
        print(f"UDP streaming → {args.pi_ip}:{args.port}")
    else:
        print("UDP disabled (pass --pi_ip to enable)")

    detector = Detector(
        families="tag36h11",
        nthreads=2,
        quad_decimate=args.decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    capture_w = int(args.capture_height * 16 / 9)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fmt = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    print(f"Format: {actual_fmt}  |  {actual_w}x{actual_h} @ {actual_fps:.0f}fps")
    if actual_fmt.strip('\x00') not in ('MJPG', 'MJPEG'):
        print("  WARNING: MJPEG not active — camera is in uncompressed mode, expect low fps")
    print(f"tag_size={args.tag_size}m  |  decimate={args.decimate}")
    if not args.no_preview:
        print("Press 'q' to quit, 's' to save a snapshot.")

    # If detect_width not specified, default to half the capture width
    detect_w = args.detect_width if args.detect_width > 0 else actual_w // 2
    detect_scale = detect_w / actual_w
    detect_h = int(actual_h * detect_scale)

    # Scale camera matrix for detection resolution
    detect_params = [
        cam_params[0] * detect_scale,
        cam_params[1] * detect_scale,
        cam_params[2] * detect_scale,
        cam_params[3] * detect_scale,
    ]

    # Pre-compute undistort maps once — much faster than cv2.undistort() per frame
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (actual_w, actual_h), cv2.CV_16SC2)

    frame_count = 0
    total_frames = 0
    fps_timer = time.monotonic()
    fps_display = 0.0
    run_start = time.monotonic()

    while True:
        t0 = time.monotonic()
        ret, frame = cap.read()
        t1 = time.monotonic()
        if not ret:
            print("Camera read failed.")
            break

        # Detect on raw (distorted) frame — avoids per-frame undistort on the hot path.
        # Distortion is small enough that tags are reliably detected; pose accuracy is unaffected
        # because we pass the correct intrinsics and the corner error is sub-pixel at typical range.
        detect_frame = cv2.resize(frame, (detect_w, detect_h))
        gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

        tags = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=detect_params,
            tag_size=args.tag_size,
        )
        t2 = time.monotonic()

        # FPS counter
        frame_count += 1
        total_frames += 1
        now = time.monotonic()
        if now - fps_timer >= 1.0:
            fps_display = frame_count / (now - fps_timer)
            capture_ms = (t1 - t0) * 1000
            detect_ms = (t2 - t1) * 1000
            print(f"fps={fps_display:.1f}  capture={capture_ms:.0f}ms  detect={detect_ms:.0f}ms  tags={len(tags)}")
            frame_count = 0
            fps_timer = now

        # Preview (optional — skip entirely with --no_preview)
        if not args.no_preview:
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            preview_h = int(actual_h * args.preview_width / actual_w)
            preview = cv2.resize(undistorted, (args.preview_width, preview_h))
            draw_scale = args.preview_width / detect_w

            for tag in tags:
                draw_tag(preview, tag, draw_scale, args.tag_size)

            cv2.putText(preview, f"{fps_display:.1f} fps  |  {len(tags)} tag(s)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1, cv2.LINE_AA)
            cv2.imshow("AprilTag Live (q=quit)", preview)

        # Console output + UDP send for any detected tags
        if tags:
            ts = time.monotonic()
            for tag in tags:
                t = tag.pose_t.flatten()
                r = tag.pose_R.flatten()
                print(f"t={ts:.3f}  ID={tag.tag_id}  "
                      f"pos=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m  "
                      f"err={tag.pose_err:.2e}")
                if udp_target:
                    packet = struct.pack(
                        PACKET_FMT,
                        ts, tag.tag_id,
                        t[0], t[1], t[2],
                        r[0], r[1], r[2],
                        r[3], r[4], r[5],
                        r[6], r[7], r[8],
                        tag.pose_err,
                    )
                    udp_sock.sendto(packet, udp_target)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not args.no_preview:
            undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
            fname = f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, undistorted)
            print(f"Saved {fname}")

    elapsed = time.monotonic() - run_start
    avg_fps = total_frames / elapsed if elapsed > 0 else 0.0
    print(f"\nAverage fps: {avg_fps:.1f}  ({total_frames} frames in {elapsed:.1f}s)")

    cap.release()
    cv2.destroyAllWindows()
    udp_sock.close()


if __name__ == "__main__":
    main()
