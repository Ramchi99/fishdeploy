#!/usr/bin/env python3
"""
Render a recorded SOFI vision session into a replay video.

Re-runs AprilTag detection on the saved video and produces a side-by-side
video: camera feed (left) + 4-panel diagnostics evolving over time (right).

Output: <session>/replay.mp4  (VFR — exact real-time playback from timestamps)

Usage:
    python deploy/laptop/render_run.py \
        --session deploy/recordings/20260506_123456 \
        --env-config deploy/configs/cpg_24d.json \
        --tag_size 0.080
"""
import argparse
import collections
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from dt_apriltags import Detector
except ModuleNotFoundError:
    from pupil_apriltags import Detector

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

SETTINGS_FILE = os.path.join(_REPO_ROOT, "deploy", "configs", "camera_settings.json")

PLOT_W   = 960
PLOT_H   = 540
CAM_H    = 540
WINDOW_S = 7.0
ALPHA    = 0.25  # matches ObsAssemblerCPG14D._EWM_ALPHA


def load_camera_params(settings_file, camera_name=None):
    with open(settings_file) as f:
        data = json.load(f)
    name = camera_name or "overhead"
    cam = data.get("cameras", {}).get(name, {})
    fx, fy, cx, cy = cam.get("camera_params", [1310.0, 1310.0, 960.0, 540.0])
    dist = np.array(cam.get("dist_coeffs", [0.0] * 5), dtype=np.float64)
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return mtx, dist, [fx, fy, cx, cy]


def world_xy(t_tag, t_anchor, R_anchor):
    p = R_anchor.T @ (t_tag - t_anchor)
    return float(p[0]), float(p[1])


def extract_yaw(R_tag, R_anchor):
    fwd = (R_anchor.T @ R_tag)[:, 1]
    return float(math.atan2(fwd[1], fwd[0]))


TAG_COLORS = {0: (0, 255, 0), 1: (255, 140, 0), 2: (0, 60, 255)}
TAG_LABELS = {0: "fish", 1: "anchor", 2: "target"}


def draw_overlays(img, tag_map, fish_id, anchor_id, target_id,
                  fish_x, fish_y, fish_yaw, sx, sy, pixel_trail=None):
    centers = {}
    for tid, tag in tag_map.items():
        color = TAG_COLORS.get(tid, (200, 200, 200))
        corners = (tag.corners * [sx, sy]).astype(int).reshape((-1, 1, 2))
        cv2.polylines(img, [corners], isClosed=True, color=color, thickness=2)
        cx = int(tag.corners[:, 0].mean() * sx)
        cy = int(tag.corners[:, 1].mean() * sy)
        centers[tid] = (cx, cy)
        cv2.putText(img, TAG_LABELS.get(tid, str(tid)),
                    (cx - 18, cy - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw trajectory trail
    if pixel_trail and len(pixel_trail) >= 2:
        pts = list(pixel_trail)
        n = len(pts)
        for i in range(1, n):
            alpha = i / n  # fade older points
            color = (int(50 + 150 * alpha), int(200 * alpha), int(50 + 150 * alpha))
            cv2.line(img, pts[i - 1], pts[i], color, 2)

    if fish_id in centers and target_id in centers:
        cv2.arrowedLine(img, centers[fish_id], centers[target_id],
                        (0, 255, 255), 2, tipLength=0.2)

    cv2.putText(img, f"fish=({fish_x:.2f},{fish_y:.2f}) yaw={math.degrees(fish_yaw):.0f}deg",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(img, f"tags={list(tag_map.keys())}",
                (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


def make_figure(frame_mode: str = "anchor"):
    dpi = 100
    fig, axes = plt.subplots(2, 2, figsize=(PLOT_W / dpi, PLOT_H / dpi), dpi=dpi)
    fig.suptitle("SOFI — Vision Diagnostics", fontsize=11)
    ax_vec, ax_vel, ax_dist, ax_spd = axes.flat

    vec_label = "fish body frame  (fwd=X, lat=Y)" if frame_mode == "fish" else "anchor / world frame"
    ax_vec.set_title(f"Fish → Target  ({vec_label})")
    ax_vec.set_aspect("equal")
    ax_vec.axhline(0, color="#888", lw=0.5)
    ax_vec.axvline(0, color="#888", lw=0.5)
    ax_vec.set_xlabel("X (m)")
    ax_vec.set_ylabel("Y (m)")
    ax_vec.set_xlim(-1, 1)
    ax_vec.set_ylim(-1, 1)
    ax_vec.plot(0, 0, "gs", ms=10, label="fish", zorder=5)
    q_vec = ax_vec.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1,
                          color="cyan", width=0.015, zorder=4)
    dot_target = ax_vec.plot([], [], "ro", ms=8, label="target", zorder=5)[0]
    ax_vec.legend(fontsize=7, loc="upper right")

    vel_label = "fish body frame" if frame_mode == "fish" else "anchor / world frame"
    ax_vel.set_title(f"Relative velocity  ({vel_label})")
    ax_vel.set_aspect("equal")
    ax_vel.axhline(0, color="#888", lw=0.5)
    ax_vel.axvline(0, color="#888", lw=0.5)
    ax_vel.set_xlabel("Vx (m/s)")
    ax_vel.set_ylabel("Vy (m/s)")
    ax_vel.set_xlim(-1, 1)
    ax_vel.set_ylim(-1, 1)
    ax_vel.plot(0, 0, "k+", ms=10, zorder=5)
    q_vel = ax_vel.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1,
                          color="orange", width=0.015, zorder=4)

    ax_dist.set_title("Distance to target")
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("m")
    (line_dist,) = ax_dist.plot([], [], "g-", lw=1.5)
    ax_dist.set_xlim(0, WINDOW_S)
    ax_dist.set_ylim(0, 2)

    ax_spd.set_title("Approach speed  (smoothed)")
    ax_spd.set_xlabel("Time (s)")
    ax_spd.set_ylabel("m/s")
    (line_spd,) = ax_spd.plot([], [], color="#cc44ff", lw=1.5)
    ax_spd.set_xlim(0, WINDOW_S)
    ax_spd.set_ylim(0, 0.5)

    plt.tight_layout()
    return fig, q_vec, dot_target, q_vel, line_dist, line_spd, ax_dist, ax_spd


def fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def main():
    parser = argparse.ArgumentParser(description="Render SOFI recording into VFR replay video")
    parser.add_argument("--session", required=True, help="Recording session directory")
    parser.add_argument("--env-config", required=True, help="env_config.json path")
    parser.add_argument("--tag_size", type=float, default=0.080)
    parser.add_argument("--camera", default=None, help="Camera profile (default: overhead)")
    parser.add_argument("--frame", default="anchor", choices=["anchor", "fish"],
                        help="Reference frame for vector/velocity plots: "
                             "'anchor' = world frame (default), "
                             "'fish' = body frame matching obs_assembler (fwd=X, lat=Y)")
    parser.add_argument("--no_flip_y", action="store_true", help="Disable Y-axis flip on plots")
    parser.add_argument("--flip_video", action="store_true",
                        help="Flip camera feed 180° (fixes upside-down MSMF recordings)")
    args = parser.parse_args()

    session    = os.path.abspath(args.session)
    ts_path    = os.path.join(session, "frame_timestamps.csv")
    out_path   = os.path.join(session, "replay.mp4")
    flip_y     = not args.no_flip_y
    flip_video = args.flip_video
    frame_mode = args.frame

    vid_path = os.path.join(session, "video.avi")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(session, "video.mp4")
    if not os.path.exists(vid_path):
        print(f"ERROR: no video found in {session}  (expected video.avi or video.mp4)")
        sys.exit(1)

    with open(args.env_config) as f:
        config = json.load(f)
    tag_ids   = config.get("tag_ids", {"fish": 0, "anchor": 1, "target": 2})
    fish_id   = tag_ids["fish"]
    anchor_id = tag_ids["anchor"]
    target_id = tag_ids["target"]

    _, _, cam_params = load_camera_params(SETTINGS_FILE, args.camera)

    detector = Detector(
        families="tag36h11", nthreads=2, quad_decimate=2.0,
        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25,
    )

    # load per-frame timestamps
    frame_times = None
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            rows = list(csv.DictReader(f))
        frame_times = {int(r["frame_index"]): float(r["laptop_time"]) for r in rows}
        t0_abs = min(frame_times.values())
        frame_times = {k: v - t0_abs for k, v in frame_times.items()}
        print(f"Loaded {len(frame_times)} timestamps from frame_timestamps.csv")

    cap    = cv2.VideoCapture(vid_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_times and len(frame_times) >= 2:
        ts_list = [frame_times.get(i, i / src_fps) for i in range(total)]
        avg_fps = (total - 1) / max(ts_list[-1] - ts_list[0], 1e-6)
        print(f"Video: {src_w}x{src_h}  avg={avg_fps:.1f}fps  ({total} frames)  → VFR output")
    else:
        ts_list = None
        avg_fps = src_fps
        print(f"Video: {src_w}x{src_h} @ {src_fps:.1f}fps  ({total} frames)  → VFR output")

    cam_w    = int(CAM_H * src_w / src_h)
    d_w      = src_w // 2
    d_h      = src_h // 2
    d_params = [p * (d_w / src_w) for p in cam_params]
    out_w    = cam_w + PLOT_W
    out_h    = max(CAM_H, PLOT_H)

    # temp dir for JPEG frames — ffmpeg concat demuxer assembles them into VFR mp4
    tmpdir = tempfile.mkdtemp(prefix="sofi_render_")
    concat_lines = ["ffconcat version 1.0"]

    fig, q_vec, dot_target, q_vel, line_dist, line_spd, ax_dist, ax_spd = make_figure(frame_mode)

    times  = collections.deque()
    dist_b = collections.deque()
    spd_b  = collections.deque()

    last_rel = None
    last_t   = None
    sv = [0.0, 0.0]
    R_anchor = None
    t_anchor = None
    R_fish   = None
    fish_x = fish_y = fish_yaw = 0.0
    pixel_trail = collections.deque(maxlen=200)
    target_x = target_y = 1.0
    t_session_start = None

    t_wall = time.time()

    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        t = ts_list[frame_idx] if ts_list else frame_idx / src_fps
        if t_session_start is None:
            t_session_start = t

        # compute this frame's display duration from timestamps
        if ts_list and frame_idx + 1 < total:
            duration = ts_list[frame_idx + 1] - ts_list[frame_idx]
            duration = max(duration, 1.0 / 120.0)  # floor at 120 fps
        else:
            duration = 1.0 / avg_fps

        # detect tags
        small = cv2.resize(frame, (d_w, d_h))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tags  = detector.detect(gray, estimate_tag_pose=True,
                                camera_params=d_params, tag_size=args.tag_size)
        tag_map = {tag.tag_id: tag for tag in tags}

        if anchor_id in tag_map:
            R_anchor = tag_map[anchor_id].pose_R.copy()
            t_anchor = tag_map[anchor_id].pose_t.flatten().copy()

        if fish_id in tag_map and R_anchor is not None:
            fish_x, fish_y = world_xy(tag_map[fish_id].pose_t.flatten(), t_anchor, R_anchor)
            R_fish   = tag_map[fish_id].pose_R
            fish_yaw = extract_yaw(R_fish, R_anchor)

        if target_id in tag_map and R_anchor is not None:
            target_x, target_y = world_xy(tag_map[target_id].pose_t.flatten(), t_anchor, R_anchor)

        rx = target_x - fish_x
        ry = target_y - fish_y
        dist = math.sqrt(rx ** 2 + ry ** 2)

        # Optionally rotate into fish body frame (same transform as obs_assembler).
        # fwd=X (along fish swim direction), lat=Y (lateral).
        if frame_mode == "fish":
            c, s = math.cos(fish_yaw), math.sin(fish_yaw)
            bx =  c * rx + s * ry
            by = -s * rx + c * ry
        else:
            bx, by = rx, ry

        vx = vy = 0.0
        if last_rel is not None and last_t is not None:
            dt = t - last_t
            if 0 < dt < 0.5:
                sv[0] = ALPHA * (bx - last_rel[0]) / dt + (1 - ALPHA) * sv[0]
                sv[1] = ALPHA * (by - last_rel[1]) / dt + (1 - ALPHA) * sv[1]
                vx, vy = sv[0], sv[1]
        last_rel = (bx, by)
        last_t   = t

        times.append(t)
        dist_b.append(dist)
        spd_b.append(math.sqrt(vx ** 2 + vy ** 2))

        plot_by = -by if flip_y else by
        plot_vy = -vy if flip_y else vy

        q_vec.set_UVC(bx, plot_by)
        dot_target.set_data([bx], [plot_by])
        q_vel.set_UVC(vx, plot_vy)

        # trim to the last WINDOW_S seconds by time, not frame count
        t_cur = t - t_session_start
        cutoff_abs = t - WINDOW_S
        while len(times) > 1 and times[0] < cutoff_abs:
            times.popleft()
            dist_b.popleft()
            spd_b.popleft()

        t_rel = [tt - t_session_start for tt in times]
        line_dist.set_data(t_rel, list(dist_b))
        ax_dist.set_xlim(max(0, t_cur - WINDOW_S), max(t_cur, 0.01))
        ax_dist.set_ylim(0, max(max(dist_b) * 1.2, 0.1))

        # speed is already EMA-smoothed (via sv[]); no convolution needed
        spd_arr = np.array(list(spd_b))
        line_spd.set_data(t_rel, spd_arr)
        ax_spd.set_xlim(max(0, t_cur - WINDOW_S), max(t_cur, 0.01))
        ax_spd.set_ylim(0, max(float(spd_arr.max()) * 1.2, 0.05))

        if fish_id in tag_map:
            _sx, _sy = cam_w / d_w, CAM_H / d_h
            _fcx = int(tag_map[fish_id].corners[:, 0].mean() * _sx)
            _fcy = int(tag_map[fish_id].corners[:, 1].mean() * _sy)
            pixel_trail.append((_fcx, _fcy))

        plot_img  = cv2.resize(fig_to_bgr(fig), (PLOT_W, PLOT_H))
        raw_frame = cv2.flip(frame, -1) if flip_video else frame
        cam_frame = cv2.resize(raw_frame, (cam_w, CAM_H))
        draw_overlays(cam_frame, tag_map, fish_id, anchor_id, target_id,
                      fish_x, fish_y, fish_yaw,
                      sx=cam_w / d_w, sy=CAM_H / d_h,
                      pixel_trail=pixel_trail)

        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:CAM_H, :cam_w]               = cam_frame
        canvas[:PLOT_H, cam_w:cam_w + PLOT_W] = plot_img

        # save frame as JPEG and record its duration for ffmpeg
        img_path = os.path.join(tmpdir, f"{frame_idx:06d}.jpg")
        cv2.imwrite(img_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        concat_lines.append(f"file '{img_path}'")
        concat_lines.append(f"duration {duration:.6f}")

        if frame_idx % 30 == 0 and frame_idx > 0:
            elapsed = time.time() - t_wall
            fps_r   = frame_idx / elapsed
            eta     = (total - frame_idx) / max(fps_r, 0.01)
            print(f"  {frame_idx}/{total}  ({100*frame_idx/total:.0f}%)  "
                  f"{fps_r:.1f} fps render  ETA {eta:.0f}s")

    cap.release()
    plt.close(fig)

    # assemble VFR mp4 via ffmpeg concat demuxer
    concat_path = os.path.join(tmpdir, "concat.txt")
    with open(concat_path, "w") as f:
        f.write("\n".join(concat_lines) + "\n")

    print("\nAssembling VFR video with ffmpeg...")
    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path,
    ], capture_output=True, text=True)

    shutil.rmtree(tmpdir)

    if result.returncode != 0:
        print("ffmpeg error:\n", result.stderr[-2000:])
        sys.exit(1)

    print(f"Done — saved to {out_path}")


if __name__ == "__main__":
    main()
