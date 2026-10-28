#!/usr/bin/env python3
"""
Render a recorded SOFI vision session into an ROI-tracking *explainer* clip.

Unlike render_run.py (which always full-frame-detects), this script re-runs the
EXACT deployed ROI tracking state machine from deploy/shared/tag_tracker.py
over the saved video, and visualises it:

  * ROI mode   — everything outside the per-tag ROI windows is dimmed by a
                 semitransparent black mask; the live ROI rectangles glow in
                 each tag's colour. Banner: "ROI TRACKING".
  * Full-frame — when a tag is occluded (miss_count >= MISS_THRESH) the pipeline
                 falls back to a whole-frame scan: the mask drops away, the frame
                 border pulses, banner: "FULL-FRAME RE-ACQUIRE".

Always overlaid: the fish->target arrow and the body-frame target vector
(fwd=X, lat=Y) — i.e. the payload the laptop ships to the Pi over UDP.

Two layouts are produced from a single detection pass:
  --layout camera      -> replay_roi_camera.mp4   (cinematic, camera only)
  --layout sidebyside  -> replay_roi_sidebyside.mp4 (camera + body-frame panel)
  --layout both        -> both files (default)

Output is VFR (real-time playback from frame_timestamps.csv via ffmpeg concat),
identical timing backbone to render_run.py.

Usage:
    python deploy/laptop/render_roi_explainer.py \
        --session data/deploy_runs/20260520_192737_figure_eight_5/raw \
        --env-config deploy/configs/cpg_24d.json \
        --start 3.5 --end 13.0
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

from deploy.shared.tag_tracker import (
    TagTracker, _roi_detect, _fullframe_detect,
    MISS_THRESH, ANCHOR_INTERVAL, TARGET_INTERVAL,
)

SETTINGS_FILE = os.path.join(_REPO_ROOT, "deploy", "configs", "camera_settings.json")

# --- pinned 20-May-evening vision tune (matches this run's tuning plot) ---------
TUNE_ALPHA    = 0.9
TUNE_BETA     = -20.0
TUNE_BLUR     = 3
TUNE_DECIMATE = 2.0

# BGR colours / labels — match render_run.py
TAG_COLORS = {0: (90, 255, 90), 1: (0, 165, 255), 2: (255, 120, 0)}   # fish=green anchor=orange target=blue
TAG_LABELS = {0: "fish", 1: "anchor", 2: "target"}

MASK_ALPHA = 0.62          # darkness of the dim outside ROIs
PANEL_W    = 540
PANEL_H    = 540


def load_camera_params(settings_file, camera_name=None):
    with open(settings_file) as f:
        data = json.load(f)
    name = camera_name or "overhead"
    cam = data.get("cameras", {}).get(name, {})
    fx, fy, cx, cy = cam.get("camera_params", [1310.0, 1310.0, 960.0, 540.0])
    return [fx, fy, cx, cy]


def world_xy(t_tag, t_anchor, R_anchor):
    p = R_anchor.T @ (t_tag - t_anchor)
    return float(p[0]), float(p[1])


def extract_yaw(R_tag, R_anchor):
    fwd = (R_anchor.T @ R_tag)[:, 1]
    return float(math.atan2(fwd[1], fwd[0]))


# --------------------------------------------------------------------------- #
#  body-frame side panel
# --------------------------------------------------------------------------- #
def make_panel():
    dpi = 100
    fig, ax = plt.subplots(figsize=(PANEL_W / dpi, PANEL_H / dpi), dpi=dpi)
    fig.patch.set_facecolor("#0b0b0b")
    ax.set_facecolor("#0b0b0b")
    ax.set_title("Fish body frame -> target  (UDP payload)", color="white", fontsize=11)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color("#444")
    ax.tick_params(colors="#888")
    ax.axhline(0, color="#555", lw=0.6)
    ax.axvline(0, color="#555", lw=0.6)
    ax.set_xlabel("forward  X (m)", color="#bbb")
    ax.set_ylabel("lateral  Y (m)", color="#bbb")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.plot(0, 0, "s", color="#5dff5d", ms=12, zorder=5)
    ax.text(0.06, -0.16, "fish", color="#5dff5d", fontsize=9)
    q = ax.quiver(0, 0, 0, 0, angles="xy", scale_units="xy", scale=1,
                  color="cyan", width=0.018, zorder=4)
    dot = ax.plot([], [], "o", color="#3cb0ff", ms=10, zorder=5)[0]
    mode_txt = ax.text(-1.32, 1.18, "", color="white", fontsize=12, fontweight="bold")
    vec_txt  = ax.text(-1.32, -1.28, "", color="cyan", fontsize=11, family="monospace")
    plt.tight_layout()
    return fig, q, dot, mode_txt, vec_txt


def fig_to_bgr(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


# --------------------------------------------------------------------------- #
#  camera annotation
# --------------------------------------------------------------------------- #
def annotate_camera(frame, trackers, tag_map, roi_mode, fallback_tags,
                    fish_id, anchor_id, target_id, pulse,
                    fish_x, fish_y, fish_yaw, bx, by, dist):
    """Draw mask / ROI windows / arrow / HUD on a full-res BGR frame (in place-ish)."""
    H, W = frame.shape[:2]
    out = frame.copy()

    if roi_mode:
        # dim everything, then punch the ROI windows back to full brightness
        dim = (out.astype(np.float32) * (1.0 - MASK_ALPHA)).astype(np.uint8)
        for tid, trk in trackers.items():
            if not trk.initialised:
                continue
            x0, y0, x1, y1 = trk.get_roi(W, H)
            dim[y0:y1, x0:x1] = out[y0:y1, x0:x1]
        out = dim
        for tid, trk in trackers.items():
            if not trk.initialised:
                continue
            x0, y0, x1, y1 = trk.get_roi(W, H)
            col = TAG_COLORS.get(tid, (200, 200, 200))
            seen = tag_map.get(tid) is not None
            cv2.rectangle(out, (x0, y0), (x1, y1), col, 3 if seen else 2)
            cv2.putText(out, f"ROI:{TAG_LABELS.get(tid, tid)}", (x0 + 4, y0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        banner, bcol = "ROI TRACKING", (90, 255, 90)
    else:
        # full-frame scan: bright everywhere, pulsing border
        thick = int(6 + 6 * pulse)
        bcol_b = (0, int(120 + 135 * pulse), int(255))
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), bcol_b, thick)
        lost = ", ".join(TAG_LABELS.get(t, str(t)) for t in fallback_tags) or "init"
        banner, bcol = f"FULL-FRAME RE-ACQUIRE  ({lost} lost)", (60, 160, 255)

    # tag corner outlines for whatever was actually detected this frame
    for tid, det in tag_map.items():
        if det is None:
            continue
        col = TAG_COLORS.get(tid, (200, 200, 200))
        corners = det.corners.astype(int).reshape((-1, 1, 2))
        cv2.polylines(out, [corners], True, col, 2)

    # fish -> target arrow in pixel space (last-known centres)
    f_trk, t_trk = trackers[fish_id], trackers[target_id]
    if f_trk.initialised and t_trk.initialised:
        p0 = (int(f_trk.cx_full), int(f_trk.cy_full))
        p1 = (int(t_trk.cx_full), int(t_trk.cy_full))
        cv2.arrowedLine(out, p0, p1, (255, 255, 0), 3, tipLength=0.05)

    # HUD banner
    cv2.rectangle(out, (0, 0), (W, 64), (0, 0, 0), -1)
    cv2.putText(out, banner, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bcol, 3)
    cv2.putText(out,
                f"fish=({fish_x:.2f},{fish_y:.2f}) yaw={math.degrees(fish_yaw):.0f}  "
                f"body:fwd={bx:+.2f} lat={by:+.2f}  d={dist:.2f}m",
                (24, H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Render ROI-tracking explainer (VFR)")
    ap.add_argument("--session", required=True, help="raw session dir (video.avi + frame_timestamps.csv)")
    ap.add_argument("--env-config", required=True)
    ap.add_argument("--tag_size", type=float, default=0.080)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--start", type=float, default=3.5, help="clip start (s, rel. to first timestamp)")
    ap.add_argument("--end", type=float, default=13.0, help="clip end (s)")
    ap.add_argument("--layout", default="both", choices=["camera", "sidebyside", "both"])
    ap.add_argument("--alpha", type=float, default=TUNE_ALPHA)
    ap.add_argument("--beta", type=float, default=TUNE_BETA)
    ap.add_argument("--blur", type=int, default=TUNE_BLUR)
    ap.add_argument("--decimate", type=float, default=TUNE_DECIMATE)
    ap.add_argument("--flip_video", action="store_true")
    ap.add_argument("--target_every_frame", action="store_true",
                    help="'cheat': re-detect the target tag every frame (like the fish) "
                         "instead of every TARGET_INTERVAL frames, for smooth ROI tracking")
    ap.add_argument("--miss_thresh", type=int, default=None,
                    help="VISUALIZATION override of MISS_THRESH (deployed value is 3). "
                         "Set 1 or 2 for a simpler 'not seen -> full-frame' story. "
                         "Does NOT change the deployed constant in tag_tracker.py.")
    ap.add_argument("--no_flip_y", action="store_true", help="disable Y flip on body-frame panel")
    ap.add_argument("--out_prefix", default="replay_roi")
    args = ap.parse_args()

    session = os.path.abspath(args.session)
    flip_y  = not args.no_flip_y

    if args.miss_thresh is not None:
        # VISUALIZATION-only override: TagTracker.use_fullframe reads the module
        # global at call time, so patching it here changes the fallback sensitivity
        # for this render without touching the deployed constant.
        import deploy.shared.tag_tracker as _tt
        _tt.MISS_THRESH = args.miss_thresh
        print(f"[viz] MISS_THRESH overridden to {args.miss_thresh} (deployed value is 3)")

    vid_path = os.path.join(session, "video.avi")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(session, "video.mp4")
    if not os.path.exists(vid_path):
        sys.exit(f"ERROR: no video in {session}")

    with open(args.env_config) as f:
        config = json.load(f)
    tag_ids   = config.get("tag_ids", {"fish": 0, "anchor": 1, "target": 2})
    fish_id, anchor_id, target_id = tag_ids["fish"], tag_ids["anchor"], tag_ids["target"]

    cam_params = load_camera_params(SETTINGS_FILE, args.camera)
    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=args.decimate,
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    # timestamps -> per-frame absolute time (rel. to first)
    ts_path = os.path.join(session, "frame_timestamps.csv")
    frame_times = None
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            rows = list(csv.DictReader(f))
        frame_times = {int(r["frame_index"]): float(r["laptop_time"]) for r in rows}
        t0 = min(frame_times.values())
        frame_times = {k: v - t0 for k, v in frame_times.items()}

    cap     = cv2.VideoCapture(vid_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ts_list = [frame_times.get(i, i / src_fps) if frame_times else i / src_fps for i in range(total)]
    print(f"Video {W}x{H}, {total} frames, span {ts_list[-1]:.1f}s -> clip [{args.start},{args.end}]s")
    print(f"Tune: alpha={args.alpha} beta={args.beta} blur={args.blur} decimate={args.decimate}")

    trackers = {fish_id: TagTracker(), anchor_id: TagTracker(), target_id: TagTracker()}

    want_cam  = args.layout in ("camera", "both")
    want_side = args.layout in ("sidebyside", "both")
    tmp_cam   = tempfile.mkdtemp(prefix="roi_cam_")  if want_cam  else None
    tmp_side  = tempfile.mkdtemp(prefix="roi_side_") if want_side else None
    cc_cam    = ["ffconcat version 1.0"]
    cc_side   = ["ffconcat version 1.0"]

    if want_side:
        fig, q, dot, mode_txt, vec_txt = make_panel()

    total_frames = 0   # mirrors vision_streamer counter (drives ROI re-detect cadence)
    out_idx = 0
    R_anchor = t_anchor = None
    fish_x = fish_y = fish_yaw = 0.0
    target_x = target_y = 0.0
    t_wall = time.time()

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        t = ts_list[fi]
        if t < args.start:
            continue
        if t > args.end:
            break
        if args.flip_video:
            frame = cv2.flip(frame, -1)

        total_frames += 1
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- EXACT deployed orchestration (see vision_streamer.main loop) ----
        f_trk, a_trk, t_trk = trackers[fish_id], trackers[anchor_id], trackers[target_id]
        fallback = f_trk.use_fullframe or t_trk.use_fullframe or a_trk.use_fullframe
        fallback_tags = [tid for tid, trk in trackers.items() if trk.use_fullframe and trk.initialised]

        if fallback:
            gray = gray_raw
            if args.alpha != 1.0 or args.beta != 0.0:
                gray = cv2.convertScaleAbs(gray, alpha=args.alpha, beta=args.beta)
            if args.blur >= 3:
                gray = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)
            tag_map = _fullframe_detect(detector, gray, cam_params, args.tag_size, trackers)
            roi_mode = False
        else:
            anchor_redetect = (total_frames % ANCHOR_INTERVAL == 0)
            target_redetect = args.target_every_frame or (total_frames % TARGET_INTERVAL == 0)
            tag_map = {
                fish_id: _roi_detect(detector, gray_raw, fish_id, f_trk, cam_params,
                                     args.tag_size, W, H, args.alpha, args.beta, args.blur),
            }
            if target_redetect:
                tag_map[target_id] = _roi_detect(detector, gray_raw, target_id, t_trk, cam_params,
                                                 args.tag_size, W, H, args.alpha, args.beta, args.blur)
            if anchor_redetect and a_trk.initialised:
                tag_map[anchor_id] = _roi_detect(detector, gray_raw, anchor_id, a_trk, cam_params,
                                                 args.tag_size, W, H, args.alpha, args.beta, args.blur)
            roi_mode = True

        # ---- pose / world transform from tracker poses ----
        if a_trk.pose_R is not None:
            R_anchor, t_anchor = a_trk.pose_R, a_trk.pose_t
        if f_trk.pose_t is not None and R_anchor is not None:
            fish_x, fish_y = world_xy(f_trk.pose_t, t_anchor, R_anchor)
            fish_yaw = extract_yaw(f_trk.pose_R, R_anchor)
        if t_trk.pose_t is not None and R_anchor is not None:
            target_x, target_y = world_xy(t_trk.pose_t, t_anchor, R_anchor)

        rx, ry = target_x - fish_x, target_y - fish_y
        dist = math.hypot(rx, ry)
        c, s = math.cos(fish_yaw), math.sin(fish_yaw)
        bx = c * rx + s * ry      # forward
        by = -s * rx + c * ry     # lateral

        pulse = 0.5 - 0.5 * math.cos(total_frames * 0.6)
        cam_full = annotate_camera(frame, trackers, tag_map, roi_mode, fallback_tags,
                                   fish_id, anchor_id, target_id, pulse,
                                   fish_x, fish_y, fish_yaw, bx, by, dist)

        # frame display duration (VFR)
        if fi + 1 < total:
            duration = max(ts_list[fi + 1] - ts_list[fi], 1.0 / 120.0)
        else:
            duration = 1.0 / src_fps

        if want_cam:
            cam_out = cv2.resize(cam_full, (1280, int(1280 * H / W)))
            p = os.path.join(tmp_cam, f"{out_idx:06d}.jpg")
            cv2.imwrite(p, cam_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cc_cam += [f"file '{p}'", f"duration {duration:.6f}"]

        if want_side:
            plot_by = -by if flip_y else by
            q.set_UVC(bx, plot_by)
            dot.set_data([bx], [plot_by])
            if roi_mode:
                mode_txt.set_text("MODE: ROI TRACKING")
                mode_txt.set_color("#5dff5d")
            else:
                mode_txt.set_text("MODE: FULL-FRAME SCAN")
                mode_txt.set_color("#ff8c3c")
            vec_txt.set_text(f"fwd={bx:+.2f}  lat={by:+.2f}  d={dist:.2f}m  yaw={math.degrees(fish_yaw):+.0f}")
            panel = cv2.resize(fig_to_bgr(fig), (PANEL_W, PANEL_H))
            cam_h = PANEL_H
            cam_w = int(cam_h * W / H)
            cam_s = cv2.resize(cam_full, (cam_w, cam_h))
            canvas = np.zeros((cam_h, cam_w + PANEL_W, 3), np.uint8)
            canvas[:, :cam_w] = cam_s
            canvas[:, cam_w:] = panel
            p = os.path.join(tmp_side, f"{out_idx:06d}.jpg")
            cv2.imwrite(p, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cc_side += [f"file '{p}'", f"duration {duration:.6f}"]

        out_idx += 1
        if out_idx % 30 == 0:
            fps_r = out_idx / (time.time() - t_wall)
            print(f"  {out_idx} frames  t={t:.2f}s  {'ROI ' if roi_mode else 'FULL'}  {fps_r:.1f} fps")

    cap.release()
    if want_side:
        plt.close(fig)
    print(f"Rendered {out_idx} frames.")

    def assemble(cc_lines, tmpdir, out_path):
        cpath = os.path.join(tmpdir, "concat.txt")
        with open(cpath, "w") as f:
            f.write("\n".join(cc_lines) + "\n")
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", cpath,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path],
                           capture_output=True, text=True)
        shutil.rmtree(tmpdir)
        if r.returncode != 0:
            print("ffmpeg error:\n", r.stderr[-1500:])
        else:
            print(f"  saved -> {out_path}")

    print("Assembling VFR video(s)...")
    if want_cam:
        assemble(cc_cam, tmp_cam, os.path.join(session, f"{args.out_prefix}_camera.mp4"))
    if want_side:
        assemble(cc_side, tmp_side, os.path.join(session, f"{args.out_prefix}_sidebyside.mp4"))
    print("Done.")


if __name__ == "__main__":
    main()
