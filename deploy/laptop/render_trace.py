#!/usr/bin/env python3
"""
Render a recorded SOFI vision session into a clean, professional trace clip.

A stripped-down sibling of render_run.py / render_roi_explainer.py. There is NO
ROI machinery, no dimming mask, no diagnostic panels, no HUD text — just the
clean camera frame with two overlays:

  * Trajectory trace  — the full path the fish has travelled, colour-coded by
                        time via the viridis colormap (dark purple = run start,
                        bright yellow = now). Anti-aliased, drawn over a thin
                        dark underline so it stays legible over bright water.
  * Fish -> target arrow — a clean white arrow (subtle dark outline) from the
                        fish tag to the target tag, drawn every frame using the
                        target's last-known position to avoid flicker.

Output is VFR (variable-framerate) — each frame is held for its real
inter-frame duration, assembled via the ffmpeg concat demuxer. Playback is
genuine real time. Output: <session>/replay_trace.mp4

DROPPED-FRAME CORRECTION
------------------------
Older recordings (made before the vision_streamer recorder fix) logged a
timestamp for every CAPTURED frame, but the async MJPEG writer silently dropped
frames whenever its encode queue backed up — so frame_timestamps.csv has MORE
rows than the AVI has frames, and frame_index is the capture index, not the AVI
position. Naively pairing AVI frame i with timestamp i compresses time and makes
the tail play several times too fast.

Empirically the drops are end-loaded: the run records 1:1 in real time until the
encoder falls behind, after which a sparse subset survives. This script detects
that onset from the fish's pixel-speed (which spikes when frames start being
skipped) and rebuilds the timeline:

    phase 1 (frames 0..K-1)      -> true timestamps t[i]              (1:1)
    phase 2 (frames K..N_avi-1)  -> spread evenly over [t[K], t_end]  (true span)

If the onset can't be found cleanly it falls back to a uniform resample across
the full session span. Use --drop_onset K to set the onset manually, or
--no_drop_fix to disable correction. Runs with matched counts (the new recorder)
need no correction and are used exactly as logged.

Usage:
    python deploy/laptop/render_trace.py \
        --session deploy/runs/20260520_192737_figure_eight_5/raw \
        --start 3.5 --end 13.0
"""
import argparse
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
import numpy as np

try:
    from dt_apriltags import Detector
except ModuleNotFoundError:
    from pupil_apriltags import Detector

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

SETTINGS_FILE = os.path.join(_REPO_ROOT, "deploy", "configs", "camera_settings.json")

# --- pinned 20-May-evening vision tune (matches render_roi_explainer.py) -------
TUNE_ALPHA    = 0.9
TUNE_BETA     = -20.0
TUNE_BLUR     = 3
TUNE_DECIMATE = 2.0

OUT_W = 1280   # output width; height derived from source aspect ratio


def load_camera_params(settings_file, camera_name=None):
    with open(settings_file) as f:
        data = json.load(f)
    name = camera_name or "overhead"
    cam = data.get("cameras", {}).get(name, {})
    fx, fy, cx, cy = cam.get("camera_params", [1310.0, 1310.0, 960.0, 540.0])
    return [fx, fy, cx, cy]


# --------------------------------------------------------------------------- #
#  viridis colormap (embedded — no matplotlib at runtime; importing it next to
#  dt_apriltags segfaults this env's Detector destructor at shutdown)
# --------------------------------------------------------------------------- #
_VIRIDIS_ANCHORS = [
    (68, 1, 84), (72, 24, 106), (71, 45, 123), (66, 64, 134),
    (59, 82, 139), (51, 99, 141), (44, 114, 142), (38, 130, 142),
    (33, 145, 140), (31, 160, 136), (40, 174, 128), (63, 188, 115),
    (94, 201, 98), (132, 212, 75), (173, 220, 48), (216, 226, 25),
    (253, 231, 37),
]


def viridis_bgr(t):
    """Map t in [0,1] -> a BGR uint8 tuple by linearly interpolating viridis."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    n = len(_VIRIDIS_ANCHORS) - 1
    f = t * n
    i = int(f)
    if i >= n:
        r, g, b = _VIRIDIS_ANCHORS[n]
    else:
        w = f - i
        r0, g0, b0 = _VIRIDIS_ANCHORS[i]
        r1, g1, b1 = _VIRIDIS_ANCHORS[i + 1]
        r = r0 + (r1 - r0) * w
        g = g0 + (g1 - g0) * w
        b = b0 + (b1 - b0) * w
    return (int(b), int(g), int(r))


def draw_trace(img, trail):
    """Draw the time-coloured trajectory trace.

    trail: list of (x, y, frac) pixel points, frac in [0,1] = fraction of the
    run elapsed at that point (0 = start, 1 = now).
    """
    if len(trail) < 2:
        return
    pts = [(int(x), int(y)) for (x, y, _f) in trail]
    for i in range(1, len(pts)):                 # dark underline (legibility)
        cv2.line(img, pts[i - 1], pts[i], (20, 20, 20), 6, cv2.LINE_AA)
    for i in range(1, len(pts)):                 # coloured trace on top
        cv2.line(img, pts[i - 1], pts[i], viridis_bgr(trail[i][2]), 3, cv2.LINE_AA)


def draw_arrow(img, p_fish, p_target):
    """Clean white fish->target arrow with a subtle dark outline."""
    if p_fish is None or p_target is None:
        return
    p0 = (int(p_fish[0]), int(p_fish[1]))
    p1 = (int(p_target[0]), int(p_target[1]))
    cv2.arrowedLine(img, p0, p1, (15, 15, 15), 6, cv2.LINE_AA, tipLength=0.06)
    cv2.arrowedLine(img, p0, p1, (255, 255, 255), 3, cv2.LINE_AA, tipLength=0.06)


def preprocess(gray, alpha, beta, blur):
    img = gray
    if alpha != 1.0 or beta != 0.0:
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if blur >= 3:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


def tag_center(det):
    return (float(det.corners[:, 0].mean()), float(det.corners[:, 1].mean()))


# --------------------------------------------------------------------------- #
#  dropped-frame timeline reconstruction
# --------------------------------------------------------------------------- #
def _interp_positions(fish_px, n):
    """fish_px: list of (x,y) or None -> (n,2) array with NaNs interpolated."""
    pos = np.full((n, 2), np.nan)
    for i, p in enumerate(fish_px):
        if p is not None:
            pos[i] = p
    idx = np.arange(n)
    for c in range(2):
        good = ~np.isnan(pos[:, c])
        if good.sum() >= 2:
            pos[:, c] = np.interp(idx, idx[good], pos[good, c])
        else:
            pos[:, c] = 0.0
    return pos


def _smooth(x, w=15):
    if len(x) < w:
        return x.copy()
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def detect_drop_onset(fish_px, t_avi):
    """Find the AVI frame index where frame-dropping begins.

    Uses the fish's pixel speed under the (uncorrected) 1:1 mapping: while the
    recording is 1:1 the speed is a smooth, bounded signal; once frames start
    being skipped each surviving step covers more real motion in the same logged
    dt, so the speed steps up and stays up to the end. Returns K or None.
    """
    n = len(fish_px)
    if n < 60:
        return None
    pos = _interp_positions(fish_px, n)
    disp = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    dt = np.diff(t_avi)
    dt = np.r_[dt, np.median(dt) if len(dt) else 1.0]
    dt = np.clip(dt, 1e-3, None)
    vs = _smooth(disp / dt)

    base = np.median(vs[: max(1, int(0.4 * n))])      # phase-1 baseline
    thr = base * 1.6
    lo = int(0.25 * n)
    for i in range(lo, n - 30):
        # sustained crossing that persists toward the end
        if vs[i] > thr and vs[i:].mean() > thr and (vs[i:] > thr).mean() > 0.55:
            return i
    return None


def build_timeline(fish_px, t_raw, n_avi, onset_override=None,
                   method="auto", src_fps=30.0):
    """Return (frame_time[n_avi], report dict) — the corrected real time of each
    AVI frame.

    t_raw : capture times (rel. to first), length n_ts (>= n_avi if drops).
    """
    n_ts = len(t_raw)
    rep = {"n_avi": n_avi, "n_ts": n_ts, "drop": n_ts - n_avi}

    if n_ts == 0:
        rep["method"] = "fps-fallback"
        return np.arange(n_avi) / src_fps, rep

    if n_ts == n_avi or method == "off":
        rep["method"] = "exact" if n_ts == n_avi else "off(naive)"
        return t_raw[:n_avi].copy(), rep

    # --- drops present: reconstruct ---
    if method in ("auto", "twophase"):
        # under the naive mapping, AVI frame i is logged at t_raw[i]
        K = onset_override if onset_override is not None else \
            detect_drop_onset(fish_px, t_raw[:n_avi])
        if K is not None and 0 < K < n_avi - 1:
            ft = np.empty(n_avi)
            ft[:K] = t_raw[:K]
            # spread phase-2 survivors evenly over their true remaining span
            ft[K:] = np.linspace(t_raw[K], t_raw[n_ts - 1], n_avi - K)
            rep.update(method="twophase", onset_frame=int(K),
                       onset_time=float(t_raw[K]),
                       phase1_fps=float(K / max(t_raw[K], 1e-6)),
                       phase2_fps=float((n_avi - K) / max(t_raw[n_ts - 1] - t_raw[K], 1e-6)))
            return ft, rep
        rep["onset_frame"] = None  # fall through to uniform

    # uniform resample across the full session span
    src_idx = np.round(np.linspace(0, n_ts - 1, n_avi)).astype(int)
    rep["method"] = "uniform" if method != "twophase" else "uniform(no-onset)"
    return t_raw[src_idx], rep


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Render a clean trace + arrow replay (VFR)")
    ap.add_argument("--session", required=True, help="raw session dir (video.avi + frame_timestamps.csv)")
    ap.add_argument("--env-config", default=None,
                    help="env_config.json for tag_ids (default: fish=0 anchor=1 target=2)")
    ap.add_argument("--tag_size", type=float, default=0.080)
    ap.add_argument("--camera", default=None)
    ap.add_argument("--start", type=float, default=0.0, help="clip start (s, corrected real time)")
    ap.add_argument("--end", type=float, default=None, help="clip end (s); default = whole run")
    ap.add_argument("--alpha", type=float, default=TUNE_ALPHA)
    ap.add_argument("--beta", type=float, default=TUNE_BETA)
    ap.add_argument("--blur", type=int, default=TUNE_BLUR)
    ap.add_argument("--decimate", type=float, default=TUNE_DECIMATE)
    ap.add_argument("--flip_video", action="store_true",
                    help="flip camera feed 180 (fixes upside-down MSMF recordings)")
    ap.add_argument("--drop_fix", default="auto",
                    choices=["auto", "twophase", "uniform", "off"],
                    help="dropped-frame timing correction (default auto = twophase, "
                         "uniform fallback; off = naive 1:1)")
    ap.add_argument("--drop_onset", type=int, default=None,
                    help="manual phase-2 onset AVI frame index (overrides detection)")
    ap.add_argument("--out_name", default="replay_trace.mp4")
    args = ap.parse_args()

    session = os.path.abspath(args.session)

    vid_path = os.path.join(session, "video.avi")
    if not os.path.exists(vid_path):
        vid_path = os.path.join(session, "video.mp4")
    if not os.path.exists(vid_path):
        sys.exit(f"ERROR: no video in {session} (expected video.avi or video.mp4)")

    tag_ids = {"fish": 0, "anchor": 1, "target": 2}
    if args.env_config:
        with open(args.env_config) as f:
            tag_ids = json.load(f).get("tag_ids", tag_ids)
    fish_id, target_id = tag_ids["fish"], tag_ids["target"]

    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=args.decimate,
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    # raw capture timestamps (relative to first)
    ts_path = os.path.join(session, "frame_timestamps.csv")
    t_raw = np.empty(0)
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            rows = list(csv.DictReader(f))
        tv = np.array([float(r["laptop_time"]) for r in rows])
        t_raw = tv - tv.min() if len(tv) else tv

    cap     = cv2.VideoCapture(vid_path)
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_h = int(round(OUT_W * H / W))
    out_h -= out_h % 2          # even dims for libx264 yuv420p
    sx, sy = OUT_W / W, out_h / H

    n_ts = len(t_raw)
    print(f"Video {W}x{H}, {total} AVI frames; {n_ts} timestamps "
          f"({n_ts - total:+d} drop)  span {t_raw[-1] if n_ts else 0:.1f}s")
    print(f"Tune: alpha={args.alpha} beta={args.beta} blur={args.blur} decimate={args.decimate}")

    # ---- PASS 1: detect + cache tag pixel centres (output coords) ----
    print("Pass 1/2: detecting tags...")
    fish_px = [None] * total            # fish centre per AVI frame (or None)
    target_px = [None] * total          # last-known target centre per AVI frame
    last_target = None
    t_wall = time.time()
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            total = fi
            break
        if args.flip_video:
            frame = cv2.flip(frame, -1)
        gray = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                          args.alpha, args.beta, args.blur)
        tag_map = {tg.tag_id: tg for tg in detector.detect(gray, estimate_tag_pose=False)}
        if fish_id in tag_map:
            cx, cy = tag_center(tag_map[fish_id])
            fish_px[fi] = (cx * sx, cy * sy)
        if target_id in tag_map:
            tx, ty = tag_center(tag_map[target_id])
            last_target = (tx * sx, ty * sy)
        target_px[fi] = last_target
        if (fi + 1) % 200 == 0:
            print(f"  {fi+1}/{total}  {(fi+1)/(time.time()-t_wall):.1f} fps")
    cap.release()
    fish_px = fish_px[:total]
    target_px = target_px[:total]

    # ---- build corrected timeline ----
    frame_time, rep = build_timeline(fish_px, t_raw, total,
                                     onset_override=args.drop_onset,
                                     method=args.drop_fix, src_fps=src_fps)
    print(f"Timeline: method={rep['method']}  "
          + (f"onset=frame {rep.get('onset_frame')} (t={rep.get('onset_time',0):.1f}s), "
             f"phase1~{rep.get('phase1_fps',0):.0f}fps  phase2~{rep.get('phase2_fps',0):.1f}fps"
             if rep.get('method') == 'twophase' else
             f"playback span {frame_time[-1]-frame_time[0]:.1f}s"))

    clip_start = args.start
    clip_end   = args.end if args.end is not None else frame_time[-1]
    span = max(clip_end - clip_start, 1e-6)

    # ---- PASS 2: render from cache + corrected timeline ----
    print("Pass 2/2: rendering...")
    cap = cv2.VideoCapture(vid_path)
    tmpdir = tempfile.mkdtemp(prefix="sofi_trace_")
    concat = ["ffconcat version 1.0"]
    trail = []
    out_idx = 0
    t_wall = time.time()
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_time[fi]
        if t < clip_start or t > clip_end:
            continue
        if args.flip_video:
            frame = cv2.flip(frame, -1)

        frac = (t - clip_start) / span
        if fish_px[fi] is not None:
            trail.append((fish_px[fi][0], fish_px[fi][1], frac))

        out = cv2.resize(frame, (OUT_W, out_h))
        draw_trace(out, trail)
        draw_arrow(out, fish_px[fi], target_px[fi])

        if fi + 1 < total:
            duration = max(frame_time[fi + 1] - frame_time[fi], 1.0 / 120.0)
        else:
            duration = 1.0 / src_fps

        p = os.path.join(tmpdir, f"{out_idx:06d}.jpg")
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 94])
        concat += [f"file '{p}'", f"duration {duration:.6f}"]
        out_idx += 1
        if out_idx % 200 == 0:
            print(f"  {out_idx} frames  t={t:.1f}s  {out_idx/(time.time()-t_wall):.1f} fps")
    cap.release()

    print(f"Rendered {out_idx} frames, trail length {len(trail)}.")
    if out_idx == 0:
        sys.exit("ERROR: no frames in clip range — check --start/--end.")

    concat_path = os.path.join(tmpdir, "concat.txt")
    with open(concat_path, "w") as f:
        f.write("\n".join(concat) + "\n")

    out_path = os.path.join(session, args.out_name)
    print("Assembling VFR video...")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path],
                       capture_output=True, text=True)
    shutil.rmtree(tmpdir)
    if r.returncode != 0:
        print("ffmpeg error:\n", r.stderr[-1500:])
        sys.exit(1)
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
