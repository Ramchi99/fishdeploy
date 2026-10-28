#!/usr/bin/env python3
"""
Build the 2026-05-20 RL-deploy session analysis: real-time trace videos for every
run + a fish-frame reachability plot for the static-target runs.

Pipeline (idempotent, resumable — skips work whose output already exists):
  Phase A  detect tags once per run -> cache/<name>.npz   (pixel + world + timeline)
  Phase B  render VFR trace video    -> videos/<name>.mp4
  Phase C  reachability plot + data tables from the caches
  Phase D  RUN_CATALOG.md

RAW DATA IS READ-ONLY. Nothing is ever written into the runs/.../raw/ folders;
all outputs land under analysis/.

Run:  python analysis/build_analysis.py [--force] [--only NAME]
"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAPTOP = os.path.join(_HERE, "..", "deploy", "laptop")
sys.path.insert(0, _LAPTOP)
from render_trace import (Detector, preprocess, tag_center, draw_trace, draw_arrow,
                          build_timeline, load_camera_params, OUT_W, SETTINGS_FILE)

RUNS_DIR = os.path.join(_HERE, "..", "deploy", "runs")
VIDEOS   = os.path.join(_HERE, "videos")
REACH    = os.path.join(_HERE, "reachability")
CACHE    = os.path.join(_HERE, "cache")
for d in (VIDEOS, REACH, CACHE):
    os.makedirs(d, exist_ok=True)

TAG = {"fish": 0, "anchor": 1, "target": 2}
TUNE = dict(alpha=0.9, beta=-20.0, blur=3, decimate=2.0)
TAG_SIZE = 0.080
MISS_FLAG_THRESH = 0.10   # flag runs where >10% of frames miss the fish tag

# ---- group/outcome colours (BGR not needed; matplotlib RGB hex) -------------
COL = {
    ("static", "success"):      "#2ca02c",   # green
    ("static", "fail"):         "#d62728",   # red
    ("disturbance", "success"): "#1f77b4",   # blue
    ("disturbance", "fail"):    "#8e44ad",   # purple
}
COL_GREEN = "#2ca02c"
COL_RED   = "#d62728"

# --------------------------------------------------------------------------- #
#  MANIFEST — every run, from the user's annotations (2026-06-30)
#  group: static | disturbance | special | moving | figure_eight
#  outcome: success | fail | semi  (semi only for figure-eight 'semi failed')
#  reach: included in the reachability plot?
# --------------------------------------------------------------------------- #
def R(folder, given, group, outcome, note=""):
    return dict(folder=folder, given=given, group=group, outcome=outcome, note=note,
                reach=group in ("static", "disturbance"))

MANIFEST = [
    R("20260520_155106_hope_didnt_work_2", "hope_didnt_work_2", "static", "success", "FIRST EVER successful RL run"),
    R("20260520_155233_hope_didnt_work_3", "hope_didnt_work_3", "static", "fail"),
    R("20260520_162457_hope_didnt_work_5", "hope_didnt_work_5", "static", "success", "has frame drops"),
    R("20260520_162813_hope_didnt_work_6", "hope_didnt_work_6", "static", "success"),
    R("20260520_163038_polaris",           "polaris",           "special", "review", "SPECIAL — flagged for later review"),
    R("20260520_163530_polaris_2",         "polaris_2",         "static", "success"),
    R("20260520_164106_crazy_run_1",       "crazy_run_1",       "static", "fail"),
    R("20260520_164504_crazy_run_3",       "crazy_run_3",       "static", "fail"),
    R("20260520_164617_crazy_run_4",       "crazy_run_4",       "static", "success"),
    R("20260520_164933_crazy_run_5",       "crazy_run_5",       "static", "success"),
    R("20260520_165359_full_success_1",    "full_success_1",    "static", "success"),
    R("20260520_165628_full_success_2",    "full_success_2",    "static", "success"),
    R("20260520_165858_full_success_3",    "full_success_3",    "static", "success", "has frame drops"),
    R("20260520_170227_insane_run_1",      "insane_run_1",      "disturbance", "fail", "initial backward velocity"),
    R("20260520_170419_insane_run_2",      "insane_run_2",      "disturbance", "success", "initial backward velocity; has frame drops"),
    R("20260520_173157_insane_run_3",      "insane_run_3",      "disturbance", "success", "initial rotational velocity"),
    R("20260520_173319_insane_run_4",      "insane_run_4",      "disturbance", "success", "slight stick disturbance"),
    R("20260520_173429_insane_run_5",      "insane_run_5",      "disturbance", "success", "heavy stick disturbance"),
    R("20260520_173558_insane_run_6",      "insane_run_6",      "disturbance", "success", "fluid disturbance"),
    R("20260520_173717_insane_run_7",      "insane_run_7",      "disturbance", "success", "fluid disturbance"),
    R("20260520_173853_insane_run_8",      "insane_run_8",      "disturbance", "success", "heavy fluid disturbance"),
    R("20260520_174144_amazing_run_1",     "amazing_run_1",     "moving", "success", "moving simple target"),
    R("20260520_174706_amazing_run_2",     "amazing_run_2",     "moving", "success", "moving target, circle"),
    R("20260520_175021_amazing_run_3",     "amazing_run_3",     "figure_eight", "fail", "figure-eight attempt 1"),
    R("20260520_175136_amazing_run_4",     "amazing_run_4",     "figure_eight", "fail", "figure-eight attempt 2 — CORRUPT VIDEO (0 frames)"),
    R("20260520_175614_amazing_run_5",     "amazing_run_5",     "figure_eight", "fail", "figure-eight attempt 3"),
    R("20260520_175736_amazing_run_6",     "amazing_run_6",     "figure_eight", "semi", "figure-eight attempt 4"),
    R("20260520_180104_amazing_run_7",     "amazing_run_7",     "figure_eight", "fail", "figure-eight attempt 5"),
    R("20260520_180234_amazing_run_8",     "amazing_run_8",     "figure_eight", "semi", "figure-eight attempt 6"),
    R("20260520_182257_figure_eight_2",    "figure_eight_2",    "figure_eight", "fail", "figure-eight attempt 7"),
    R("20260520_182505_figure_eight_3",    "figure_eight_3",    "figure_eight", "fail", "figure-eight attempt 8"),
    R("20260520_192737_figure_eight_5",    "figure_eight_5",    "figure_eight", "semi", "figure-eight attempt 9; has frame drops"),
    R("20260520_193023_figure_eight_6",    "figure_eight_6",    "figure_eight", "success", "figure-eight attempt 10 — SUCCESS; has frame drops"),
    R("20260520_193345_finetune_run_1",    "finetune_run_1",    "static", "success"),
    R("20260520_193531_finetune_run_2",    "finetune_run_2",    "static", "fail", "target moved by hand at the end"),
    R("20260520_194106_finetune_run_3",    "finetune_run_3",    "static", "fail"),
    R("20260520_194717_finetune_run_4",    "finetune_run_4",    "static", "success", "heavy frame drops (~50%)"),
    R("20260520_195021_finetune_run_5",    "finetune_run_5",    "static", "success"),
    R("20260520_195142_finetune_run_6",    "finetune_run_6",    "static", "success", "target out-of-distribution (behind)"),
    R("20260520_195359_finetune_run_7",    "finetune_run_7",    "static", "success", "target out-of-distribution (behind)"),
    R("20260520_195615_finetune_run_8",    "finetune_run_8",    "static", "fail"),
    R("20260520_200038_finetune_run_9",    "finetune_run_9",    "static", "fail", "heavy frame drops (~48%)"),
    R("20260520_200406_finetune_run_10",   "finetune_run_10",   "static", "fail"),
]


# --------------------------------------------------------------------------- #
#  world-frame helpers (anchor tag defines the pool-fixed metric frame)
# --------------------------------------------------------------------------- #
# Right-handed top-down pool frame: origin at anchor, Y negated so +Y = fish's
# left (camera is Y-down). The fish's swim direction is the tag Y-axis
# (R_fish[:,1]); expressed in this same Y-negated frame its heading angle is
# atan2(-fwd_y, fwd_x). Positions and heading are therefore in ONE consistent
# right-handed frame, so the body-frame (fwd, lat) below is a true top-down view
# (fwd=+x = swim direction, lat=+y = fish's left).
def world_xy(p_t, t_anchor, R_anchor):
    p = R_anchor.T @ (p_t - t_anchor)
    return float(p[0]), -float(p[1])


def yaw_of(R_fish, R_anchor):
    fwd = (R_anchor.T @ R_fish)[:, 1]
    return float(math.atan2(-fwd[1], fwd[0]))


# --------------------------------------------------------------------------- #
#  Phase A — detect once, cache everything
# --------------------------------------------------------------------------- #
def phase_a_detect(run, detector, cam_params, force=False):
    name = run["given"]
    cache_path = os.path.join(CACHE, name + ".npz")
    if os.path.exists(cache_path) and not force:
        return cache_path
    raw = os.path.join(RUNS_DIR, run["folder"], "raw")
    vid = os.path.join(raw, "video.avi")
    if not os.path.exists(vid):
        print(f"  [{name}] NO video.avi — skipping")
        return None

    # raw capture timestamps
    ts_path = os.path.join(raw, "frame_timestamps.csv")
    t_raw = np.empty(0)
    if os.path.exists(ts_path):
        rows = list(csv.DictReader(open(ts_path)))
        tv = np.array([float(r["laptop_time"]) for r in rows])
        t_raw = tv - tv.min() if len(tv) else tv

    cap = cv2.VideoCapture(vid)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    Wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    Hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total == 0 or Wv == 0:
        cap.release()
        print(f"  [{name}] CORRUPT (0 frames) — skipping")
        np.savez(cache_path, corrupt=True, total=0)
        return cache_path
    out_h = int(round(OUT_W * Hv / Wv)); out_h -= out_h % 2
    sx, sy = OUT_W / Wv, out_h / Hv
    need_world = run["reach"]

    fish_px = np.full((total, 2), np.nan)
    tgt_px  = np.full((total, 2), np.nan)
    fish_world = np.full((total, 2), np.nan)
    fish_yaw   = np.full(total, np.nan)
    tgt_world  = np.full((total, 2), np.nan)
    R_anchor = t_anchor = None
    fish_seen = 0
    last_tgt_px = None
    t0 = time.time()
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            total = fi; break
        gray = preprocess(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                          TUNE["alpha"], TUNE["beta"], TUNE["blur"])
        tags = detector.detect(gray, estimate_tag_pose=need_world,
                               camera_params=cam_params if need_world else None,
                               tag_size=TAG_SIZE)
        tm = {t.tag_id: t for t in tags}
        if TAG["fish"] in tm:
            cx, cy = tag_center(tm[TAG["fish"]])
            fish_px[fi] = (cx * sx, cy * sy)
            fish_seen += 1
        if TAG["target"] in tm:
            cx, cy = tag_center(tm[TAG["target"]])
            last_tgt_px = (cx * sx, cy * sy)
        tgt_px[fi] = last_tgt_px if last_tgt_px is not None else (np.nan, np.nan)

        if need_world:
            if TAG["anchor"] in tm:
                R_anchor = tm[TAG["anchor"]].pose_R.copy()
                t_anchor = tm[TAG["anchor"]].pose_t.flatten().copy()
            if R_anchor is not None:
                if TAG["fish"] in tm:
                    fw = world_xy(tm[TAG["fish"]].pose_t.flatten(), t_anchor, R_anchor)
                    fish_world[fi] = fw
                    fish_yaw[fi] = yaw_of(tm[TAG["fish"]].pose_R, R_anchor)
                if TAG["target"] in tm:
                    tgt_world[fi] = world_xy(tm[TAG["target"]].pose_t.flatten(), t_anchor, R_anchor)
        if (fi + 1) % 300 == 0:
            print(f"  [{name}] detect {fi+1}/{total}  {(fi+1)/(time.time()-t0):.0f} fps")
    cap.release()

    fish_px = fish_px[:total]; tgt_px = tgt_px[:total]
    fish_world = fish_world[:total]; fish_yaw = fish_yaw[:total]; tgt_world = tgt_world[:total]
    miss_frac = 1.0 - fish_seen / max(total, 1)

    # corrected timeline (handles dropped frames)
    fish_px_list = [None if np.isnan(p[0]) else (p[0], p[1]) for p in fish_px]
    frame_time, rep = build_timeline(fish_px_list, t_raw, total, src_fps=30.0)

    np.savez(cache_path,
             corrupt=False, total=total, Wv=Wv, Hv=Hv, out_h=out_h,
             fish_px=fish_px, tgt_px=tgt_px,
             fish_world=fish_world, fish_yaw=fish_yaw, tgt_world=tgt_world,
             frame_time=frame_time, t_raw=t_raw,
             miss_frac=miss_frac, n_ts=len(t_raw),
             tl_method=rep.get("method", ""), tl_onset=rep.get("onset_frame", -1) or -1,
             need_world=need_world)
    print(f"  [{name}] cached  frames={total} miss={miss_frac*100:.1f}% "
          f"timeline={rep.get('method')}")
    return cache_path


# --------------------------------------------------------------------------- #
#  Phase B — render VFR trace video from cache
# --------------------------------------------------------------------------- #
def phase_b_render(run, force=False):
    name = run["given"]
    out_path = os.path.join(VIDEOS, name + ".mp4")
    if os.path.exists(out_path) and not force:
        return "exists"
    cache_path = os.path.join(CACHE, name + ".npz")
    if not os.path.exists(cache_path):
        return "no-cache"
    d = np.load(cache_path, allow_pickle=True)
    if bool(d["corrupt"]):
        return "corrupt"
    raw = os.path.join(RUNS_DIR, run["folder"], "raw")
    vid = os.path.join(raw, "video.avi")
    fish_px = d["fish_px"]; tgt_px = d["tgt_px"]; frame_time = d["frame_time"]
    total = int(d["total"]); out_h = int(d["out_h"])
    Wv = int(d["Wv"]); Hv = int(d["Hv"])
    src_fps = 30.0
    t0v = frame_time[0]; span = max(frame_time[-1] - frame_time[0], 1e-6)

    import tempfile
    tmp = tempfile.mkdtemp(prefix="an_trace_")
    concat = ["ffconcat version 1.0"]
    cap = cv2.VideoCapture(vid)
    trail = []
    oi = 0
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_time[fi]
        frac = (t - t0v) / span
        fpx = None if np.isnan(fish_px[fi][0]) else (fish_px[fi][0], fish_px[fi][1])
        tpx = None if np.isnan(tgt_px[fi][0]) else (tgt_px[fi][0], tgt_px[fi][1])
        if fpx is not None:
            trail.append((fpx[0], fpx[1], frac))
        out = cv2.resize(frame, (OUT_W, out_h))
        draw_trace(out, trail)
        draw_arrow(out, fpx, tpx)
        if fi + 1 < total:
            dur = max(frame_time[fi + 1] - frame_time[fi], 1.0 / 120.0)
        else:
            dur = 1.0 / src_fps
        p = os.path.join(tmp, f"{oi:06d}.jpg")
        cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 94])
        concat += [f"file '{p}'", f"duration {dur:.6f}"]
        oi += 1
    cap.release()
    with open(os.path.join(tmp, "c.txt"), "w") as f:
        f.write("\n".join(concat) + "\n")
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", os.path.join(tmp, "c.txt"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", out_path],
                       capture_output=True, text=True)
    shutil.rmtree(tmp)
    if r.returncode != 0:
        print(f"  [{name}] ffmpeg error: {r.stderr[-400:]}")
        return "ffmpeg-error"
    return "ok"


# --------------------------------------------------------------------------- #
#  Phase A/B per-run derived reachability geometry (fish frame)
# --------------------------------------------------------------------------- #
def _velocity_gate(P, T, v_max=1.0, floor=0.12):
    """Mark world points that jump faster than physically possible (AprilTag
    pose-ambiguity / anchor-flip outliers). Returns a boolean keep-mask.
    A point is rejected if its step from the last ACCEPTED point exceeds
    v_max*dt + floor, so isolated outliers don't drag the chain."""
    n = len(P)
    keep = np.ones(n, bool)
    last = 0
    for k in range(1, n):
        dt = max(T[k] - T[last], 1e-3)
        if math.hypot(P[k, 0] - P[last, 0], P[k, 1] - P[last, 1]) > v_max * dt + floor:
            keep[k] = False
        else:
            last = k
    return keep


def _smooth_xy(P, w=5):
    if len(P) < w or w < 2:
        return P
    k = np.ones(w) / w
    out = P.copy()
    for c in range(2):
        out[:, c] = np.convolve(P[:, c], k, mode="same")
        out[:w, c] = P[:w, c]          # keep raw endpoints (convolve edge bias)
        out[-w:, c] = P[-w:, c]
    return out


def reach_geometry(run):
    """Return dict with fish-frame trajectory, target (fwd,lat,radius,angle),
    truncation, detection stats — or None if unusable.

    Robustness: world-frame fish track is velocity-gated to drop pose-ambiguity
    outliers (jumps > ~1 m/s), rejected points re-interpolated, then lightly
    smoothed. Initial pose uses a circular-mean yaw over the first kept frames.
    """
    name = run["given"]
    d = np.load(os.path.join(CACHE, name + ".npz"), allow_pickle=True)
    if bool(d["corrupt"]):
        return None
    fw = d["fish_world"]; yaw = d["fish_yaw"]; tw = d["tgt_world"]
    ft = d["frame_time"]
    good = ~np.isnan(fw[:, 0]) & ~np.isnan(yaw)
    if good.sum() < 10:
        return None
    gi = np.where(good)[0]
    P = fw[gi].astype(float)
    T = ft[gi].astype(float)
    yg = yaw[gi].astype(float)

    # reject physically-impossible jumps, re-interpolate the gaps
    keep = _velocity_gate(P, T)
    n_out = int((~keep).sum())
    if keep.sum() >= 5:
        idx = np.arange(len(P))
        for c in range(2):
            P[~keep, c] = np.interp(idx[~keep], idx[keep], P[keep, c])

    # initial pose: position median + circular-mean yaw over first kept frames
    kept = np.where(keep)[0]
    first = kept[:8] if len(kept) >= 8 else kept
    x0, y0 = np.median(P[first, 0]), np.median(P[first, 1])
    yaw0 = math.atan2(np.mean(np.sin(yg[first])), np.mean(np.cos(yg[first])))
    c, s = math.cos(yaw0), math.sin(yaw0)

    def to_body(x, y):
        dx, dy = x - x0, y - y0
        return (c * dx + s * dy, -s * dx + c * dy)   # fwd, lat (+lat = fish left)

    traj = np.array([to_body(px, py) for px, py in P])
    traj = _smooth_xy(traj, w=5)
    # target: median of all valid target detections, in body frame
    tg = tw[~np.isnan(tw[:, 0])]
    if len(tg) < 3:
        return None
    tgx, tgy = np.median(tg[:, 0]), np.median(tg[:, 1])
    t_fwd, t_lat = to_body(tgx, tgy)
    radius = math.hypot(t_fwd, t_lat)
    angle = math.degrees(math.atan2(t_lat, t_fwd))   # 0=ahead +=left -=right

    # truncation: success -> closest approach; fail -> full
    dist = np.hypot(traj[:, 0] - t_fwd, traj[:, 1] - t_lat)
    if run["outcome"] == "success":
        kmin = int(np.argmin(dist))
        traj_plot = traj[:kmin + 1]
        min_dist = float(dist[kmin])
    else:
        traj_plot = traj
        min_dist = float(dist.min())
    return dict(name=name, traj=traj_plot, t_fwd=t_fwd, t_lat=t_lat,
                radius=radius, angle=angle, min_dist=min_dist,
                miss_frac=float(d["miss_frac"]), n_outlier=n_out, group=run["group"],
                outcome=run["outcome"], note=run["note"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="process only this given-name (substring)")
    ap.add_argument("--skip_video", action="store_true")
    args = ap.parse_args()

    runs = [r for r in MANIFEST if (args.only is None or args.only in r["given"])]
    cam_params = load_camera_params(SETTINGS_FILE)
    detector = Detector(families="tag36h11", nthreads=4, quad_decimate=TUNE["decimate"],
                        quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)

    print(f"=== Phase A: detect ({len(runs)} runs) ===")
    for run in runs:
        phase_a_detect(run, detector, cam_params, force=args.force)

    if not args.skip_video:
        print(f"\n=== Phase B: render videos ===")
        for run in runs:
            st = phase_b_render(run, force=args.force)
            print(f"  [{run['given']}] video: {st}")

    print("\n=== Phase C/D handled by build_reach.py and build_catalog.py ===")
    # flush and avoid dt_apriltags destructor segfault at shutdown
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
