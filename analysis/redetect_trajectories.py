#!/usr/bin/env python3
"""Re-detect metric fish/target trajectories from the raw video (2026-07-09).

Multi-pass AprilTag pose estimation (tag_size=0.080 + overhead intrinsics) over
every frame; merges detections across passes (a tag missed by one parameter set
is often found by another); fixes a stable anchor-tag world frame; linearly fills
only SHORT gaps. Writes {run}_redetect.npz (fish_world, tgt_world, frame_time) --
does not touch the existing cache. Usage: python3 redetect_trajectories.py <run> [<run> ...]
"""
import cv2, numpy as np, os, glob, sys
from dt_apriltags import Detector

CAM = [1055.0202, 1054.2105, 945.8357, 489.0351]   # overhead intrinsics (vision_settings.json)
TAG = 0.080
FISH, ANCHOR, TARGET = 0, 1, 2
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "..", "deploy", "runs")
CACHE_DIR = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(HERE, "cache_redetect")
os.makedirs(OUT_DIR, exist_ok=True)

PASSES = [
    dict(quad_decimate=1.0, quad_sigma=0.0, decode_sharpening=0.25),
    dict(quad_decimate=1.0, quad_sigma=0.8, decode_sharpening=0.5),
    dict(quad_decimate=2.0, quad_sigma=0.0, decode_sharpening=1.0),
]

def world_xy(t_tag, t_anchor, R_anchor):
    p = R_anchor.T @ (t_tag - t_anchor)
    return float(p[0]), float(p[1])

def interp_short_gaps(a, max_gap=12):
    out = a.copy()
    for c in range(a.shape[1]):
        col = out[:, c]
        valid = np.where(~np.isnan(col))[0]
        for k in range(len(valid) - 1):
            i0, i1 = valid[k], valid[k + 1]
            if 1 < i1 - i0 <= max_gap + 1:
                col[i0:i1 + 1] = np.interp(np.arange(i0, i1 + 1), [i0, i1], [col[i0], col[i1]])
    return out

def redetect(run):
    vids = glob.glob(os.path.join(RUNS_DIR, f"*{run}*", "raw", "video.avi"))
    if not vids:
        print(f"  no video for {run}; skipping"); return None
    vid = vids[0]
    cap = cv2.VideoCapture(vid)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fish = np.full((total, 3), np.nan); tgt = np.full((total, 3), np.nan)
    anc_t = np.full((total, 3), np.nan); anc_R = np.full((total, 3, 3), np.nan)
    for pi, ps in enumerate(PASSES):
        det = Detector(families="tag36h11", nthreads=4, refine_edges=1, **ps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fi = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fi += 1
            if (not np.isnan(anc_t[fi, 0])) and (not np.isnan(fish[fi, 0])) and (not np.isnan(tgt[fi, 0])):
                continue                                          # frame already complete
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tm = {t.tag_id: t for t in det.detect(g, estimate_tag_pose=True, camera_params=CAM, tag_size=TAG)}
            if ANCHOR in tm and np.isnan(anc_t[fi, 0]):
                anc_t[fi] = tm[ANCHOR].pose_t.flatten(); anc_R[fi] = tm[ANCHOR].pose_R
            if FISH in tm and np.isnan(fish[fi, 0]):
                fish[fi] = tm[FISH].pose_t.flatten()
            if TARGET in tm and np.isnan(tgt[fi, 0]):
                tgt[fi] = tm[TARGET].pose_t.flatten()
        na = np.sum(~np.isnan(anc_t[:, 0])); nf = np.sum(~np.isnan(fish[:, 0])); nt = np.sum(~np.isnan(tgt[:, 0]))
        print(f"  pass {pi} {ps}: anchor {na}/{total}  fish {nf}/{total}  target {nt}/{total}")

    # stable (fixed) anchor world frame: median translation, rotation from the nearest frame
    av = np.where(~np.isnan(anc_t[:, 0]))[0]
    t_anc = np.median(anc_t[av], axis=0)
    R_anc = anc_R[av[np.argmin(np.linalg.norm(anc_t[av] - t_anc, axis=1))]]

    fw = np.full((total, 2), np.nan); tw = np.full((total, 2), np.nan)
    for fi in range(total):
        if not np.isnan(fish[fi, 0]):
            fw[fi] = world_xy(fish[fi], t_anc, R_anc)
        if not np.isnan(tgt[fi, 0]):
            tw[fi] = world_xy(tgt[fi], t_anc, R_anc)
    fw = interp_short_gaps(fw); tw = interp_short_gaps(tw)

    # frame_time: prefer the raw frame_timestamps.csv (true, variable-fps); else cache; else assume 30 fps
    ft = None
    ts = glob.glob(os.path.join(RUNS_DIR, f"*{run}*", "raw", "frame_timestamps.csv"))
    if ts:
        try:
            arr = np.loadtxt(ts[0], delimiter=",", skiprows=1)
            t = arr[:, 1] - arr[0, 1]
            if len(t) >= total:
                ft = t[:total]
        except Exception:
            ft = None
    if ft is None:
        c = np.load(os.path.join(CACHE_DIR, f"{run}.npz"), allow_pickle=True)
        ft = c["frame_time"] if "frame_time" in c.files and len(c["frame_time"]) == total else np.arange(total) / 30.0
    out = os.path.join(OUT_DIR, f"{run}_redetect.npz")
    np.savez(out, fish_world=fw, tgt_world=tw, frame_time=ft, total=total)
    nf = int(np.sum(~np.isnan(fw[:, 0]))); nt = int(np.sum(~np.isnan(tw[:, 0]))); na = int(len(av))
    print(f"  -> saved {out}  (fish {nf}/{total}, target {nt}/{total})")
    return dict(run=run, total=total, fish=nf, target=nt, anchor=na,
                fish_pct=100.0 * nf / total, target_pct=100.0 * nt / total, anchor_pct=100.0 * na / total)

def all_cache_runs():
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(CACHE_DIR, "*.npz")))

if __name__ == "__main__":
    args = sys.argv[1:]
    runs = all_cache_runs() if (not args or args[0] == "all") else args
    force = "force" in args
    print(f"Re-detecting {len(runs)} runs -> {OUT_DIR}  (skip-existing={not force})\n")
    for i, r in enumerate(runs):
        if (not force) and os.path.exists(os.path.join(OUT_DIR, f"{r}_redetect.npz")):
            print(f"=== [{i+1}/{len(runs)}] {r} — already done, skip ===")
            continue
        print(f"=== [{i+1}/{len(runs)}] {r} ===")
        try:
            redetect(r)
        except Exception as e:
            print(f"  ERROR on {r}: {e}")
    # coverage summary from ALL redetect npz (fish/target % recomputed from saved arrays), worst-first
    rows = []
    for f in glob.glob(os.path.join(OUT_DIR, "*_redetect.npz")):
        d = np.load(f, allow_pickle=True)
        tot = int(d["total"])
        fw, tw = d["fish_world"], d["tgt_world"]
        rows.append((os.path.basename(f)[:-len("_redetect.npz")], tot,
                     100.0 * np.sum(~np.isnan(fw[:, 0])) / tot,
                     100.0 * np.sum(~np.isnan(tw[:, 0])) / tot))
    rows.sort(key=lambda t: t[2])
    md = os.path.join(OUT_DIR, "coverage_summary.md")
    with open(md, "w") as fh:
        fh.write("# Re-detection coverage per run (each frame of the raw video)\n\n")
        fh.write("| run | frames | fish % | target % |\n|---|--:|--:|--:|\n")
        for name, tot, fp, tp in rows:
            fh.write(f"| {name} | {tot} | {fp:.1f} | {tp:.1f} |\n")
    print(f"\n=== DONE: {len(rows)} runs in cache_redetect. Coverage summary -> {md} ===")
    for name, tot, fp, tp in rows:
        print(f"  {name:26s} fish {fp:5.1f}%  target {tp:5.1f}%")
