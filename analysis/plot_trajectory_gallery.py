#!/usr/bin/env python3
"""Report §5.2 trajectory-plot gallery (2026-07-09).

Renders the six deployment runs as clean XY trajectory plots in Liam's style:
time-coloured LineCollection, light/yellow = start -> dark/violet = end
(colormap viridis_r), fish start at the origin, target marked, metric axes.
Thinner line than plot_trajectory.py, full opacity. Writes NEW files
`{run}_traj.png` in strobe_plots/ -- it does NOT overwrite any existing plot
(plot_trajectory.py and figure_eight_trajectory.png are left untouched).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "cache")
OUT_DIR = os.path.join(_HERE, "strobe_plots")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP = "plasma_r"    # yellow(start) -> red -> violet -> blue(end), Liam's palette
LW = 2.0             # thin coloured line
LW_BG = 3.5          # dark outline for legibility (full opacity)

# (run, truncate_at_closest_approach, moving_target)
RUNS = [
    ("hope_didnt_work_2", True,  False),   # (a) first success
    ("insane_run_8",      True,  False),   # (b) heavy fluid disturbance
    ("insane_run_2",      True,  False),   # (c) backward-velocity success
    ("finetune_run_6",    True,  False),   # (d) OOD success
    ("finetune_run_8",    False, False),   # (e) OOD failure (full path)
    ("figure_eight_6",    False, True),    # (f) figure-eight (moving target)
]

def get_scale():
    d = np.load(os.path.join(CACHE_DIR, "insane_run_8.npz"), allow_pickle=True)
    fw, fp = d["fish_world"], d["fish_px"]
    valid = ~np.isnan(fw[:, 0]) & ~np.isnan(fp[:, 0])
    dx_w = np.nanmax(fw[valid, 0]) - np.nanmin(fw[valid, 0])
    dx_p = np.nanmax(fp[valid, 0]) - np.nanmin(fp[valid, 0])
    return dx_w / dx_p if dx_p > 0 else 0.002321

def min_dist_index(d):
    fw, tw = d["fish_world"], d["tgt_world"]
    valid = ~np.isnan(fw[:, 0]) & ~np.isnan(tw[:, 0])
    if valid.sum() == 0:
        return None
    tgt_med = np.median(tw[valid], axis=0)
    dist = np.hypot(fw[:, 0] - tgt_med[0], fw[:, 1] - tgt_med[1])
    vidx = np.where(valid)[0]
    return int(vidx[np.argmin(dist[valid])])

def plot_run(run, truncate, moving, m_per_px):
    d = np.load(os.path.join(CACHE_DIR, f"{run}.npz"), allow_pickle=True)
    fp, tp = d["fish_px"], d["tgt_px"]

    vf = ~np.isnan(fp[:, 0]) & ~np.isnan(fp[:, 1])
    cx, cy = fp[vf, 0][0], fp[vf, 1][0]           # origin = fish start

    end = len(fp)
    if truncate:
        mi = min_dist_index(d)
        if mi is not None:
            end = mi + 1

    fpt = fp[:end]
    vft = ~np.isnan(fpt[:, 0]) & ~np.isnan(fpt[:, 1])
    fx = (fpt[vft, 0] - cx) * m_per_px
    fy = -(fpt[vft, 1] - cy) * m_per_px           # flip Y

    fig = plt.figure(figsize=(6, 5), facecolor="white")
    ax = plt.gca()

    pts = np.array([fx, fy]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(0, len(segs))
    ax.add_collection(LineCollection(segs, colors="black", linewidth=LW_BG, zorder=1))
    lc = LineCollection(segs, cmap=CMAP, norm=norm, linewidth=LW, zorder=2)
    lc.set_array(np.arange(len(segs)))
    ax.add_collection(lc)

    ax.plot([0], [0], "o", ms=8, color="white", mec="black", mew=1.5,
            linestyle="None", zorder=4, label="Start")

    if moving:
        tv = ~np.isnan(tp[:end, 0]) & ~np.isnan(tp[:end, 1])
        ax.plot((tp[:end][tv, 0] - cx) * m_per_px, -(tp[:end][tv, 1] - cy) * m_per_px,
                "-", color="orange", lw=1.4, zorder=3, label="Target")
    else:
        tv = ~np.isnan(tp[:, 0]) & ~np.isnan(tp[:, 1])
        if tv.sum() > 0:
            tx = np.median((tp[tv, 0] - cx) * m_per_px)
            ty = np.median(-(tp[tv, 1] - cy) * m_per_px)
            ax.plot([tx], [ty], marker="*", ms=20, color="red", mec="black",
                    mew=1.0, linestyle="None", zorder=4, label="Target")

    ax.autoscale_view()
    ax.set_xlabel("x (m)", fontsize=12)
    ax.set_ylabel("y (m)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(fontsize=10, loc="best")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(OUT_DIR, f"{run}_traj.png")
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", out)

def main():
    m = get_scale()
    for run, trunc, moving in RUNS:
        plot_run(run, trunc, moving, m)

if __name__ == "__main__":
    main()
