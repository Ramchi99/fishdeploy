#!/usr/bin/env python3
"""
Animated fish-frame reachability build-up for the 2026-05-20 session.

The 30 reachability runs (static + disturbance) appear one after another in the
order  static-success -> static-fail -> disturbance-success -> disturbance-fail.
For each run its target square pops in and the trajectory grows out from the
fish (origin) with a bright leading head; finished runs stay on screen (faded),
so the animation builds up into the full reachability plot.

Reveal is index-uniform with a slow->fast ramp (first run slowest). Total ~20 s.

Output: reachability/reachability_animation.mp4   (CFR, H.264)
Run:    python analysis/build_reach_anim.py
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import build_analysis as B
import build_reach as BR

REACH = os.path.join(_HERE, "reachability")

RMAX        = 3.0
FPS         = 30
DRAW_BUDGET = 22.0    # total seconds spent drawing trajectories
HOLD        = 0.12    # pause after each run finishes
T_BRIGHT    = 2.0     # seconds to brighten every run at the end
END_HOLD    = 2.4     # hold on the finished (bright) plot
SHAPE_HALF  = 2.5     # ramp half-life (runs) — smaller = quicker speed-up
# alpha: completed runs stay dim while the active one is bright; all brighten at the end
DIM_LINE, FINAL_LINE = 0.13, 0.60     # completed-trajectory alpha: during draw / final
DIM_SQ,   FINAL_SQ   = 0.40, 1.00     # completed-target-square alpha: during draw / final
ACTIVE_LINE          = 0.97           # the currently-drawing trajectory

ORDER = [("static", "success"), ("static", "fail"),
         ("disturbance", "success"), ("disturbance", "fail")]
LABEL = {("static", "success"): "static — success", ("static", "fail"): "static — fail",
         ("disturbance", "success"): "disturbance — success",
         ("disturbance", "fail"): "disturbance — fail"}
LEGEND = [(B.COL[k], LABEL[k]) for k in ORDER]


def setup_axes(ax):
    ax.clear()
    ax.set_facecolor("white")
    for r in range(1, int(RMAX) + 1):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=0.8, ec="#cccccc", zorder=0))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.annotate("", xy=(0.55, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1f4e9c", lw=3))
    ax.plot(0, 0, "o", color="#1f4e9c", ms=15, zorder=5)
    ax.text(0.05, -0.22, "fish", color="#1f4e9c", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-RMAX, RMAX); ax.set_ylim(-RMAX, RMAX)
    ax.set_xlabel("forward   x_body  (m)", fontsize=12)
    ax.set_ylabel("left   y_body  (m)", fontsize=12)
    handles = [Line2D([0], [0], marker="s", ls="", mec="#222", mfc=c, ms=11, label=l)
               for c, l in LEGEND]
    ax.legend(handles=handles, loc="upper left", fontsize=9.5, framealpha=0.9)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reachability_animation.mp4",
                    help="output filename inside reachability/")
    ap.add_argument("--slow-first-disturbance", action="store_true",
                    help="draw the first disturbance run as slowly as the very first "
                         "run (extra time to narrate the disturbance section)")
    args = ap.parse_args()

    geos = BR.gather()
    buckets = {k: [] for k in ORDER}
    for g in geos:
        buckets[(g["group"], g["outcome"])].append(g)
    seq = [g for k in ORDER for g in buckets[k]]      # ordered run sequence
    n = len(seq)
    print(f"animating {n} reachability runs")

    # per-run draw duration: decreasing ramp, normalised to DRAW_BUDGET
    shape = np.array([0.35 + 1.0 * (0.5 ** (i / SHAPE_HALF)) for i in range(n)])
    draw = shape / shape.sum() * DRAW_BUDGET
    if args.slow_first_disturbance:                   # add a narration pause
        fd = next((i for i, g in enumerate(seq) if g["group"] == "disturbance"), None)
        if fd is not None:
            draw[fd] = draw[0]                        # as slow as the very first run
            print(f"  slowed first disturbance run (seq #{fd+1}) to {draw[0]:.2f}s")
    sched, t = [], 0.0
    for i, g in enumerate(seq):
        t0 = t; t1 = t + draw[i]; t = t1 + HOLD
        sched.append((g, t0, t1))
    t_draw_end = t
    total = t_draw_end + T_BRIGHT + END_HOLD
    frames = int(round(total * FPS))
    print(f"total {total:.1f}s  ({frames} frames @ {FPS}fps)")

    n_s_ok = len(buckets[("static", "success")])
    n_s = n_s_ok + len(buckets[("static", "fail")])
    n_d_ok = len(buckets[("disturbance", "success")])
    n_d = n_d_ok + len(buckets[("disturbance", "fail")])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    fig.suptitle("RL deploy 2026-05-20 — fish-frame reachability (build-up)", fontsize=13)

    def draw_full(g, col, line_a, sq_a):
        tr = g["traj"]
        ax.scatter([g["t_fwd"]], [g["t_lat"]], s=70, marker="s", color=col,
                   edgecolors="#222", linewidths=0.6, alpha=sq_a, zorder=4)
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.7, alpha=line_a, zorder=2)
        ax.plot(tr[-1, 0], tr[-1, 1], "o", color=col, ms=4,
                alpha=min(line_a + 0.25, 1.0), zorder=3)

    def draw_active(g, col, frac):
        tr = g["traj"]; N = len(tr)
        ax.scatter([g["t_fwd"]], [g["t_lat"]], s=80, marker="s", color=col,
                   edgecolors="#222", linewidths=0.8, alpha=1.0, zorder=8)
        k = max(2, int(frac * N)); seg = tr[:k]
        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=2.2, alpha=ACTIVE_LINE, zorder=6)
        ax.plot(seg[-1, 0], seg[-1, 1], "o", color=col, ms=10, zorder=7,
                mec="white", mew=1.4)

    def update(fi):
        t = fi / FPS
        setup_axes(ax)
        if t <= t_draw_end:                      # drawing phase
            b, drawing = 0.0, True
        elif t <= t_draw_end + T_BRIGHT:         # brighten-all phase
            b, drawing = (t - t_draw_end) / T_BRIGHT, False
        else:                                    # final bright hold
            b, drawing = 1.0, False
        line_a = DIM_LINE + (FINAL_LINE - DIM_LINE) * b
        sq_a = DIM_SQ + (FINAL_SQ - DIM_SQ) * b

        cur = None
        for i, (g, t0, t1) in enumerate(sched):
            if t < t0:
                continue
            col = B.COL[(g["group"], g["outcome"])]
            if drawing and t0 <= t < t1:
                draw_active(g, col, (t - t0) / max(t1 - t0, 1e-6))
            else:
                draw_full(g, col, line_a, sq_a)   # completed: dim, then brighten
            cur = i

        if drawing and cur is not None:           # per-run caption during build-up
            g = seq[cur]; col = B.COL[(g["group"], g["outcome"])]
            ax.text(0.985, 0.975, f"{LABEL[(g['group'], g['outcome'])]}   ·   run {cur+1}/{n}",
                    transform=ax.transAxes, fontsize=12, fontweight="bold", color=col,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))
        elif not drawing:                          # summary caption at the end
            ax.text(0.985, 0.975,
                    f"static {n_s_ok}/{n_s} · disturbance {n_d_ok}/{n_d} reached",
                    transform=ax.transAxes, fontsize=12, fontweight="bold", color="#333",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.85))
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / FPS, blit=False)
    out = os.path.join(REACH, args.out)
    writer = FFMpegWriter(fps=FPS, bitrate=5000,
                          extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer, dpi=110)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
