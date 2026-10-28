#!/usr/bin/env python3
"""
Animated build-up of the SIMULATION reachability map (CPG stage-8 policy, DR on,
deterministic) on the 22 real targets — same look/behaviour as the real-data
reachability animation (build_reach_anim.py): targets pop in, trajectories grow
with a bright head, completed runs dim, everything brightens at the end,
slow->fast ramp, ~30 s.

Only two buckets here (no disturbance runs): success (green) then fail (red).

Reads:  reachability/sim_mujoco/single_DR_deterministic.npz
Output: reachability/sim_mujoco/sim_reach_DR_deterministic_animation.mp4
"""
import argparse
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))   # repository root
from paths import ANALYSIS  # noqa: E402
SIMDIR = str(ANALYSIS / "reachability" / "sim_mujoco")

RMAX        = 3.0
FPS         = 30
DRAW_BUDGET = 22.0
HOLD        = 0.12
T_BRIGHT    = 2.0
END_HOLD    = 2.4
SHAPE_HALF  = 2.5
DIM_LINE, FINAL_LINE = 0.13, 0.60
DIM_SQ,   FINAL_SQ   = 0.40, 1.00
ACTIVE_LINE          = 0.97

COL = {True: "#2ca02c", False: "#d62728"}       # success green / fail red
LABEL = {True: "success", False: "fail"}
LEGEND = [("#2ca02c", "success"), ("#d62728", "fail")]


def load_geos(npz):
    d = np.load(npz, allow_pickle=True)
    trajs = d["trajectories"]; kmin = d["kmin"]; succ = d["success"]
    tfwd = d["tgt_fwd"]; tlat = d["tgt_lat"]; names = d["target_names"]
    geos = []
    for i in range(len(trajs)):
        tr = np.asarray(trajs[i], dtype=float)
        tr = tr[:int(kmin[i]) + 1] if bool(succ[i]) else tr
        geos.append(dict(traj=tr, t_fwd=float(tfwd[i]), t_lat=float(tlat[i]),
                         success=bool(succ[i]), name=str(names[i]),
                         angle=math.atan2(float(tlat[i]), float(tfwd[i]))))
    return geos


def setup_axes(ax):
    ax.clear(); ax.set_facecolor("white")
    for r in range(1, int(RMAX) + 1):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=0.8, ec="#cccccc", zorder=0))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.annotate("", xy=(0.55, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1f4e9c", lw=3))
    ax.plot(0, 0, "o", color="#1f4e9c", ms=15, zorder=5)
    ax.text(0.05, -0.22, "fish", color="#1f4e9c", fontsize=10, fontweight="bold")
    ax.set_aspect("equal"); ax.set_xlim(-RMAX, RMAX); ax.set_ylim(-RMAX, RMAX)
    ax.set_xlabel("forward   x_body  (m)", fontsize=12)
    ax.set_ylabel("left   y_body  (m)", fontsize=12)
    handles = [Line2D([0], [0], marker="s", ls="", mec="#222", mfc=c, ms=11, label=l)
               for c, l in LEGEND]
    ax.legend(handles=handles, loc="upper left", fontsize=9.5, framealpha=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(SIMDIR, "single_DR_deterministic.npz"))
    ap.add_argument("--out", default=os.path.join(SIMDIR, "sim_reach_DR_deterministic_animation.mp4"))
    args = ap.parse_args()

    geos = load_geos(args.npz)
    # order: success bucket then fail bucket; angular sweep within each
    seq = (sorted([g for g in geos if g["success"]], key=lambda g: g["angle"]) +
           sorted([g for g in geos if not g["success"]], key=lambda g: g["angle"]))
    n = len(seq)
    n_ok = sum(g["success"] for g in geos)
    print(f"animating {n} sim runs ({n_ok} success / {n - n_ok} fail)")

    shape = np.array([0.35 + 1.0 * (0.5 ** (i / SHAPE_HALF)) for i in range(n)])
    draw = shape / shape.sum() * DRAW_BUDGET
    sched, t = [], 0.0
    for i, g in enumerate(seq):
        t0 = t; t1 = t + draw[i]; t = t1 + HOLD
        sched.append((g, t0, t1))
    t_draw_end = t
    total = t_draw_end + T_BRIGHT + END_HOLD
    frames = int(round(total * FPS))
    print(f"total {total:.1f}s ({frames} frames)")

    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    fig.suptitle("SIM (DR on, deterministic) — fish-frame reachability build-up",
                 fontsize=13)

    def draw_full(g, col, line_a, sq_a):
        tr = g["traj"]
        ax.scatter([g["t_fwd"]], [g["t_lat"]], s=70, marker="s", color=col,
                   edgecolors="#222", linewidths=0.6, alpha=sq_a, zorder=4)
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.7, alpha=line_a, zorder=2)
        ax.plot(tr[-1, 0], tr[-1, 1], "o", color=col, ms=4, alpha=min(line_a + 0.25, 1.0), zorder=3)

    def draw_active(g, col, frac):
        tr = g["traj"]; N = len(tr)
        ax.scatter([g["t_fwd"]], [g["t_lat"]], s=80, marker="s", color=col,
                   edgecolors="#222", linewidths=0.8, alpha=1.0, zorder=8)
        k = max(2, int(frac * N)); seg = tr[:k]
        ax.plot(seg[:, 0], seg[:, 1], color=col, lw=2.2, alpha=ACTIVE_LINE, zorder=6)
        ax.plot(seg[-1, 0], seg[-1, 1], "o", color=col, ms=10, zorder=7, mec="white", mew=1.4)

    def update(fi):
        t = fi / FPS
        setup_axes(ax)
        if t <= t_draw_end:
            b, drawing = 0.0, True
        elif t <= t_draw_end + T_BRIGHT:
            b, drawing = (t - t_draw_end) / T_BRIGHT, False
        else:
            b, drawing = 1.0, False
        line_a = DIM_LINE + (FINAL_LINE - DIM_LINE) * b
        sq_a = DIM_SQ + (FINAL_SQ - DIM_SQ) * b
        cur = None
        for i, (g, t0, t1) in enumerate(sched):
            if t < t0:
                continue
            col = COL[g["success"]]
            if drawing and t0 <= t < t1:
                draw_active(g, col, (t - t0) / max(t1 - t0, 1e-6))
            else:
                draw_full(g, col, line_a, sq_a)
            cur = i
        if drawing and cur is not None:
            g = seq[cur]; col = COL[g["success"]]
            ax.text(0.985, 0.975, f"{LABEL[g['success']]}   ·   run {cur+1}/{n}",
                    transform=ax.transAxes, fontsize=12, fontweight="bold", color=col,
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, alpha=0.85))
        elif not drawing:
            ax.text(0.985, 0.975, f"{n_ok}/{n} reached in sim",
                    transform=ax.transAxes, fontsize=12, fontweight="bold", color="#333",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#888", alpha=0.85))
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000.0 / FPS, blit=False)
    writer = FFMpegWriter(fps=FPS, bitrate=5000, extra_args=["-pix_fmt", "yuv420p"])
    anim.save(args.out, writer=writer, dpi=110)
    plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()
