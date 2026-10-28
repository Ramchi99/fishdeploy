#!/usr/bin/env python3
"""
Phase C — fish-frame reachability plots + angle/radius tables for the 2026-05-20
static-target session. Reads the per-run caches written by build_analysis.py.

Trajectory plots (all capped at 3 m radius):
  reachability_trajectories.png   static + disturbance, 4 categories
                                  (green success, red fail, blue dist-success, purple dist-fail)
  reachability_static_only.png    static runs only — green success / red fail
  reachability_all_by_outcome.png all runs, coloured purely by outcome (green/red)
  reachability_polar.png          target angle vs radius (4 categories)

Tables:
  run_geometry.csv                per-run radius / angle / min-dist / detection
  reachability_summary.md         reached ratios + angle/radius coverage + caveats
"""
import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))   # repository root
from paths import ANALYSIS  # noqa: E402
sys.path.insert(0, _HERE)
import build_analysis as B

REACH = str(ANALYSIS / "reachability")
os.makedirs(REACH, exist_ok=True)

RMAX = 3.0    # fixed plot radius (m) for every plot

# fixed legend order requested: green, red, blue, purple
LEGEND_ORDER = [
    (B.COL[("static", "success")],      "success"),
    (B.COL[("static", "fail")],         "fail"),
    (B.COL[("disturbance", "success")], "disturbance — success"),
    (B.COL[("disturbance", "fail")],    "disturbance — fail"),
]


def gather():
    out = []
    for run in B.MANIFEST:
        if not run["reach"]:
            continue
        if not os.path.exists(os.path.join(B.CACHE, run["given"] + ".npz")):
            continue
        g = B.reach_geometry(run)
        if g is not None:
            out.append(g)
    return out


def col_4cat(g):
    return B.COL[(g["group"], g["outcome"])]


def col_outcome(g):
    return B.COL_GREEN if g["outcome"] == "success" else B.COL_RED


def plot_traj(geos, colorfn, legend_items, title, path):
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.set_facecolor("white")
    for r in range(1, int(RMAX) + 1):                     # range rings
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=0.8,
                                 ec="#cccccc", zorder=0))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)

    for g in geos:
        col = colorfn(g)
        tr = g["traj"]
        ax.plot(tr[:, 0], tr[:, 1], color=col, lw=1.6, alpha=0.55, zorder=2)
        ax.plot(tr[-1, 0], tr[-1, 1], "o", color=col, ms=4, alpha=0.9, zorder=3)
        ax.scatter([g["t_fwd"]], [g["t_lat"]], s=70, marker="s", color=col,
                   edgecolors="#222222", linewidths=0.6, zorder=4)

    ax.annotate("", xy=(0.55, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1f4e9c", lw=3))
    ax.plot(0, 0, "o", color="#1f4e9c", ms=16, zorder=5)
    ax.text(0.05, -0.20, "fish", color="#1f4e9c", fontsize=11, fontweight="bold")

    ax.set_aspect("equal")
    ax.set_xlim(-RMAX, RMAX); ax.set_ylim(-RMAX, RMAX)
    ax.set_xlabel("forward   x_body  (m)", fontsize=12)
    ax.set_ylabel("left   y_body  (m)", fontsize=12)
    ax.set_title(title, fontsize=12)
    handles = [Line2D([0], [0], marker="s", ls="", mec="#222", mfc=c, ms=11, label=l)
               for c, l in legend_items]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print("saved", path)


def main():
    geos = gather()
    if not geos:
        sys.exit("no reachability geometry found — run build_analysis.py first")

    stat = [g for g in geos if g["group"] == "static"]
    dist = [g for g in geos if g["group"] == "disturbance"]
    n_s, n_s_ok = len(stat), sum(g["outcome"] == "success" for g in stat)
    n_d, n_d_ok = len(dist), sum(g["outcome"] == "success" for g in dist)
    n_all, n_all_ok = len(geos), sum(g["outcome"] == "success" for g in geos)

    # ---- plot 1: static + disturbance, 4 categories ----
    plot_traj(
        geos, col_4cat, LEGEND_ORDER,
        "RL deploy 2026-05-20 — fish-frame reachability  (static + disturbance)\n"
        f"static {n_s_ok}/{n_s} reached · disturbance {n_d_ok}/{n_d} reached"
        "   · fish at origin facing +x · targets = squares · max radius 3 m",
        os.path.join(REACH, "reachability_trajectories.png"))

    # ---- plot 2: static only, green/red ----
    plot_traj(
        stat, col_outcome, LEGEND_ORDER[:2],
        "RL deploy 2026-05-20 — fish-frame reachability  (static targets only)\n"
        f"{n_s_ok}/{n_s} reached   · fish at origin facing +x · max radius 3 m",
        os.path.join(REACH, "reachability_static_only.png"))

    # ---- plot 3: all runs, coloured by outcome only ----
    plot_traj(
        geos, col_outcome, LEGEND_ORDER[:2],
        "RL deploy 2026-05-20 — fish-frame reachability  (all runs, by outcome)\n"
        f"{n_all_ok}/{n_all} reached  (disturbance runs included as green/red)"
        "   · fish at origin facing +x · max radius 3 m",
        os.path.join(REACH, "reachability_all_by_outcome.png"))

    # ---- polar (4 categories, capped at 3 m) ----
    figp = plt.figure(figsize=(9, 9), dpi=150)
    axp = figp.add_subplot(111, projection="polar")
    axp.set_theta_zero_location("E"); axp.set_theta_direction(1)
    axp.set_rmax(RMAX)
    for g in geos:
        axp.scatter(math.radians(g["angle"]), min(g["radius"], RMAX), s=90, marker="s",
                    color=col_4cat(g), edgecolors="#222", linewidths=0.6, zorder=3)
    axp.set_title("Target angle vs radius in fish frame  (max 3 m)\n"
                  "0° = straight ahead, +90° = fish's left, −90° = right, ±180° = behind",
                  fontsize=12, pad=20)
    axp.set_rlabel_position(135)
    handles = [Line2D([0], [0], marker="s", ls="", mec="#222", mfc=c, ms=10, label=l)
               for c, l in LEGEND_ORDER]
    axp.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.15, -0.05), fontsize=9)
    figp.tight_layout()
    p2 = os.path.join(REACH, "reachability_polar.png")
    figp.savefig(p2, dpi=150); plt.close(figp)
    print("saved", p2)

    # ---- CSV table ----
    csv_path = os.path.join(REACH, "run_geometry.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "group", "outcome", "radius_m", "angle_deg",
                    "min_dist_m", "fish_miss_pct", "note"])
        for g in sorted(geos, key=lambda x: (x["group"],
                        -1 if x["outcome"] == "success" else 1, x["angle"])):
            w.writerow([g["name"], g["group"], g["outcome"], f"{g['radius']:.3f}",
                        f"{g['angle']:.1f}", f"{g['min_dist']:.3f}",
                        f"{g['miss_frac']*100:.1f}", g["note"]])
    print("saved", csv_path)

    # ---- summary md ----
    a_ok = sorted(g["angle"] for g in geos if g["outcome"] == "success")
    r_ok = [g["radius"] for g in geos if g["outcome"] == "success"]
    a_no = sorted(g["angle"] for g in geos if g["outcome"] == "fail")
    r_no = [g["radius"] for g in geos if g["outcome"] == "fail"]
    L = ["# Reachability summary — 2026-05-20 session\n",
         f"- **Static targets reached:** {n_s_ok}/{n_s} ({100*n_s_ok/max(n_s,1):.0f}%)",
         f"- **Disturbance targets reached:** {n_d_ok}/{n_d} ({100*n_d_ok/max(n_d,1):.0f}%)",
         f"- **All reachability runs:** {n_all_ok}/{n_all} ({100*n_all_ok/max(n_all,1):.0f}%)\n",
         "> ⚠️ **Caveat:** targets were placed by hand, NOT sampled randomly/uniformly, "
         "so these ratios are descriptive of this session only — not an unbiased "
         "reachability rate.\n"]
    if a_ok:
        L.append(f"- Reached target angles span **{min(a_ok):.0f}°…{max(a_ok):.0f}°**, "
                 f"radii **{min(r_ok):.2f}…{max(r_ok):.2f} m**.")
    if a_no:
        L.append(f"- Failed target angles span **{min(a_no):.0f}°…{max(a_no):.0f}°**, "
                 f"radii **{min(r_no):.2f}…{max(r_no):.2f} m**.")
    L.append("\n(angle: 0° = straight ahead, +90° = fish's left, −90° = right, "
             "±180° = directly behind. Plots are capped at 3 m radius.)")
    open(os.path.join(REACH, "reachability_summary.md"), "w").write("\n".join(L) + "\n")
    print("saved", os.path.join(REACH, "reachability_summary.md"))


if __name__ == "__main__":
    main()
