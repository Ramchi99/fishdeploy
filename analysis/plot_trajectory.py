#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))   # repository root
from paths import ANALYSIS  # noqa: E402
CACHE_DIR = str(ANALYSIS / "cache")
OUT_DIR = str(ANALYSIS / "strobe_plots")
os.makedirs(OUT_DIR, exist_ok=True)

def get_scale():
    # Use insane_run_8 to find the meter-to-pixel scale factor
    d_ref = np.load(os.path.join(CACHE_DIR, "insane_run_8.npz"), allow_pickle=True)
    fw_ref = d_ref["fish_world"]
    fp_ref = d_ref["fish_px"]
    valid = ~np.isnan(fw_ref[:,0]) & ~np.isnan(fp_ref[:,0])
    
    dx_w = np.nanmax(fw_ref[valid, 0]) - np.nanmin(fw_ref[valid, 0])
    dx_p = np.nanmax(fp_ref[valid, 0]) - np.nanmin(fp_ref[valid, 0])
    
    if dx_p > 0:
        return dx_w / dx_p
    return 0.002321

def main():
    m_per_px = get_scale()
    
    d = np.load(os.path.join(CACHE_DIR, "figure_eight_6.npz"), allow_pickle=True)
    
    fp = d["fish_px"]
    valid_f = ~np.isnan(fp[:, 0]) & ~np.isnan(fp[:, 1])
    
    # Use the starting position of the fish as the (0,0) origin!
    cx = fp[valid_f, 0][0]
    cy = fp[valid_f, 1][0]
    print("Using fish start position as (0,0)")
    
    fx_m = (fp[valid_f, 0] - cx) * m_per_px
    fy_m = -(fp[valid_f, 1] - cy) * m_per_px  # Flip Y
    
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.gca()
    
    # Use a LineCollection to draw the trajectory with a colormap (viridis)
    points = np.array([fx_m, fy_m]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a continuous norm to map from 0 to the number of segments
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=4, zorder=2)
    # Set the values used for colormapping (time fraction)
    lc.set_array(np.arange(len(segments)))
    
    # Add a subtle dark outline for legibility (like the strobe plots!)
    lc_bg = LineCollection(segments, colors='black', linewidth=7, zorder=1)
    
    ax.add_collection(lc_bg)
    line = ax.add_collection(lc)
    
    # Add a subtle marker for the start point
    plt.plot([0], [0], marker='o', markersize=10, color='white', markeredgecolor='black', markeredgewidth=2, label='Start', linestyle='None', zorder=3)
    
    # Ensure axes limits encapsulate the data
    ax.autoscale_view()
    
    plt.xlabel("X Position (meters)", fontsize=14)
    plt.ylabel("Y Position (meters)", fontsize=14)
    plt.title("Figure 8 Pursuit Trajectory", fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6, zorder=0)
    plt.legend(fontsize=12, loc='upper right')
    plt.axis('equal')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    out_path = os.path.join(OUT_DIR, "figure_eight_trajectory.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
