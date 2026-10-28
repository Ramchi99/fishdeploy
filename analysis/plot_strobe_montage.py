#!/usr/bin/env python3
"""Assemble the 6 v2 'traceover' deployment strobes into a 2x3 montage (matches the
trajectory-collage style: serif, medium-weight titles, tight uniform gaps)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.titlesize": 13, "axes.titleweight": "medium",
})

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, "report_strobes")
OUT = os.path.join(_HERE, "report_strobes", "deploy_strobe_montage.png")

PANELS = [
    ("hope_didnt_work_2_highres_traceover_strobe.png",      "(a) First successful run"),
    ("insane_run_8_middle_bg_highres_traceover_strobe.png", "(b) Fluid disturbance"),
    ("insane_run_2_highres_traceover_strobe.png",           "(c) Backward-velocity start"),
    ("finetune_run_7_highres_traceover_strobe.png",         "(d) Out-of-distribution (success)"),
    ("finetune_run_8_highres_traceover_strobe.png",         "(e) Lateral-rear failure"),
    ("figure_eight_6_clean_highres_traceover_strobe.png",   "(f) Figure-eight pursuit"),
]

fig, axes = plt.subplots(2, 3, figsize=(13.5, 5.9), layout="constrained")
for ax, (fn, title) in zip(axes.flat, PANELS):
    ax.imshow(plt.imread(os.path.join(SRC, fn)))
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.01, hspace=0.03)
fig.savefig(OUT, dpi=200, facecolor="white")
print("Saved", OUT)
