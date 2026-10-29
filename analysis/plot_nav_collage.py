#!/usr/bin/env python3
"""Report §5.2 navigation collage (2026-07-09) -- Liam-matched style.

One 2x3 figure: plasma trajectory (yellow start -> blue end) with direction
arrows, blue START dot, red CLOSEST-APPROACH dot, green TARGET star (target at
origin), green-dashed moving-target path, per-panel title + distance/time
annotation, shared legend, serif (LaTeX-like) font, square equal-aspect panels.
Uses the clean pixel tracks (x calibrated m/px) -- fish_world is noisy/NaN for
some runs. Writes a NEW file deploy_nav_collage.png -- overwrites nothing else.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.path import Path as MPath

def fat_star(inner=0.50, n=5):
    """A fuller 5-point star marker (larger inner radius => less pointy spikes, like Liam's)."""
    verts, codes = [], []
    for i in range(2 * n):
        ang = np.pi / 2 + i * np.pi / n
        rad = 1.0 if i % 2 == 0 else inner
        verts.append((rad * np.cos(ang), rad * np.sin(ang)))
        codes.append(MPath.MOVETO if i == 0 else MPath.LINETO)
    verts.append(verts[0]); codes.append(MPath.CLOSEPOLY)
    return MPath(verts, codes)

STAR = fat_star()

# The collage look (the report's "collage" tier, kept here so fishsim's Fig. 7 draws the same panels):
# roomy canvas scaled down by LaTeX, serif via usetex (REPORT_USETEX=0 for a STIX fallback), framed axes
# with a light grid (REPORT_VARIANT=airy removes the frame), Okabe-Ito colours.
COLLAGE_LW = 2.4                  # collage data lines (on-page ~1.1 after LaTeX scaling)
_CB_CYCLE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]
_FRAMED = os.environ.get("REPORT_VARIANT", "framed") != "airy"
_rc = {
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "font.size": 11.0, "axes.titlesize": 11.0, "axes.labelsize": 11.0,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10.5,
    "lines.linewidth": 1.1, "lines.markersize": 4.5,
    "axes.linewidth": 1.3, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5.0, "ytick.major.size": 5.0, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": _FRAMED, "axes.spines.right": _FRAMED,
    "axes.grid": _FRAMED, "grid.linewidth": 0.5, "grid.alpha": 0.30,
    "legend.frameon": False, "axes.axisbelow": True,
    "axes.prop_cycle": matplotlib.cycler(color=_CB_CYCLE),
    "axes.titleweight": "medium", "axes.labelweight": "medium",
}
if os.environ.get("REPORT_USETEX", "1") != "0":
    _rc.update({"text.usetex": True, "font.family": "serif", "text.latex.preamble": r"\usepackage{lmodern}"})
else:
    _rc.update({"text.usetex": False, "font.family": "STIXGeneral", "mathtext.fontset": "stix"})
matplotlib.rcParams.update(_rc)


def savefig(fig, stem):
    """stem.pdf + stem.png ('_airy' appended for the airy variant); returns the pdf path."""
    s = stem + ("" if _FRAMED else "_airy")
    fig.savefig(s + ".pdf"); fig.savefig(s + ".png"); plt.close(fig)
    return s + ".pdf"

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))   # repository root
from paths import ANALYSIS  # noqa: E402
CACHE_DIR = str(ANALYSIS / "cache")
REDET_DIR = str(ANALYSIS / "cache_redetect")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "deploy_nav_collage.png")

def load_run(run):
    """Prefer the re-detected metric cache; merge its world tracks over the base cache."""
    d = dict(np.load(os.path.join(CACHE_DIR, f"{run}.npz"), allow_pickle=True))
    rp = os.path.join(REDET_DIR, f"{run}_redetect.npz")
    if os.path.exists(rp):
        r = np.load(rp, allow_pickle=True)
        d["fish_world"], d["tgt_world"] = r["fish_world"], r["tgt_world"]
        if "frame_time" in r.files:
            d["frame_time"] = r["frame_time"]
    return d

C_START, C_CLOSE, C_TGT = "#1f77b4", "#d62728", "#2ca02c"

PANELS = [
    ("hope_didnt_work_2", "(a) First successful run",           True,  False),
    ("insane_run_8",      "(b) Fluid disturbance",              True,  False),
    ("insane_run_2",      "(c) Backward-velocity start",        True,  False),
    ("finetune_run_7",    "(d) Out-of-distribution (success)",  True,  False),
    ("finetune_run_8",    "(e) Lateral-rear failure",           False, False),
    ("figure_eight_6",    "(f) Figure-eight pursuit",           False, True),
]

def get_scale():
    d = np.load(os.path.join(CACHE_DIR, "insane_run_8.npz"), allow_pickle=True)
    fw, fp = d["fish_world"], d["fish_px"]
    v = ~np.isnan(fw[:, 0]) & ~np.isnan(fp[:, 0])
    return (np.nanmax(fw[v, 0]) - np.nanmin(fw[v, 0])) / (np.nanmax(fp[v, 0]) - np.nanmin(fp[v, 0]))

def to_m(px_xy, scale):
    return np.column_stack([px_xy[:, 0] * scale, -px_xy[:, 1] * scale])   # flip Y (image -> world up)

def despike(fm, vf, thr=0.20):
    """Drop isolated tracking spikes: a valid point far from both neighbours which are themselves close."""
    vi = np.where(vf)[0]
    out = vf.copy()
    for j in range(1, len(vi) - 1):
        i, p, n = vi[j], vi[j - 1], vi[j + 1]
        dp = np.hypot(fm[i, 0] - fm[p, 0], fm[i, 1] - fm[p, 1])
        dn = np.hypot(fm[i, 0] - fm[n, 0], fm[i, 1] - fm[n, 1])
        dpn = np.hypot(fm[p, 0] - fm[n, 0], fm[p, 1] - fm[n, 1])
        if dp > thr and dn > thr and dpn < 0.5 * thr:
            out[i] = False
    return out

def panel(ax, run, title, truncate, moving, scale):
    d = load_run(run)
    fp, tp, ft = d["fish_px"], d["tgt_px"], d["frame_time"]
    fwld, twld = d["fish_world"].astype(float), d["tgt_world"].astype(float)
    world_ok = (np.sum(~np.isnan(fwld[:, 0])) > 0) and (np.sum(~np.isnan(twld[:, 0])) > 0)

    # geometry in metres: accurate metric WORLD where available; px*scale fallback for non-metric runs (fig-8)
    fm, tm = (fwld.copy(), twld.copy()) if world_ok else (to_m(fp, scale), to_m(tp, scale))
    vf = ~np.isnan(fm[:, 0]) & ~np.isnan(fm[:, 1])
    vt = ~np.isnan(tm[:, 0]) & ~np.isnan(tm[:, 1])
    if world_ok:
        vf = despike(fm, vf)                              # drop isolated tracking spikes
    fm[:, 1] *= -1.0                                      # flip Y (match the earlier orientation)
    tm[:, 1] *= -1.0
    g0 = fm[np.where(vf)[0][0]].copy()                    # fish start at origin
    fm = fm - g0
    tm = tm - g0

    # distance fish->target (moving: per-frame; static: to the fixed target)
    if moving:
        both = vf & vt
        dist = np.full(len(fm), np.nan)
        dist[both] = np.hypot(fm[both, 0] - tm[both, 0], fm[both, 1] - tm[both, 1])
    else:
        tref = np.median(tm[vt], axis=0)
        dist = np.where(vf, np.hypot(fm[:, 0] - tref[0], fm[:, 1] - tref[1]), np.nan)
    idx = np.where(vf)[0]
    dvi = np.where(~np.isnan(dist))[0]

    # plotted segment: truncate successes at closest approach; failure/moving show full path
    if truncate:
        ci = int(np.nanargmin(dist))
        end = int(idx[np.argmin(np.abs(idx - ci))]) + 1
    else:
        end = len(fm)
    m = vf.copy(); m[end:] = False
    plotted = np.where(m)[0]
    x, y = fm[m, 0], fm[m, 1]

    # markers use the END of the drawn trajectory (final fish position); note reports start->end distance
    end_i = int(plotted[-1])
    start_dist = float(dist[dvi[0]])
    end_dist = float(dist[end_i])
    t_span = float(ft[plotted[-1]] - ft[plotted[0]])      # duration of the drawn trajectory

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    seglen = np.hypot(segs[:, 1, 0] - segs[:, 0, 0], segs[:, 1, 1] - segs[:, 0, 1])
    ok = seglen < 0.30                                    # break the line across tracking gaps
    carr = np.arange(len(segs))
    lc = LineCollection(segs[ok], cmap="plasma_r", norm=plt.Normalize(0, len(segs)), linewidth=COLLAGE_LW, zorder=3)
    lc.set_array(carr[ok])
    ax.add_collection(lc)

    # direction arrows: evenly spaced by ARC LENGTH, using a look-ahead averaged heading (robust to jitter)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total_len = cum[-1]
    if total_len > 0.15:
        for frac in np.linspace(0.10, 0.92, 6):
            s = frac * total_len
            j = int(np.searchsorted(cum, s))
            j = min(max(j, 1), len(x) - 2)
            la = min(j + max(2, len(x) // 25), len(x) - 1)   # look ahead a few points
            dx, dy = x[la] - x[j], y[la] - y[j]
            if np.hypot(dx, dy) < 1e-3 or seglen[min(j, len(seglen) - 1)] > 0.30:
                continue
            ang = np.arctan2(dy, dx)
            ax.annotate("", xytext=(x[j], y[j]),
                        xy=(x[j] + 0.05 * np.cos(ang), y[j] + 0.05 * np.sin(ang)),
                        arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0,
                                        alpha=0.45, mutation_scale=13), zorder=5)

    if moving:
        mt = vt.copy(); mt[end:] = False
        ax.plot(tm[mt, 0], tm[mt, 1], "--", color=C_TGT, lw=1.3, zorder=2)
        tgt_star = tm[np.where(mt)[0][-1]]                # target at the END of its path
    else:
        tgt_star = np.median(tm[vt], axis=0)

    ax.plot(0, 0, "o", color=C_START, ms=9, mec="none", zorder=6)
    ax.plot(fm[end_i, 0], fm[end_i, 1], "o", color=C_CLOSE, ms=9, mec="none", zorder=7)   # final fish position
    ax.plot(tgt_star[0], tgt_star[1], marker=STAR, color=C_TGT, ms=12, mec="none", linestyle="None", zorder=6)

    ax.axhline(0, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.axvline(0, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.grid(True, which="major", ls="--", lw=0.8, color="0.75", alpha=0.6, zorder=0)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)

    # square panel: equal-range limits centred on the data, origin at fish start
    px_all = np.concatenate([x, [tgt_star[0], 0.0]])
    py_all = np.concatenate([y, [tgt_star[1], 0.0]])
    cx, cy = 0.5 * (px_all.min() + px_all.max()), 0.5 * (py_all.min() + py_all.max())
    half = 0.5 * max(px_all.max() - px_all.min(), py_all.max() - py_all.min()) * 1.09 + 0.05
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MultipleLocator(1.0))     # gridlines every 1 m
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))     # 0.5 m ticks, labelled like majors, no grid
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax.tick_params(axis="both", which="minor", labelsize=plt.rcParams["xtick.labelsize"])  # minor == major, always
    ax.spines["top"].set_visible(_FRAMED); ax.spines["right"].set_visible(_FRAMED)

    note = (f"figure-eight, {t_span:.0f} s" if moving
            else f"{start_dist:.2f} m $\\rightarrow$ {end_dist:.3f} m, {t_span:.0f} s")
    ax.text(0.97, 0.04, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, style="italic", color="0.3")

def main():
    scale = get_scale()
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.8), layout="constrained")
    for ax, p in zip(axes.flat, PANELS):
        try:
            panel(ax, *p, scale)
        except Exception as e:
            ax.text(0.5, 0.5, f"{p[0]}\n{e}", ha="center", va="center", transform=ax.transAxes)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_START, mec="none", ms=9, label="Start position"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CLOSE, mec="none", ms=9, label="Final position"),
        Line2D([0], [0], marker=STAR, color="none", markerfacecolor=C_TGT, mec="none", ms=14, linestyle="None", label="Target"),
        Line2D([0], [0], color=C_TGT, ls="--", lw=1.3, label="Moving target path"),
    ]
    # (baked-in suptitle removed: the LaTeX caption titles the figure)
    # equal inch-padding around every panel (incl. titles/labels) => uniform horizontal & vertical gaps
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.09, wspace=0.0, hspace=0.04)
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    p = savefig(fig, OUT.replace(".png",""))
    print("Saved", p)

if __name__ == "__main__":
    main()
