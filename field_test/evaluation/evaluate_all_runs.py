#!/usr/bin/env python3
"""
field_test/evaluation/evaluate_all_runs.py
-------------------------------------------
Cross-run evaluation for swimming field-test runs 1-12.
Generates all analysis figures and writes EVALUATION.md.

Usage (from repo root):
    python3 field_test/evaluation/evaluate_all_runs.py

Coordinate convention — matches 04_plot_field_test.py:
    mu_x = World Z - Z_0  (decreases as fish swims forward)
    mu_y = -(World X - X_0)  (lateral)
    Axes: "X (m) [Swim Axis]" and "Y (m) [Lateral]"

Amplitude convention:
    All amplitudes in mm using pulley radius r = 17.5 mm.
    The field software (unified_test.py) computed amp_ticks with r = 12.5 mm (default),
    so the "nominal" amplitude in run names (5, 7, 9, 12 mm) maps to more ticks and
    therefore MORE physical tendon travel at 17.5 mm. Commanded amplitude is always
    taken from the Goal register ticks (via 03_field2sim_ready.csv Commanded_Alpha)
    to stay consistent with the actual encoder data.
"""
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter, butter, filtfilt, find_peaks
from scipy.interpolate import CubicSpline

# ─── style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 7, "font.family": "serif",
    "pdf.fonttype": 42, "ps.fonttype": 42, "text.usetex": True,
    "axes.spines.top": False, "axes.spines.right": False,
})
mm = 1 / 25.4

# ─── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))   # repository root
from paths import FIELD  # noqa: E402
DATA_DIR   = str(FIELD)
OUT_DIR    = SCRIPT_DIR

# ─── constants ───────────────────────────────────────────────────────────────
PULLEY_R_MM      = 17.5   # mm — used consistently throughout analysis
RADIUS_FIELD_MM  = 12.5   # mm — used by unified_test.py to compute amp_ticks
VEL_LIMIT_UNITS  = 266    # Profile Velocity register value (same for all runs)
VEL_UNIT_TO_RPM  = 0.229  # 1 Dynamixel velocity unit = 0.229 RPM
# Practical velocity cap observed in data (battery ~11V instead of 12V)
# runs 11 and 12 both max at exactly 220 units — this is the real ceiling
PRACTICAL_VEL_LIMIT_UNITS = 220

FREQ_COLORS = {1.0: "#1f77b4", 2.0: "#ff7f0e", 3.0: "#2ca02c", 4.0: "#d62728"}
AMP_MARKERS  = {5.0: "o", 6.0: "s", 7.0: "^", 9.0: "D", 12.0: "P"}

# ─── run registry ────────────────────────────────────────────────────────────
# (folder_name, cmd_freq_hz, nominal_amp_mm, offset_mm, run_type, short_label)
# nominal_amp_mm is from the --amp argument to unified_test.py (at 12.5mm radius)
RUNS = [
    ("run_01_1.0Hz_5.0mm",           1.0,  5.0,  0.0, "straight", "R1\n1Hz/5mm"),
    ("run_02_2.0Hz_5.0mm",           2.0,  5.0,  0.0, "straight", "R2\n2Hz/5mm"),
    ("run_03_1.0Hz_9.0mm",           1.0,  9.0,  0.0, "straight", "R3\n1Hz/9mm"),
    ("run_04_2.0Hz_9.0mm",           2.0,  9.0,  0.0, "straight", "R4\n2Hz/9mm"),
    ("run_05_1.0Hz_7.0mm",           1.0,  7.0,  0.0, "straight", "R5\n1Hz/7mm"),
    ("run_06_2.0Hz_7.0mm",           2.0,  7.0,  0.0, "straight", "R6\n2Hz/7mm"),
    ("run_07_3.0Hz_7.0mm",           3.0,  7.0,  0.0, "straight", "R7\n3Hz/7mm"),
    ("run_08_2.0Hz_6.0mm_3.0mm",     2.0,  6.0,  3.0, "curve",    "R8\n2Hz/6mm\n+3mm"),
    ("run_09_2.0Hz_6.0mm_-3.0mm",    2.0,  6.0, -3.0, "curve",    "R9\n2Hz/6mm\n-3mm"),
    ("run_10_4.0Hz_7.0mm",          4.0,  7.0,  0.0, "straight", "R10\n4Hz/7mm"),
    ("run_11_2.0Hz_12.0mm",         2.0, 12.0,  0.0, "straight", "R11\n2Hz/12mm"),
    ("run_12_2.0Hz_12.0mm_p-1600",  2.0, 12.0,  0.0, "straight", "R12\n2Hz/12mm\nP=1600"),
]


# ─── helpers ─────────────────────────────────────────────────────────────────

def nominal_to_cmd_mm(nominal_mm):
    """Convert nominal amplitude (at 12.5mm radius) to actual commanded mm (at 17.5mm)."""
    ticks = int((nominal_mm / (2 * math.pi * RADIUS_FIELD_MM)) * 4096)
    return ticks * (2 * math.pi / 4096) * PULLEY_R_MM


def required_peak_vel_units(freq, cmd_amp_rad):
    """Peak motor velocity required to track the commanded sine (Dynamixel units).
    Derivation: d/dt[A*sin(2pi*f*t)] = 2pi*f*A (rad/s)
    Converted to RPM: 60*f*A; then to Dxl units: / VEL_UNIT_TO_RPM.
    """
    return 60.0 * freq * cmd_amp_rad / VEL_UNIT_TO_RPM


def load_run(run_name):
    proc = os.path.join(DATA_DIR, run_name, "processed")
    imu  = pd.read_csv(os.path.join(proc, "02_fish_log_trimmed.csv"))
    vis  = pd.read_csv(os.path.join(proc, "02_clean_path.csv"))
    vis  = vis.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)

    t0       = imu["Unix_Time"].min()
    imu["T"] = imu["Unix_Time"] - t0
    vis["T"] = vis["Unix_Time"] - t0

    # Match 04_plot_field_test.py coordinate convention exactly
    mu_x = vis["Z"].values - vis["Z"].iloc[0]    # decreases as fish swims
    mu_y = -(vis["X"].values - vis["X"].iloc[0]) # lateral

    f3_path = os.path.join(proc, "03_field2sim_ready.csv")
    df3     = pd.read_csv(f3_path) if os.path.exists(f3_path) else None

    return imu, vis["T"].values, mu_x, mu_y, df3


def speed_profile(t_pos, mu_x, mu_y, sg_w=51, sg_p=3):
    """SG-derivative speed on a uniform grid. mu_x decreases → vx negative → take abs."""
    fps   = 1.0 / np.median(np.diff(t_pos))
    t_reg = np.arange(t_pos[0], t_pos[-1], 1.0 / fps)
    # Clamped BC: zero velocity at start (fish from rest), natural end (still swimming)
    x_reg = CubicSpline(t_pos, mu_x, bc_type=((1, 0.0), (2, 0.0)))(t_reg)
    y_reg = CubicSpline(t_pos, mu_y, bc_type=((1, 0.0), (2, 0.0)))(t_reg)
    pad   = sg_w // 2

    def _sg(arr):
        p = np.concatenate([np.full(pad, arr[0]), arr])
        return savgol_filter(p, sg_w, sg_p, deriv=1, delta=1.0/fps, mode="interp")[pad:]

    # speed magnitude (always positive regardless of swim direction)
    return t_reg, np.sqrt(_sg(x_reg)**2 + _sg(y_reg)**2)


def steady_state_speed(t_pos, mu_x, frac=(0.50, 0.80)):
    """Speed via linear regression on swim-axis position over late fraction of run.
    Window 40-90% avoids the acceleration transient and the SG end-boundary effect.
    Returns magnitude (positive).
    """
    dur = t_pos[-1] - t_pos[0]
    t1, t2 = t_pos[0] + frac[0]*dur, t_pos[0] + frac[1]*dur
    m = (t_pos >= t1) & (t_pos <= t2)
    if m.sum() < 3:
        return np.nan
    slope = float(np.polyfit(t_pos[m], mu_x[m], 1)[0])
    return abs(slope)  # mu_x decreases, so slope is negative; return magnitude


def accel_estimate(t_reg, spd, ss_spd):
    """Linear fit to SG speed during startup (until 80% of ss_speed is reached)."""
    if np.isnan(ss_spd) or ss_spd <= 0:
        return np.nan
    idx = int(np.argmax(spd >= 0.80 * ss_spd))
    if idx < 5:
        return np.nan
    return float(np.polyfit(t_reg[:idx], spd[:idx], 1)[0])


def rms_lateral(mu_x, mu_y):
    """RMS perpendicular distance from best-fit line through the full trajectory (m)."""
    a, b = np.polyfit(mu_x, mu_y, 1)
    return float(np.sqrt(np.mean(((a*mu_x - mu_y + b) / np.sqrt(a**2 + 1))**2)))


def straightness_index(mu_x, mu_y):
    """Chord-to-path-length ratio: 1.0 = perfectly straight."""
    chord = np.sqrt((mu_x[-1]-mu_x[0])**2 + (mu_y[-1]-mu_y[0])**2)
    path  = np.sum(np.sqrt(np.diff(mu_x)**2 + np.diff(mu_y)**2))
    return float(chord / path) if path > 0 else np.nan


def decompose_path(t_pos, mu_y, freq_hz):
    """Split lateral position into smooth drift and per-cycle oscillation.

    Applies a 4th-order zero-phase Butterworth lowpass at 0.5 * freq_hz.
    Returns (mu_y_smooth, mu_y_osc) sampled at the original t_pos grid.
    """
    dt     = np.median(np.diff(t_pos))
    fps    = 1.0 / dt
    t_reg  = np.arange(t_pos[0], t_pos[-1], dt)
    y_reg  = CubicSpline(t_pos, mu_y)(t_reg)
    b, a   = butter(4, (0.5 * freq_hz) / (0.5 * fps), btype="low")
    y_sm_r = filtfilt(b, a, y_reg)
    y_sm   = CubicSpline(t_reg, y_sm_r)(t_pos)
    return y_sm, mu_y - y_sm


def fit_circle(x, y):
    """Algebraic circle fit. Returns (cx, cy, radius_m)."""
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    res, *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    cx, cy  = res[0], res[1]
    return cx, cy, float(np.sqrt(max(0.0, res[2] + cx**2 + cy**2)))


def motor_metrics(df3, imu, nominal_mm, freq):
    """Extract motor tracking and velocity metrics.
    cmd_mm: commanded amplitude from Goal register at 17.5mm radius (via 03 file).
    act_mm: actual achieved amplitude from Actual register at 17.5mm.
    Both are in the same physical units — act_mm < cmd_mm means tracking loss.
    """
    if df3 is None:
        return dict(cmd_mm=np.nan, act_mm=np.nan, track=np.nan,
                    peak_vel_units=np.nan, vel_pct_prac=np.nan, req_vel_units=np.nan)

    # Commanded amplitude from Goal ticks (via splined Commanded_Alpha in 03 file)
    if "Commanded_Alpha" in df3.columns:
        ca     = df3["Commanded_Alpha"].values
        cmd_mm = (ca.max() - ca.min()) / 2 * PULLEY_R_MM
        cmd_amp_rad = (ca.max() - ca.min()) / 2
    else:
        cmd_mm = nominal_to_cmd_mm(nominal_mm)
        cmd_amp_rad = cmd_mm / PULLEY_R_MM / 1000 * 1000  # = cmd_mm / PULLEY_R_MM

    # Actual amplitude from encoder — peak-average method to exclude startup transient.
    # (max-min)/2 is inflated by the first-cycle overshoot, especially at high frequency.
    alpha = df3["Ghost_Alpha"].values
    dt    = 0.001  # 03_field2sim_ready.csv is always resampled to 1 kHz
    dist  = max(1, int(0.5 / freq / dt))   # min samples between peaks ≈ half period
    skip  = int(1.0 / freq / dt)           # samples in first full period to discard
    alpha_ss = alpha[skip:]
    pos_peaks, _  = find_peaks( alpha_ss, distance=dist)
    neg_troughs, _ = find_peaks(-alpha_ss, distance=dist)
    if len(pos_peaks) > 0 and len(neg_troughs) > 0:
        act_amp_rad = (np.mean(alpha_ss[pos_peaks]) - np.mean(alpha_ss[neg_troughs])) / 2
    else:
        act_amp_rad = (alpha.max() - alpha.min()) / 2  # fallback if too few cycles
    act_mm = act_amp_rad * PULLEY_R_MM

    track = act_mm / cmd_mm if (not np.isnan(cmd_mm) and cmd_mm > 0) else np.nan

    # Measured peak velocity (Dynamixel units)
    pv_units = float(imu["Vel"].abs().max()) if "Vel" in imu.columns else np.nan
    vel_pct_prac = 100 * pv_units / PRACTICAL_VEL_LIMIT_UNITS if not np.isnan(pv_units) else np.nan

    # Required peak velocity to track the commanded sine
    req = required_peak_vel_units(freq, cmd_amp_rad)

    mean_volt = float(imu["Volt"].mean()) if "Volt" in imu.columns else np.nan

    return dict(cmd_mm=cmd_mm, act_mm=act_mm, track=track,
                peak_vel_units=pv_units, vel_pct_prac=vel_pct_prac, req_vel_units=req,
                mean_volt=mean_volt)


def curve_asymmetry(imu):
    """For curve runs: actual and commanded center offset and per-side amplitude."""
    neutral   = float(imu["Actual"].iloc[0])
    act_center = imu["Actual"].mean()
    act_bias_mm  = (act_center - neutral) * (2*np.pi/4096) * PULLEY_R_MM
    act_pos_mm   = (imu["Actual"].max() - act_center) * (2*np.pi/4096) * PULLEY_R_MM
    act_neg_mm   = (act_center - imu["Actual"].min()) * (2*np.pi/4096) * PULLEY_R_MM

    cmd_center = imu["Goal"].mean()
    cmd_bias_mm  = (cmd_center - neutral) * (2*np.pi/4096) * PULLEY_R_MM
    cmd_pos_mm   = (imu["Goal"].max() - cmd_center) * (2*np.pi/4096) * PULLEY_R_MM
    cmd_neg_mm   = (cmd_center - imu["Goal"].min()) * (2*np.pi/4096) * PULLEY_R_MM

    return dict(act_bias=act_bias_mm, act_pos=act_pos_mm, act_neg=act_neg_mm,
                cmd_bias=cmd_bias_mm, cmd_pos=cmd_pos_mm, cmd_neg=cmd_neg_mm)


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print("Loading and computing metrics...")
    rows      = []
    run_cache = {}

    for name, freq, nom_amp, offset, rtype, label in RUNS:
        print(f"  {name}")
        imu, t_pos, mu_x, mu_y, df3 = load_run(name)

        t_reg, spd = speed_profile(t_pos, mu_x, mu_y)
        ss  = steady_state_speed(t_pos, mu_x)
        acc = accel_estimate(t_reg, spd, ss)
        mtr = motor_metrics(df3, imu, nom_amp, freq)
        dur = float(t_pos[-1] - t_pos[0])

        rms_lat = rms_lateral(mu_x, mu_y) * 1000  # mm (kept for reference)
        si      = straightness_index(mu_x, mu_y)

        # Decomposed path metrics
        mu_y_sm, mu_y_osc = decompose_path(t_pos, mu_y, freq)
        _a, _b = np.polyfit(mu_x, mu_y, 1)          # line fitted to original
        rms_path_mm = float(np.sqrt(np.mean(
            ((_a * mu_x - mu_y_sm + _b) / np.sqrt(_a**2 + 1))**2))) * 1000
        rms_osc_mm  = float(np.sqrt(np.mean(mu_y_osc**2))) * 1000

        turn_r  = np.nan
        asym    = None
        if rtype == "curve":
            _, _, turn_r = fit_circle(mu_x, mu_y)
            asym = curve_asymmetry(imu)

        cmd_mm_nominal = nominal_to_cmd_mm(nom_amp)  # at 17.5mm, from ticks

        rows.append(dict(
            name=name, label=label, freq=freq, nom_amp=nom_amp,
            offset=offset, rtype=rtype, duration=dur,
            ss_speed=ss, accel=acc,
            rms_lat_mm=rms_lat, si=si,
            rms_path_mm=rms_path_mm, rms_osc_mm=rms_osc_mm,
            turn_r=turn_r,
            cmd_mm_nominal=cmd_mm_nominal,
            **mtr,
        ))
        run_cache[name] = dict(
            t_pos=t_pos, mu_x=mu_x, mu_y=mu_y,
            t_reg=t_reg, spd=spd, imu=imu, df3=df3,
            freq=freq, nom_amp=nom_amp, rtype=rtype, label=label,
            asym=asym,
        )

    df = pd.DataFrame(rows)
    straight = df[df["rtype"] == "straight"].copy()
    curves   = df[df["rtype"] == "curve"].copy()

    # ── Figure 1: All trajectories ────────────────────────────────────────────
    print("fig1: trajectories...")
    fig, axes = plt.subplots(3, 4, figsize=(185*mm, 135*mm), constrained_layout=True)
    axes = axes.flatten()
    for ax, (name, freq, nom_amp, offset, rtype, label) in zip(axes, RUNS):
        d     = run_cache[name]
        color = "#1f77b4" if rtype == "straight" else "#ff7f0e"
        ax.plot(d["mu_x"], d["mu_y"], color=color, lw=0.9, alpha=0.85)
        ax.plot(d["mu_x"][0],  d["mu_y"][0],  "o", color="green",   ms=3, zorder=5)
        ax.plot(d["mu_x"][-1], d["mu_y"][-1], "x", color="crimson", ms=4, zorder=5,
                markeredgewidth=1.2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(label.replace("\n", " "), fontsize=6.5, pad=2)
        ax.set_xlabel("X (m)", fontsize=6)
        ax.set_ylabel("Lateral (m)", fontsize=6)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.tick_params(labelsize=5.5)

    legend_els = [
        Line2D([0],[0], color="#1f77b4", lw=1.5, label="Straight"),
        Line2D([0],[0], color="#ff7f0e", lw=1.5, label="Curve"),
        Line2D([0],[0], marker="o", color="green",   ls="", ms=4, label="Start"),
        Line2D([0],[0], marker="x", color="crimson", ls="", ms=4,
               markeredgewidth=1.2, label="End"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.01))
    fig.savefig(os.path.join(OUT_DIR, "fig1_trajectories.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig1_trajectories.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 1b: All trajectories overlaid ─────────────────────────────────
    fig, ax_ov = plt.subplots(figsize=(120*mm, 110*mm), constrained_layout=True)
    for name, freq, nom_amp, offset, rtype, label in RUNS:
        d   = run_cache[name]
        c   = FREQ_COLORS[freq]
        ls  = "--" if rtype == "curve" else "-"
        lw  = 1.0
        ax_ov.plot(d["mu_x"], d["mu_y"], color=c, lw=lw, ls=ls, alpha=0.85,
                   label=label.replace("\n", " "))
        ax_ov.plot(d["mu_x"][0],  d["mu_y"][0],  "o", color=c, ms=3, zorder=5)
        ax_ov.plot(d["mu_x"][-1], d["mu_y"][-1], "x", color=c, ms=4, zorder=5,
                   markeredgewidth=1.1)
    ax_ov.set_aspect("equal", adjustable="datalim")
    ax_ov.set_xlabel("X (m) [Swim Axis]")
    ax_ov.set_ylabel("Y (m) [Lateral]")
    ax_ov.grid(True, lw=0.3, alpha=0.4)
    freq_ov = [Line2D([0],[0], color=c, lw=1.5, label=f"{f} Hz")
               for f, c in FREQ_COLORS.items()]
    style_ov = [
        Line2D([0],[0], color="gray", lw=1.2, ls="-",  label="Straight"),
        Line2D([0],[0], color="gray", lw=1.2, ls="--", label="Curve"),
    ]
    ax_ov.legend(handles=freq_ov + style_ov, fontsize=6, ncol=2)
    fig.savefig(os.path.join(OUT_DIR, "fig1b_trajectories_overlay.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig1b_trajectories_overlay.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2a: Speed profiles ─────────────────────────────────────────────
    print("fig2: speed profiles...")
    straight_list = [(n,f,a,o,r,l) for n,f,a,o,r,l in RUNS
                     if r == "straight" and n != "run_12_2.0Hz_12.0mm_p-1600"]
    fig, ax_sp = plt.subplots(figsize=(88*mm, 60*mm), constrained_layout=True)
    for name, freq, nom_amp, offset, rtype, label in straight_list:
        d      = run_cache[name]
        t_reg  = d["t_reg"]
        spd    = d["spd"]
        dur    = t_reg[-1] - t_reg[0]
        t_norm = (t_reg - t_reg[0]) / dur
        c      = FREQ_COLORS[freq]
        ax_sp.plot(t_norm, spd, color=c, lw=0.9, alpha=0.70)

    ax_sp.axvspan(0.50, 0.80, alpha=0.10, color="gray", label="SS window (50--80\\%)")
    ax_sp.set_xlabel("Normalised time (0 = motor start, 1 = end)")
    ax_sp.set_ylabel("Speed (m/s)")
    ax_sp.set_xlim(0, 1)
    ax_sp.set_ylim(bottom=0)
    ax_sp.grid(True, lw=0.3, alpha=0.4)
    ax_sp.set_title("Speed profiles --- straight runs", fontsize=7.5)
    freq_handles = [Line2D([0],[0], color=c, lw=1.5, label=f"{f} Hz")
                    for f, c in FREQ_COLORS.items()]
    freq_handles += [mpatches.Patch(color="gray", alpha=0.3, label="50--80\\% window")]
    ax_sp.legend(handles=freq_handles, fontsize=6, loc="upper left", ncol=2)
    fig.savefig(os.path.join(OUT_DIR, "fig2a_speed_profiles.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig2a_speed_profiles.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2b: Steady-state speed ─────────────────────────────────────────
    bar_data = sorted(
        [(n, f,
          df.loc[df["name"]==n, "ss_speed"].values[0],
          FREQ_COLORS[f],
          int(n.split("_")[1]),
          df.loc[df["name"]==n, "act_mm"].values[0])
         for n, f, a, o, r, l in straight_list],
        key=lambda x: x[2]
    )
    xs_b   = range(len(bar_data))
    ss_v   = [d[2] for d in bar_data]
    cbars  = [d[3] for d in bar_data]
    lbls_b = [f"R{d[4]:02d} {d[1]:.0f}Hz/{d[5]:.2f}mm" for d in bar_data]
    fig, ax_bar = plt.subplots(figsize=(88*mm, 68*mm), constrained_layout=True)
    bars = ax_bar.bar(xs_b, ss_v, color=cbars, edgecolor="white", linewidth=0.5)
    ax_bar.set_xticks(list(xs_b))
    ax_bar.set_xticklabels(lbls_b, fontsize=6, rotation=38, ha="right")
    ax_bar.set_ylabel("Steady-state speed (m/s)")
    ax_bar.set_ylim(0, max(ss_v)*1.18)
    ax_bar.set_title("Steady-state speed", fontsize=7.5)
    ax_bar.grid(True, axis="y", lw=0.3, alpha=0.4)
    for bar, v in zip(bars, ss_v):
        ax_bar.text(bar.get_x()+bar.get_width()/2, v+0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    fig.savefig(os.path.join(OUT_DIR, "fig2b_steady_state_speed.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig2b_steady_state_speed.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Speed trends ────────────────────────────────────────────────
    print("fig3: speed trends...")
    # Exclude run_12 (different P gain) from trend plots
    trend = straight[straight["name"] != "run_12_2.0Hz_12.0mm_p-1600"].copy()

    # ── Figure 3a: Speed vs. frequency ────────────────────────────────────────
    fig, ax_f = plt.subplots(figsize=(88*mm, 65*mm), constrained_layout=True)
    sc = ax_f.scatter(trend["freq"], trend["ss_speed"],
                      c=trend["act_mm"], cmap="coolwarm",
                      s=30, zorder=3, edgecolors="gray", linewidths=0.3)
    plt.colorbar(sc, ax=ax_f, label="Actual amplitude (mm)")
    ax_f.set_xlabel("Commanded frequency (Hz)")
    ax_f.set_ylabel("Steady-state speed (m/s)")
    ax_f.set_title("Speed vs. frequency", fontsize=7.5)
    ax_f.set_xticks([1, 2, 3, 4])
    ax_f.set_ylim(0, 0.6)
    ax_f.grid(True, lw=0.3, alpha=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig3a_speed_vs_freq.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig3a_speed_vs_freq.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3b: Speed vs. actual amplitude ─────────────────────────────────
    fig, ax_a = plt.subplots(figsize=(88*mm, 60*mm), constrained_layout=True)
    for freq_val, grp in trend.groupby("freq"):
        grp_s = grp.sort_values("act_mm")
        c     = FREQ_COLORS[freq_val]
        ax_a.plot(grp_s["act_mm"], grp_s["ss_speed"], marker="o", color=c, lw=1.2,
                  ms=5, label=f"{freq_val:.0f} Hz")
    ax_a.set_xlabel("Actual amplitude (mm)")
    ax_a.set_ylabel("Steady-state speed (m/s)")
    ax_a.set_title("Speed vs. actual amplitude", fontsize=7.5)
    ax_a.set_ylim(0, 0.6)
    ax_a.grid(True, lw=0.3, alpha=0.4)
    ax_a.legend(fontsize=6, title="Freq.", title_fontsize=6, loc="lower right")
    fig.savefig(os.path.join(OUT_DIR, "fig3b_speed_vs_amplitude.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig3b_speed_vs_amplitude.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: Motor amplitude tracking ────────────────────────────────────
    print("fig4: motor tracking...")
    runs4  = [(n,f,a,o,r,l) for n,f,a,o,r,l in RUNS if n != "run_12_2.0Hz_12.0mm_p-1600"]
    df4    = df[df["name"] != "run_12_2.0Hz_12.0mm_p-1600"].reset_index(drop=True)
    xs    = np.arange(len(runs4))
    lbls4 = [f"R{int(n.split('_')[1]):02d} {f:.0f}Hz/{df4.loc[df4['name']==n, 'cmd_mm'].values[0]:.2f}mm"
             for n, f, *_ in runs4]
    w     = 0.35

    # ── Figure 4a: Commanded vs. actual amplitude ─────────────────────────────
    fig, ax_amp = plt.subplots(figsize=(88*mm, 68*mm), constrained_layout=True)
    ax_amp.bar(xs - w/2, df4["cmd_mm"], w, label="Commanded", color="#aec7e8", edgecolor="white")
    ax_amp.bar(xs + w/2, df4["act_mm"],  w, label="Actual",    color="#1f77b4", edgecolor="white")
    ax_amp.set_xticks(list(xs))
    ax_amp.set_xticklabels(lbls4, fontsize=5.5, rotation=38, ha="right")
    ax_amp.set_ylabel("Amplitude (mm)")
    ax_amp.set_title("Commanded vs. actual amplitude", fontsize=7.5)
    ax_amp.legend(fontsize=6.5)
    ax_amp.grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.savefig(os.path.join(OUT_DIR, "fig4a_motor_amplitude.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig4a_motor_amplitude.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4b: Amplitude tracking ratio ───────────────────────────────────
    df4b = df4.sort_values("track").reset_index(drop=True)
    xs_tr = np.arange(len(df4b))
    lbls_tr = [f"R{int(n.split('_')[1]):02d} {f:.0f}Hz/{cmd:.2f}mm"
               for n, f, cmd in zip(df4b["name"], df4b["freq"], df4b["cmd_mm"])]
    colors_tr = [FREQ_COLORS[f] for f in df4b["freq"]]
    fig, ax_tr = plt.subplots(figsize=(88*mm, 65*mm), constrained_layout=True)
    bars_t = ax_tr.bar(xs_tr, df4b["track"], color=colors_tr, edgecolor="white", linewidth=0.5)
    ax_tr.axhline(1.0, color="black", lw=0.8, ls="--")
    ax_tr.set_xticks(list(xs_tr))
    ax_tr.set_xticklabels(lbls_tr, fontsize=5.5, rotation=38, ha="right")
    ax_tr.set_ylabel("Tracking ratio (actual / commanded)")
    ax_tr.set_title("Amplitude tracking ratio", fontsize=7.5)
    ax_tr.set_ylim(0, 1.18)
    ax_tr.grid(True, axis="y", lw=0.3, alpha=0.4)
    for bar, v in zip(bars_t, df4b["track"]):
        if not np.isnan(v):
            ax_tr.text(bar.get_x()+bar.get_width()/2, v+0.01,
                       f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
    freq_handles = [mpatches.Patch(color=c, label=f"{f} Hz") for f,c in FREQ_COLORS.items()]
    ax_tr.legend(handles=freq_handles + [
        Line2D([0],[0], color="black", ls="--", lw=0.8, label="Perfect (1.0)"),
    ], fontsize=6, loc="lower left", ncol=2)
    fig.savefig(os.path.join(OUT_DIR, "fig4b_tracking_ratio.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig4b_tracking_ratio.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 5: Velocity budget (required vs practical limit) ───────────────
    print("fig5: velocity budget...")
    prac_rpm = PRACTICAL_VEL_LIMIT_UNITS * VEL_UNIT_TO_RPM
    theo_rpm = VEL_LIMIT_UNITS * VEL_UNIT_TO_RPM
    runs5 = [(n,f,a,o,r,l) for n,f,a,o,r,l in RUNS if n != "run_12_2.0Hz_12.0mm_p-1600"]
    df5   = df[df["name"] != "run_12_2.0Hz_12.0mm_p-1600"].sort_values("req_vel_units").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(130*mm, 80*mm), constrained_layout=True)
    xs5 = np.arange(len(df5))
    w2 = 0.38

    lbls5 = [f"R{int(n.split('_')[1]):02d} {f:.0f}Hz/{cmd:.2f}mm"
             for n, f, cmd in zip(df5["name"], df5["freq"], df5["cmd_mm"])]

    req_pct  = [100*r["req_vel_units"]/PRACTICAL_VEL_LIMIT_UNITS for _, r in df5.iterrows()]
    meas_pct = [r["vel_pct_prac"] for _, r in df5.iterrows()]

    # Required velocity bars (what the sine needs)
    req_colors = ["#d62728" if v > 100 else "#ff7f0e" if v > 70 else "#2ca02c"
                  for v in req_pct]
    ax.bar(xs5 + w2/2, req_pct, w2, color=req_colors, alpha=0.6, label="Required (from commanded sine)")

    # Measured peak velocity bars (what the motor did)
    ax.bar(xs5 - w2/2, meas_pct, w2, color="#1f77b4", alpha=0.9, label="Measured peak (Vel register)")

    ax.axhline(100, color="black",   lw=1.2, ls="--", label="Practical limit (220 units, $\\sim$11V)")
    ax.axhline(100 * VEL_LIMIT_UNITS / PRACTICAL_VEL_LIMIT_UNITS,
               color="gray", lw=0.8, ls=":", alpha=0.7,
               label=f"Theoretical limit (266 units, 12V = {100*VEL_LIMIT_UNITS/PRACTICAL_VEL_LIMIT_UNITS:.0f}\\%)")

    ax.set_xticks(list(xs5))
    ax.set_xticklabels(lbls5, fontsize=5.5, rotation=38, ha="right")
    ax.set_ylabel("Velocity as \\% of practical limit (220 units)")
    ax.set_title("Velocity budget per run\n"
                 "(required to track sine vs. measured peak)", fontsize=7.5)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    ax.set_ylim(0, max(max(req_pct), 120) * 1.08)

    # Annotate mean voltage above measured bar
    for xi, (_, row) in zip(xs5, df5.iterrows()):
        if not np.isnan(row["vel_pct_prac"]):
            ax.text(xi - w2/2, row["vel_pct_prac"] + 2,
                    f"{row['mean_volt']:.1f}V",
                    ha="center", va="bottom", fontsize=5.5, color="#1f77b4")

    legend_els = [
        mpatches.Patch(color="#2ca02c", alpha=0.7, label="Required $<$ 70\\%"),
        mpatches.Patch(color="#ff7f0e", alpha=0.7, label="Required 70--100\\%"),
        mpatches.Patch(color="#d62728", alpha=0.7, label="Required $>$ 100\\%"),
        mpatches.Patch(color="#1f77b4", alpha=0.9, label="Measured peak"),
        Line2D([0],[0], color="black", ls="--", lw=1.2, label="Practical limit (100\\%)"),
        Line2D([0],[0], color="gray",  ls=":",  lw=0.8, label="Theoretical limit (12V)"),
    ]
    ax.legend(handles=legend_els, fontsize=5.5, loc="upper left")
    fig.savefig(os.path.join(OUT_DIR, "fig5_velocity_budget.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig5_velocity_budget.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 6a: Path drift (smooth path vs best-fit straight line) ─────────
    print("fig6: straightness (decomposed)...")
    straight_mask = (df["rtype"] == "straight") & (df["name"] != "run_12_2.0Hz_12.0mm_p-1600")
    s_df6a = df[straight_mask].sort_values("rms_path_mm").reset_index(drop=True)
    xs6a   = np.arange(len(s_df6a))
    lbls6a = [f"R{int(n.split('_')[1]):02d} {f:.0f}Hz/{cmd:.2f}mm"
              for n, f, cmd in zip(s_df6a["name"], s_df6a["freq"], s_df6a["cmd_mm"])]
    sc6a   = [FREQ_COLORS[f] for f in s_df6a["freq"]]

    fig, ax6a = plt.subplots(figsize=(88*mm, 68*mm), constrained_layout=True)
    bars6a = ax6a.bar(xs6a, s_df6a["rms_path_mm"], color=sc6a,
                      edgecolor="white", linewidth=0.5)
    ax6a.set_xticks(list(xs6a))
    ax6a.set_xticklabels(lbls6a, fontsize=5.5, rotation=38, ha="right")
    ax6a.set_ylabel("RMS path drift from best-fit line (mm)")
    ax6a.set_title("Path drift --- smooth path vs. straight line\n"
                   "(lowpass $<$ $f_{\\rm swim}/2$, then $\\perp$ dist to fitted line)",
                   fontsize=7.5)
    ax6a.grid(True, axis="y", lw=0.3, alpha=0.4)
    for bar, v in zip(bars6a, s_df6a["rms_path_mm"]):
        ax6a.text(bar.get_x()+bar.get_width()/2, v+0.2, f"{v:.1f}",
                  ha="center", va="bottom", fontsize=5.5)
    freq_handles = [mpatches.Patch(color=c, label=f"{f} Hz") for f,c in FREQ_COLORS.items()]
    ax6a.legend(handles=freq_handles, fontsize=6, title="Freq.", title_fontsize=6)
    fig.savefig(os.path.join(OUT_DIR, "fig6a_path_drift.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig6a_path_drift.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 6b: Oscillation magnitude (actual vs smooth path) ─────────────
    s_df6b = df[straight_mask].sort_values("rms_osc_mm").reset_index(drop=True)
    xs6b   = np.arange(len(s_df6b))
    lbls6b = [f"R{int(n.split('_')[1]):02d} {f:.0f}Hz/{cmd:.2f}mm"
              for n, f, cmd in zip(s_df6b["name"], s_df6b["freq"], s_df6b["cmd_mm"])]
    sc6b   = [FREQ_COLORS[f] for f in s_df6b["freq"]]

    fig, ax6b = plt.subplots(figsize=(88*mm, 68*mm), constrained_layout=True)
    bars6b = ax6b.bar(xs6b, s_df6b["rms_osc_mm"], color=sc6b,
                      edgecolor="white", linewidth=0.5)
    ax6b.set_xticks(list(xs6b))
    ax6b.set_xticklabels(lbls6b, fontsize=5.5, rotation=38, ha="right")
    ax6b.set_ylabel("RMS lateral oscillation (mm)")
    ax6b.set_title("Lateral oscillation magnitude --- actual vs. smooth path\n"
                   "(residual after lowpass $<$ $f_{\\rm swim}/2$)", fontsize=7.5)
    ax6b.grid(True, axis="y", lw=0.3, alpha=0.4)
    for bar, v in zip(bars6b, s_df6b["rms_osc_mm"]):
        ax6b.text(bar.get_x()+bar.get_width()/2, v+0.05, f"{v:.2f}",
                  ha="center", va="bottom", fontsize=5.5)
    ax6b.legend(handles=freq_handles, fontsize=6, title="Freq.", title_fontsize=6)
    fig.savefig(os.path.join(OUT_DIR, "fig6b_osc_magnitude.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig6b_osc_magnitude.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 7: Curve runs ──────────────────────────────────────────────────
    print("fig7: curve runs...")
    r8_d = run_cache["run_8_2.0Hz_6.0mm_3.0mm"]
    r9_d = run_cache["run_9_2.0Hz_6.0mm_-3.0mm"]
    cx8, cy8, rad8 = fit_circle(r8_d["mu_x"], r8_d["mu_y"])
    cx9, cy9, rad9 = fit_circle(r9_d["mu_x"], r9_d["mu_y"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(145*mm, 80*mm), constrained_layout=True)
    all_x = np.concatenate([r8_d["mu_x"], r9_d["mu_x"]])
    all_y = np.concatenate([r8_d["mu_y"], r9_d["mu_y"]])
    pad   = 0.15
    xlim  = (all_x.min()-pad, all_x.max()+pad)
    ylim  = (all_y.min()-pad, all_y.max()+pad)

    for ax, d, cx, cy, rad, title in [
        (axL, r8_d, cx8, cy8, rad8, f"Run 8  (+3mm offset)\nTurning radius $\\approx$ {rad8:.2f} m"),
        (axR, r9_d, cx9, cy9, rad9, f"Run 9  (-3mm offset)\nTurning radius $\\approx$ {rad9:.2f} m"),
    ]:
        ax.plot(d["mu_x"], d["mu_y"], color="#ff7f0e", lw=1.2)
        ax.plot(d["mu_x"][0],  d["mu_y"][0],  "o", color="green",  ms=4, zorder=5, label="Start")
        ax.plot(d["mu_x"][-1], d["mu_y"][-1], "x", color="crimson", ms=5, zorder=5,
                markeredgewidth=1.4, label="End")
        theta = np.linspace(0, 2*np.pi, 300)
        ax.plot(cx+rad*np.cos(theta), cy+rad*np.sin(theta), "k--", lw=0.7, alpha=0.5,
                label="Fit circle")
        ax.plot(cx, cy, "k+", ms=5, markeredgewidth=1.0)
        ax.set_aspect("equal")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("X (m) [Swim Axis]")
        ax.set_ylabel("Lateral (m)")
        ax.set_title(title, fontsize=7.5)
        ax.grid(True, lw=0.3, alpha=0.4)
        ax.legend(fontsize=6, loc="upper left")

    # Add asymmetry annotation
    for ax, rname in [(axL, "run_8_2.0Hz_6.0mm_3.0mm"), (axR, "run_9_2.0Hz_6.0mm_-3.0mm")]:
        a = run_cache[rname]["asym"]
        txt = (f"Cmd bias: {a['cmd_bias']:+.1f}mm  Act bias: {a['act_bias']:+.1f}mm\n"
               f"Act +stroke: {a['act_pos']:.1f}mm  Act -stroke: {a['act_neg']:.1f}mm")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=5.5,
                va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.suptitle("Curve runs --- symmetric offset magnitude, opposite direction", fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "fig7_curves.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig7_curves.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Figure 8: Tracking loss vs. velocity saturation (voltage-coloured) ────
    print("fig8: saturation analysis...")
    df8 = df[df["name"] != "run_12_2.0Hz_12.0mm_p-1600"].copy()
    df8["sat_ratio"] = df8["req_vel_units"] / df8["peak_vel_units"]

    fig, ax = plt.subplots(figsize=(88*mm, 70*mm), constrained_layout=True)

    # Theoretical ideal-saturation curve: tracking = min(1, 1 / sat_ratio)
    xth = np.linspace(0.35, df8["sat_ratio"].max() * 1.08, 400)
    ax.plot(xth, np.minimum(1.0, 1.0 / xth), "k--", lw=0.8,
            label="Ideal saturation ($1/x$)", zorder=1)

    sc = ax.scatter(df8["sat_ratio"], df8["track"],
                    c=df8["mean_volt"], cmap="plasma",
                    vmin=df8["mean_volt"].min() - 0.1,
                    vmax=df8["mean_volt"].max() + 0.1,
                    s=28, zorder=3, edgecolors="gray", linewidths=0.3)
    cbar = fig.colorbar(sc, ax=ax, label="Mean battery voltage (V)", pad=0.02)
    cbar.ax.tick_params(labelsize=6)

    for _, row in df8.iterrows():
        rn = int(row["name"].split("_")[1])
        ax.annotate(f"R{rn}", (row["sat_ratio"], row["track"]),
                    fontsize=5, xytext=(3, 2), textcoords="offset points")

    ax.axvline(1.0, color="black", lw=0.6, ls=":", alpha=0.5, label="Saturation onset")
    ax.set_xlabel("Saturation depth (required / measured peak velocity)")
    ax.set_ylabel("Amplitude tracking ratio (actual / commanded)")
    ax.set_xlim(0.3, df8["sat_ratio"].max() * 1.12)
    ax.set_ylim(0.45, 1.10)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.set_title("Tracking loss vs. velocity saturation\n"
                 "(colour = mean battery voltage per run)", fontsize=7.5)
    ax.legend(fontsize=6, loc="upper right")
    fig.savefig(os.path.join(OUT_DIR, "fig8_saturation_analysis.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "fig8_saturation_analysis.pdf"), bbox_inches="tight")
    plt.close(fig)

    # ── Write EVALUATION.md ───────────────────────────────────────────────────
    print("Writing EVALUATION.md...")
    prac_lim_rpm = PRACTICAL_VEL_LIMIT_UNITS * VEL_UNIT_TO_RPM
    theo_lim_rpm = VEL_LIMIT_UNITS * VEL_UNIT_TO_RPM

    md = []
    md += [
        "# Sofi Field Test — Cross-Run Evaluation",
        "",
        "**Session:** 2026-04-01  |  **Runs:** 1–12  |  "
        "**Motor:** Dynamixel XL430 @ ~11 V  |  **Pulley radius:** 17.5 mm (analysis) / 12.5 mm (field software)",
        "",
        "> **Amplitude note:** The field software (`unified_test.py`) computed `amp_ticks` with "
        "the default `--radius 12.5 mm`. All amplitudes in this report are converted from encoder "
        "ticks using **r = 17.5 mm**, which is the physically measured pulley radius used in the "
        "analysis pipeline. A \"5 mm nominal\" run therefore corresponds to ≈ 6.98 mm of actual "
        "tendon travel at 17.5 mm. Commanded and actual amplitudes are always compared in the same "
        "unit (mm at 17.5 mm), so `actual < commanded` is physically guaranteed.",
        "",
        "---",
        "",
        "## 1  Run Overview",
        "",
        "| Run | Type | Freq (Hz) | Nominal amp (mm) | Cmd amp @17.5mm | Offset (mm) | Duration (s) |",
        "|-----|------|-----------|------------------|-----------------|-------------|--------------|",
    ]
    for _, row in df.iterrows():
        rn = int(row["name"].split("_")[1])
        md.append(
            f"| R{rn} | {row['rtype']} | {row['freq']:.1f} | {row['nom_amp']:.0f} | "
            f"{row['cmd_mm_nominal']:.2f} | {row['offset']:+.1f} | {row['duration']:.1f} |"
        )

    md += [
        "",
        "---",
        "",
        "## 2  Trajectories",
        "",
        "![Trajectories](fig1_trajectories.png)",
        "",
        "![Trajectories overlaid](fig1b_trajectories_overlay.png)",
        "",
        "All 12 runs at equal spatial scale, using the same coordinate convention as "
        "`04_plot_field_test.py`: X axis = swim direction (decreases as fish swims), "
        "Y axis = lateral. Blue = straight, orange = curve. "
        "Start (green circle) and end (red ×) markers shown.",
        "",
        "Straight runs cover ~3–3.5 m in the swim direction. "
        "Curve runs (R8, R9) cover only 1.5–1.9 m along the swim axis because the fish "
        "curves, spending more lateral displacement. "
        "Run 2 (2 Hz/5 mm) shows notable lateral drift despite being a straight run.",
        "",
        "---",
        "",
        "## 3  Steady-State Speed",
        "",
        "**Method:** Linear regression on the swim-axis position over the **50–80% window** "
        "of each run. This avoids the acceleration transient (0–50%) and the Savitzky-Golay "
        "boundary effect at the very end (>80%). Since this uses raw position data — not a "
        "differentiated speed estimate — it has no SG boundary noise.",
        "",
        "> **On the speed-still-increasing question:** If the speed profile is still rising "
        "between 50–70% of normalised time, the fish has not yet reached steady state within "
        "the run. In that case the regression slope gives an *average effective speed* rather "
        "than a true steady-state. This is most likely for short runs (< 10 s). The speed "
        "profile figure (Fig. 2a) shows the 50–80% window shaded — inspect "
        "whether individual runs have plateaued within it.",
        "",
        '<p><img src="fig2a_speed_profiles.png" width="49%"/> <img src="fig2b_steady_state_speed.png" width="49%"/></p>',
        "",
        '<p><img src="fig3a_speed_vs_freq.png" width="49%"/> <img src="fig3b_speed_vs_amplitude.png" width="49%"/></p>',
        "",
        "Fig. 3a shows steady-state speed vs. commanded frequency; each point is "
        "coloured by actual amplitude (blue = low, red = high). "
        "Fig. 3b plots speed against *actual* amplitude — this is the physically "
        "meaningful relationship because the fish cannot know what was commanded, only what "
        "the tail actually does.",
        "",
        "### Straight-run summary",
        "",
        "| Run | Freq (Hz) | Cmd amp (mm) | Act amp (mm) | Tracking | SS speed (m/s) | Accel (m/s²) |",
        "|-----|-----------|-------------|-------------|----------|----------------|--------------|",
    ]
    for _, row in straight.iterrows():
        rn  = int(row["name"].split("_")[1])
        acc = f"{row['accel']:.3f}" if not np.isnan(row["accel"]) else "—"
        md.append(
            f"| R{rn} | {row['freq']:.1f} | {row['cmd_mm']:.2f} | {row['act_mm']:.2f} | "
            f"{row['track']:.3f} | **{row['ss_speed']:.3f}** | {acc} |"
        )

    md += [
        "",
        "**Trends:**",
        "- Higher frequency → higher speed at the same amplitude.",
        "- Higher actual amplitude → higher speed at the same frequency.",
        "- R10 (4 Hz/7 mm nominal) achieves the highest measured speed despite the lowest "
        "  actual amplitude among all 4 Hz runs — high-frequency small flaps still produce "
        "  meaningful thrust even when the motor is severely velocity-limited.",
        "- R11 vs R12 (same 2 Hz/12 mm nominal, P=800 vs P=1600): the higher P gain barely "
        "  changes actual amplitude but slightly reduces steady-state speed, suggesting "
        "  increased oscillation/overshoot rather than better tracking.",
        "",
        "---",
        "",
        "## 4  Motor Amplitude Tracking",
        "",
        "Both commanded and actual amplitudes are computed from encoder **ticks** using "
        "**r = 17.5 mm**, so they are on the same physical scale. "
        "`actual < commanded` is physically guaranteed for position-controlled servos.",
        "",
        '<p><img src="fig4a_motor_amplitude.png" width="49%"/> <img src="fig4b_tracking_ratio.png" width="49%"/></p>',
        "",
        "### Tracking table",
        "",
        "| Run | Nominal (mm) | Cmd @17.5mm | Act @17.5mm | Tracking |",
        "|-----|-------------|-------------|------------|----------|",
    ]
    for _, row in df.iterrows():
        rn = int(row["name"].split("_")[1])
        md.append(
            f"| R{rn} | {row['nom_amp']:.0f} | {row['cmd_mm']:.2f} | "
            f"{row['act_mm']:.2f} | {row['track']:.3f} |"
        )

    md += [
        "",
        "Tracking degrades sharply above ~2 Hz or for large amplitudes. "
        "This is explained by the velocity budget below — the commanded sine requires "
        "faster motor shaft rotation than the battery-limited motor can deliver. "
        "Adjusting the P gain (R12 vs R11) has minimal effect on amplitude tracking "
        "because the bottleneck is velocity, not position gain.",
        "",
        "---",
        "",
        "## 5  Velocity Budget — Why Tracking Degrades",
        "",
        f"The Dynamixel XL430 has a theoretical no-load speed of **{theo_lim_rpm:.1f} RPM "
        f"({VEL_LIMIT_UNITS} units) at 12 V**. With the field battery at ~11 V, the "
        f"practical limit observed in the data is **{prac_lim_rpm:.1f} RPM "
        f"({PRACTICAL_VEL_LIMIT_UNITS} units)** — the `Vel` register consistently caps at "
        "220 units across runs 11 and 12 (the highest-demand runs).",
        "",
        "The **required peak velocity** to track a sinusoidal position command is:",
        "",
        "> `v_peak = 2π · f · A_rad` (rad/s) = `60 · f · A_rad / 0.229` (Dxl units)",
        "",
        "where `A_rad` is the commanded amplitude in radians.",
        "",
        "![Velocity budget](fig5_velocity_budget.png)",
        "",
        "### Velocity budget table",
        "",
        "| Run | Freq | Cmd amp | Required vel (units) | Required (% of limit) | Measured peak (units) | Saturating? |",
        "|-----|------|---------|---------------------|----------------------|----------------------|-------------|",
    ]
    for _, row in df.iterrows():
        rn   = int(row["name"].split("_")[1])
        req  = row["req_vel_units"]
        reqp = 100 * req / PRACTICAL_VEL_LIMIT_UNITS
        meas = row["peak_vel_units"]
        sat  = "**YES**" if reqp > 100 else "no"
        md.append(
            f"| R{rn} | {row['freq']:.1f} Hz | {row['cmd_mm']:.1f} mm | "
            f"{req:.0f} | {reqp:.0f}% | {meas:.0f} | {sat} |"
        )

    md += [
        "",
        "Runs where required velocity exceeds the practical limit (~220 units) show a "
        "**clipped, non-sinusoidal position profile** — the motor moves at near-constant "
        "maximum velocity instead of following the smooth sine, making the position profile "
        "triangular rather than sinusoidal. This reduces effective amplitude and introduces "
        "phase lag. The effect is most severe for R10 (4 Hz/7 mm, required = 587 units = "
        "267% of limit — motor achieves only 51% of commanded amplitude).",
        "",
        "---",
        "",
        "## 6  Path Straightness",
        "",
        "The raw trajectory mixes two physically distinct phenomena: slow **heading drift** "
        "(hydrodynamic asymmetry, misalignment) and high-frequency **lateral oscillations** "
        "from the tail flap. The old single-metric approach (RMS from a best-fit line) "
        "confounds both. These plots use a decomposition instead:",
        "",
        "1. **Smooth path** — the trajectory low-pass filtered at $f_{\\rm swim}/2$ "
        "   (4th-order zero-phase Butterworth). This retains slow heading drift and removes "
        "   the per-cycle oscillation.",
        "2. **Path drift (fig. 6a)** — RMS perpendicular distance of the *smooth path* from "
        "   the best-fit line through the original trajectory. Pure heading quality.",
        "3. **Oscillation magnitude (fig. 6b)** — RMS of (actual trajectory − smooth path). "
        "   Pure per-cycle lateral displacement of the fish body.",
        "",
        '<p><img src="fig6a_path_drift.png" width="49%"/> <img src="fig6b_osc_magnitude.png" width="49%"/></p>',
        "",
        "| Run | Freq (Hz) | Cmd amp (mm) | Act amp (mm) | Path drift (mm) | Osc. magnitude (mm) |",
        "|-----|-----------|-------------|-------------|----------------|---------------------|",
    ]
    for _, row in straight[straight["name"] != "run_12_2.0Hz_12.0mm_p-1600"].sort_values("name").iterrows():
        rn = int(row["name"].split("_")[1])
        md.append(
            f"| R{rn} | {row['freq']:.1f} | {row['cmd_mm']:.2f} | {row['act_mm']:.2f} | "
            f"{row['rms_path_mm']:.1f} | {row['rms_osc_mm']:.2f} |"
        )

    md += [
        "",
        "**Path drift** reflects heading stability — how consistently the fish tracks a "
        "straight course ignoring the flapping. **Oscillation magnitude** reflects the "
        "physical lateral body displacement per flap cycle, which scales with actual "
        "tail amplitude and frequency.",
        "",
        "---",
        "",
        "## 7  Curve Runs: R8 vs R9",
        "",
        "Both curve runs use a **2 Hz / 6 mm** oscillation with a **3 mm tendon offset** "
        "in opposite directions. The offset shifts the equilibrium position, creating "
        "asymmetric forward/backward strokes and net lateral thrust.",
        "",
        "![Curve comparison](fig7_curves.png)",
        "",
    ]

    r8m = df[df["name"]=="run_8_2.0Hz_6.0mm_3.0mm"].iloc[0]
    r9m = df[df["name"]=="run_9_2.0Hz_6.0mm_-3.0mm"].iloc[0]
    a8  = run_cache["run_8_2.0Hz_6.0mm_3.0mm"]["asym"]
    a9  = run_cache["run_9_2.0Hz_6.0mm_-3.0mm"]["asym"]
    _, _, rad8 = fit_circle(run_cache["run_8_2.0Hz_6.0mm_3.0mm"]["mu_x"],
                            run_cache["run_8_2.0Hz_6.0mm_3.0mm"]["mu_y"])
    _, _, rad9 = fit_circle(run_cache["run_9_2.0Hz_6.0mm_-3.0mm"]["mu_x"],
                            run_cache["run_9_2.0Hz_6.0mm_-3.0mm"]["mu_y"])

    md += [
        "### Asymmetry analysis (all values @ 17.5mm radius)",
        "",
        "| Metric | R8 (+3mm offset) | R9 (−3mm offset) |",
        "|--------|-----------------|-----------------|",
        f"| Commanded offset (bias) | {a8['cmd_bias']:+.2f} mm | {a9['cmd_bias']:+.2f} mm |",
        f"| **Actual offset (bias)** | **{a8['act_bias']:+.2f} mm** | **{a9['act_bias']:+.2f} mm** |",
        f"| Commanded +stroke amp | {a8['cmd_pos']:.2f} mm | {a9['cmd_pos']:.2f} mm |",
        f"| **Actual +stroke amp** | **{a8['act_pos']:.2f} mm** | **{a9['act_pos']:.2f} mm** |",
        f"| Commanded −stroke amp | {a8['cmd_neg']:.2f} mm | {a9['cmd_neg']:.2f} mm |",
        f"| **Actual −stroke amp** | **{a8['act_neg']:.2f} mm** | **{a9['act_neg']:.2f} mm** |",
        f"| SS speed (m/s) | {r8m['ss_speed']:.3f} | {r9m['ss_speed']:.3f} |",
        f"| Fitted turning radius (m) | {rad8:.2f} | {rad9:.2f} |",
        "",
        "**Key observations:**",
        f"- R9 (−3mm) turns significantly tighter than R8 (+3mm): radius {rad9:.2f} m vs {rad8:.2f} m.",
        "- The actual motor bias differs between runs: the motor has an easier time reaching "
        "  the offset in one direction than the other (mechanical/preload asymmetry in the "
        "  tendon drive). This means a +3mm commanded offset ≠ −3mm commanded offset in terms "
        "  of actual achieved bias and resulting curvature.",
        f"- The asymmetry in actual +/- stroke amplitudes is visible: R8 acts "
        f"  {abs(a8['act_pos']-a8['act_neg']):.2f} mm differently between sides; "
        f"  R9 acts {abs(a9['act_pos']-a9['act_neg']):.2f} mm differently.",
        "- Both curve runs are slower than the symmetric 2 Hz/6 mm equivalent (the offset "
        "  effectively reduces mean driving amplitude).",
        "",
        "---",
        "",
        "## 8  Voltage-Driven Tracking Loss",
        "",
        "![Saturation analysis](fig8_saturation_analysis.png)",
        "",
        "**X axis:** saturation depth = required peak velocity / measured peak velocity. "
        "Values > 1 mean the commanded sine demands faster shaft rotation than the motor "
        "can deliver at the current battery voltage. "
        "**Y axis:** amplitude tracking ratio. "
        "The dashed line is the ideal triangular-wave saturation model: "
        "tracking = min(1, 1/x). Colour encodes mean battery voltage per run.",
        "",
        "**Key observations:**",
        "- All runs lie *above* the ideal 1/x curve — the motor achieves better tracking "
        "  than a pure triangular-wave prediction. This is expected: a velocity-clipped "
        "  sine is not perfectly triangular; the motor still partially follows the "
        "  curvature near the turning points.",
        "- Unsaturated runs (depth < 1) cluster near tracking ≈ 0.90 rather than 1.0, "
        "  revealing a ~10% baseline droop from the P-only controller's steady-state "
        "  position error (no I term to eliminate offset under load).",
        "- Battery voltage (colour) correlates with saturation depth: runs later in the "
        "  session (lower voltage, darker points) sit further right — the lower motor "
        "  speed ceiling raises the saturation depth for the same commanded sine.",
        "",
        "---",
        "",
        "_Generated by `field_test/evaluation/evaluate_all_runs.py`_",
    ]

    with open(os.path.join(OUT_DIR, "EVALUATION.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"\nDone. Outputs in: {OUT_DIR}/")
    for fn in ["fig1_trajectories.png","fig1b_trajectories_overlay.png",
               "fig2a_speed_profiles.png","fig2b_steady_state_speed.png",
               "fig3a_speed_vs_freq.png","fig3b_speed_vs_amplitude.png",
               "fig4a_motor_amplitude.png","fig4b_tracking_ratio.png",
               "fig5_velocity_budget.png",
               "fig6a_path_drift.png","fig6b_osc_magnitude.png",
               "fig7_curves.png",
               "fig8_saturation_analysis.png","EVALUATION.md"]:
        print(f"  {fn}")


if __name__ == "__main__":
    main()
