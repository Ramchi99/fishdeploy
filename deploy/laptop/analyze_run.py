#!/usr/bin/env python3
"""
Post-run diagnostics for Workflow B (open-loop / shadow) runs.

Analyzes a single run.csv and prints a text report + saves a PNG with:
  - Control loop timing & jitter
  - Vision latency distribution
  - UDP frame drop rate & arrival jitter
  - Motor tracking accuracy (trajectory vs encoder)
  - IMU sensor noise

Usage (from repo root):
    python deploy/laptop/analyze_run.py data/deploy_runs/YYYYMMDD_HHMMSS_<name>
    python deploy/laptop/analyze_run.py path/to/run.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ────────────────────────────────────────────────────────────────────

def _pct(arr, p):
    return float(np.percentile(arr, p))

def _fmt(val, unit="", decimals=2):
    return f"{val:.{decimals}f}{(' ' + unit) if unit else ''}"


def _banner(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def _row(label, value, note=""):
    note_str = f"  ({note})" if note else ""
    print(f"  {label:<35s} {value}{note_str}")


# ── main analysis ──────────────────────────────────────────────────────────────

def analyze(run_dir: str, save_png: bool = True) -> dict:
    # Locate run.csv
    if run_dir.endswith(".csv"):
        csv_path = run_dir
        run_dir = os.path.dirname(csv_path)
    else:
        csv_path = os.path.join(run_dir, "run.csv")

    if not os.path.isfile(csv_path):
        sys.exit(f"[ERROR] run.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Load config for clock offset
    cfg = {}
    cfg_path = os.path.join(run_dir, "run_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)

    n = len(df)
    if n < 2:
        sys.exit("[ERROR] run.csv has fewer than 2 rows — nothing to analyze.")

    is_shadow = "policy_motor_setpoint" in df.columns
    mode = cfg.get("mode", "shadow" if is_shadow else "open_loop")
    clock_offset_ms = cfg.get("clock_offset_ms", 0.0)

    # ── 1. Loop timing ──────────────────────────────────────────────────────────
    pi_times = df["pi_time"].to_numpy()
    step_dt_ms = np.diff(pi_times) * 1000.0
    total_duration_s = pi_times[-1] - pi_times[0]
    actual_hz = (n - 1) / total_duration_s if total_duration_s > 0 else 0.0
    overrun_frac = np.mean(step_dt_ms > 30.0)  # fraction of steps >1.5× 20 ms target

    # ── 2. Vision / UDP ─────────────────────────────────────────────────────────
    vision_age = df["vision_age_ms"].to_numpy()
    frame_seq = df["frame_seq"].to_numpy(dtype=np.int64)

    # Detect new frames: rows where frame_seq changed from the previous row
    seq_diff = np.diff(frame_seq)
    new_frame_mask = np.concatenate([[True], seq_diff != 0])
    new_frame_indices = np.where(new_frame_mask)[0]

    # Frame drops: expected range vs actually received unique seqs
    first_seq, last_seq = int(frame_seq.min()), int(frame_seq.max())
    expected_frames = last_seq - first_seq + 1
    received_unique = len(np.unique(frame_seq))
    dropped_frames = expected_frames - received_unique
    drop_rate = dropped_frames / expected_frames if expected_frames > 0 else 0.0

    # Inter-arrival jitter (time between control steps that received a new frame)
    if len(new_frame_indices) >= 2:
        new_frame_times = pi_times[new_frame_indices]
        arrival_gaps_ms = np.diff(new_frame_times) * 1000.0
        # Filter out gaps caused by multi-frame drops (>5× expected camera interval)
        camera_dt_est_ms = float(np.median(arrival_gaps_ms))
        normal_gaps = arrival_gaps_ms[arrival_gaps_ms < camera_dt_est_ms * 5]
        if len(normal_gaps) > 0:
            arrival_jitter_ms = float(np.std(normal_gaps))
            mean_arrival_gap_ms = float(np.mean(normal_gaps))
        else:
            arrival_jitter_ms = float(np.std(arrival_gaps_ms))
            mean_arrival_gap_ms = float(np.mean(arrival_gaps_ms))
    else:
        arrival_jitter_ms = 0.0
        mean_arrival_gap_ms = 0.0
        arrival_gaps_ms = np.array([])

    # Fraction of steps with stale / lost vision
    stale_frac = float(np.mean(vision_age > 200.0))
    lost_frac = float(np.mean(vision_age > 500.0))

    # ── 3. Motor tracking ───────────────────────────────────────────────────────
    motor_pos = np.arctan2(df["obs_01"].to_numpy(), df["obs_00"].to_numpy())
    traj_cmd = df["trajectory_cmd"].to_numpy()
    tracking_err = motor_pos - traj_cmd
    tracking_rmse = float(np.sqrt(np.mean(tracking_err ** 2)))
    tracking_max = float(np.max(np.abs(tracking_err)))

    # ── 4. IMU noise ────────────────────────────────────────────────────────────
    # Two observation layouts exist in the logs: the deployed 24-D one (ObsAssemblerCPG24D: yaw rate at
    # obs_03, the most recent vision frame at obs_14..17, no target velocity) and the 14-D one of the
    # May 2026 shadow runs (wx, wy, wz at obs_03..05, dist/vec at obs_06..08, target velocity at 09..10).
    n_obs = sum(c.startswith("obs_") for c in df.columns)
    layout_24d = n_obs == 24
    if layout_24d:
        wz = df["obs_03"].to_numpy()
        wx = wy = np.zeros_like(wz)                       # not in the observation
    else:
        wx = df["obs_03"].to_numpy()
        wy = df["obs_04"].to_numpy()
        wz = df["obs_05"].to_numpy()
    imu_std = {"wx": float(np.std(wx)), "wy": float(np.std(wy)), "wz": float(np.std(wz))}
    imu_mean = {"wx": float(np.mean(wx)), "wy": float(np.mean(wy)), "wz": float(np.mean(wz))}

    # ── 5. Motor velocity noise ─────────────────────────────────────────────────
    motor_vel = df["obs_02"].to_numpy()
    motor_vel_std = float(np.std(motor_vel))
    motor_vel_mean = float(np.mean(motor_vel))

    # ── 6. Vision channel noise (meaningful only for static / zero-trajectory runs)
    if layout_24d:
        dist_arr   = df["obs_14"].to_numpy()
        vec_x_arr  = df["obs_15"].to_numpy()
        vec_y_arr  = df["obs_16"].to_numpy()
        vvel_x_arr = vvel_y_arr = np.zeros_like(dist_arr)  # not in the observation
    else:
        dist_arr   = df["obs_06"].to_numpy()
        vec_x_arr  = df["obs_07"].to_numpy()
        vec_y_arr  = df["obs_08"].to_numpy()
        vvel_x_arr = df["obs_09"].to_numpy()
        vvel_y_arr = df["obs_10"].to_numpy()
    vision_noise = {
        "dist_mean":   float(np.mean(dist_arr)),
        "dist_std":    float(np.std(dist_arr)),
        "vec_x_mean":  float(np.mean(vec_x_arr)),
        "vec_x_std":   float(np.std(vec_x_arr)),
        "vec_y_mean":  float(np.mean(vec_y_arr)),
        "vec_y_std":   float(np.std(vec_y_arr)),
        "vvel_x_std":  float(np.std(vvel_x_arr)),
        "vvel_y_std":  float(np.std(vvel_y_arr)),
    }

    is_zero_traj = (traj_cmd == 0).all()

    # ── Text report ─────────────────────────────────────────────────────────────
    run_name = os.path.basename(run_dir.rstrip("/"))
    _banner(f"Run diagnostics: {run_name}")
    _row("Mode", mode)
    _row("Total steps", str(n))
    _row("Duration", _fmt(total_duration_s, "s"), f"target {args.duration if hasattr(args, 'duration') else '?'} s")
    _row("Actual loop rate", _fmt(actual_hz, "Hz"))
    _row("Clock offset (Pi − laptop)", _fmt(clock_offset_ms, "ms"))

    _banner("Control loop timing  (target 20 ms)")
    _row("Mean step dt", _fmt(float(np.mean(step_dt_ms)), "ms"))
    _row("Std step dt", _fmt(float(np.std(step_dt_ms)), "ms"))
    _row("Max step dt", _fmt(float(np.max(step_dt_ms)), "ms"))
    _row("Overrun fraction (>30 ms)", _fmt(overrun_frac * 100, "%", 1))

    _banner("Vision / UDP connection")
    _row("Vision age — mean", _fmt(float(np.mean(vision_age)), "ms"))
    _row("Vision age — median", _fmt(_pct(vision_age, 50), "ms"))
    _row("Vision age — p95", _fmt(_pct(vision_age, 95), "ms"))
    _row("Vision age — p99", _fmt(_pct(vision_age, 99), "ms"))
    _row("Vision age — max", _fmt(float(np.max(vision_age)), "ms"))
    _row("Stale fraction  (>200 ms)", _fmt(stale_frac * 100, "%", 1))
    _row("Lost fraction   (>500 ms)", _fmt(lost_frac * 100, "%", 1))
    _row("Unique frames received", str(received_unique))
    _row("Frames expected (seq range)", str(expected_frames))
    _row("Dropped frames", str(dropped_frames))
    _row("Frame drop rate", _fmt(drop_rate * 100, "%", 1))
    if mean_arrival_gap_ms > 0:
        _row("Mean camera arrival gap", _fmt(mean_arrival_gap_ms, "ms"),
             f"≈ {1000/mean_arrival_gap_ms:.1f} Hz")
        _row("Camera arrival jitter (1σ)", _fmt(arrival_jitter_ms, "ms"))

    _banner("Motor tracking & noise")
    _row("Tracking RMSE", _fmt(tracking_rmse, "rad"))
    _row("Max absolute error", _fmt(tracking_max, "rad"))
    _row("Motor pos noise (std)", _fmt(float(np.std(motor_pos)), "rad"),
         "encoder noise floor — obs[0,1]")
    _row("Motor vel noise (std)", _fmt(motor_vel_std, "rad/s"),
         "velocity noise floor — obs[2]")
    _row("Motor vel bias (mean)", _fmt(motor_vel_mean, "rad/s"))

    _banner("IMU noise (full run)")
    for ax in ("wx", "wy", "wz"):
        _row(f"  {ax} — mean / std",
             f"{imu_mean[ax]:+.4f}  /  {imu_std[ax]:.4f} rad/s")

    if is_zero_traj:
        _banner("Vision noise  (valid: zero trajectory — fish static)")
    else:
        _banner("Vision noise  (NOTE: non-zero trajectory — values include motion signal)")
    _row("  dist_target std",  _fmt(vision_noise["dist_std"],   "m"),   "obs[6]")
    _row("  vec_target_x std", _fmt(vision_noise["vec_x_std"],  "m"),   "obs[7]")
    _row("  vec_target_y std", _fmt(vision_noise["vec_y_std"],  "m"),   "obs[8]")
    _row("  vel_vec_x std",    _fmt(vision_noise["vvel_x_std"], "m/s"), "obs[9]")
    _row("  vel_vec_y std",    _fmt(vision_noise["vvel_y_std"], "m/s"), "obs[10]")

    # ── Diagnostic PNG ──────────────────────────────────────────────────────────
    if save_png:
        t = pi_times - pi_times[0]  # relative time axis

        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f"Run diagnostics — {run_name}", fontsize=11)

        # Panel 1: loop dt histogram
        ax = axes[0, 0]
        ax.hist(step_dt_ms, bins=60, color="steelblue", edgecolor="none")
        ax.axvline(20.0, color="green", lw=1.5, label="target 20 ms")
        ax.axvline(30.0, color="red", lw=1.5, ls="--", label="30 ms threshold")
        ax.set_xlabel("Step Δt [ms]")
        ax.set_ylabel("Count")
        ax.set_title(f"Control Loop Timing  (mean {np.mean(step_dt_ms):.1f} ms, "
                     f"std {np.std(step_dt_ms):.1f} ms)")
        ax.legend(fontsize=8)

        # Panel 2: vision age histogram
        ax = axes[0, 1]
        ax.hist(vision_age, bins=60, color="darkorange", edgecolor="none")
        ax.axvline(200.0, color="gold", lw=1.5, ls="--", label="stale 200 ms")
        ax.axvline(500.0, color="red", lw=1.5, ls="--", label="lost 500 ms")
        ax.set_xlabel("Vision age at control step [ms]")
        ax.set_ylabel("Count")
        ax.set_title(f"Vision Latency  (mean {np.mean(vision_age):.1f} ms, "
                     f"p99 {_pct(vision_age, 99):.1f} ms)")
        ax.legend(fontsize=8)

        # Panel 3: motor tracking over time
        ax = axes[1, 0]
        ax.plot(t, traj_cmd, lw=1, color="steelblue", label="trajectory cmd")
        ax.plot(t, motor_pos, lw=0.7, alpha=0.7, color="tomato", label="encoder")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [rad]")
        ax.set_title(f"Motor Tracking  (RMSE {tracking_rmse:.4f} rad)")
        ax.legend(fontsize=8)

        # Panel 4: tracking error over time
        ax = axes[1, 1]
        ax.plot(t, tracking_err, lw=0.7, color="purple")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Error [rad]")
        ax.set_title("Tracking Error (encoder − setpoint)")

        # Panel 5: IMU gyro time series
        ax = axes[2, 0]
        ax.plot(t, wx, lw=0.6, label=f"wx  σ={imu_std['wx']:.3f}")
        ax.plot(t, wy, lw=0.6, label=f"wy  σ={imu_std['wy']:.3f}")
        ax.plot(t, wz, lw=0.6, label=f"wz  σ={imu_std['wz']:.3f}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angular velocity [rad/s]")
        ax.set_title("IMU Gyroscope")
        ax.legend(fontsize=8)

        # Panel 6: frame arrival gaps or vision age time series
        ax = axes[2, 1]
        ax.plot(t, vision_age, lw=0.7, color="darkorange")
        ax.axhline(200.0, color="gold", lw=1, ls="--", label="stale")
        ax.axhline(500.0, color="red", lw=1, ls="--", label="lost")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Vision age [ms]")
        ax.set_title(f"Vision Age Over Time  (drop rate {drop_rate*100:.1f}%)")
        ax.legend(fontsize=8)

        fig.tight_layout()
        out_path = os.path.join(run_dir, "diagnostics.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"\n  Saved diagnostic plot → {out_path}")

    motor_pos_std = float(np.std(motor_pos))

    results = {
        "mode": mode,
        "n_steps": n,
        "duration_s": total_duration_s,
        "actual_hz": actual_hz,
        "loop_dt_mean_ms": float(np.mean(step_dt_ms)),
        "loop_dt_std_ms": float(np.std(step_dt_ms)),
        "loop_dt_max_ms": float(np.max(step_dt_ms)),
        "overrun_frac": overrun_frac,
        "vision_age_mean_ms": float(np.mean(vision_age)),
        "vision_age_p95_ms": _pct(vision_age, 95),
        "vision_age_p99_ms": _pct(vision_age, 99),
        "vision_age_max_ms": float(np.max(vision_age)),
        "stale_frac": stale_frac,
        "lost_frac": lost_frac,
        "drop_rate": drop_rate,
        "camera_hz_est": 1000.0 / mean_arrival_gap_ms if mean_arrival_gap_ms > 0 else 0.0,
        "arrival_jitter_ms": arrival_jitter_ms,
        "tracking_rmse_rad": tracking_rmse,
        "tracking_max_err_rad": tracking_max,
        "motor_pos_noise_std_rad": motor_pos_std,
        "motor_vel_noise_std_rad_s": motor_vel_std,
        "motor_vel_bias_rad_s": motor_vel_mean,
        "imu_std": imu_std,
        "imu_mean": imu_mean,
        "vision_noise": vision_noise,
        "is_zero_traj": bool(is_zero_traj),
        "clock_offset_ms": clock_offset_ms,
    }

    # ── Noise config JSON (for RL training) ────────────────────────────────────
    noise_cfg = {
        "_note": (
            "Generated by analyze_run.py. "
            f"Source run: {run_name}. "
            f"{'Zero-trajectory run — noise estimates are valid.' if is_zero_traj else 'WARNING: non-zero trajectory — vision noise includes motion signal.'}"
        ),
        "control_loop": {
            "actual_hz":         round(actual_hz, 2),
            # → sim dt should match: dt = 1 / actual_hz (currently hardcoded as 1/50)
            "step_dt_std_ms":    round(float(np.std(step_dt_ms)), 2),
            # → add timing jitter in sim: dt ~ Normal(1/hz, step_dt_std_ms/1000)
        },
        "obs_noise_std": {
            # Physical channels; the obs_NN prefixes are the indices of the 14-D layout the May 2026 runs used
            "obs_00_01_motor_pos_rad":    round(motor_pos_std, 5),
            # → noise on motor angle BEFORE cos/sin encoding: pos_rad += N(0, this)
            "obs_02_motor_vel_rad_s":     round(motor_vel_std, 5),
            # → obs[2] += N(0, this)
            "obs_03_imu_wx_rad_s":        round(imu_std["wx"], 5),
            # → obs[3] += N(0, this)
            "obs_04_imu_wy_rad_s":        round(imu_std["wy"], 5),
            # → obs[4] += N(0, this)
            "obs_05_imu_wz_rad_s":        round(imu_std["wz"], 5),
            # → obs[5] += N(0, this)
            "obs_06_dist_target_m":       round(vision_noise["dist_std"], 5),
            # → obs[6] += N(0, this)  [only reliable if zero-trajectory]
            "obs_07_vec_target_x_m":      round(vision_noise["vec_x_std"], 5),
            # → obs[7] += N(0, this)  [only reliable if zero-trajectory]
            "obs_08_vec_target_y_m":      round(vision_noise["vec_y_std"], 5),
            # → obs[8] += N(0, this)  [only reliable if zero-trajectory]
            "obs_09_vel_vec_x_m_s":       round(vision_noise["vvel_x_std"], 5),
            # → obs[9] += N(0, this)  [only reliable if zero-trajectory]
            "obs_10_vel_vec_y_m_s":       round(vision_noise["vvel_y_std"], 5),
            # → obs[10] += N(0, this) [only reliable if zero-trajectory]
        },
        "obs_bias": {
            # Systematic offsets (non-zero mean) — subtract in sim or add to obs
            "obs_02_motor_vel_rad_s":  round(motor_vel_mean, 5),
            # → motor vel reads this much even when stationary
            "obs_03_imu_wx_rad_s":     round(imu_mean["wx"], 5),
            "obs_04_imu_wy_rad_s":     round(imu_mean["wy"], 5),
            "obs_05_imu_wz_rad_s":     round(imu_mean["wz"], 5),
        },
        "vision_latency": {
            "mean_ms":   round(float(np.mean(vision_age)), 1),
            # → tune vision_stale_warn_ms: set to ~3× mean (currently 200 ms)
            "p95_ms":    round(_pct(vision_age, 95), 1),
            "p99_ms":    round(_pct(vision_age, 99), 1),
            "max_ms":    round(float(np.max(vision_age)), 1),
            # → tune vision_stale_stop_ms: should be > max observed (currently 500 ms)
            "drop_rate": round(drop_rate, 3),
            # → sim vision dropout: skip obs update with this probability each step
            "camera_hz": round(1000.0 / mean_arrival_gap_ms if mean_arrival_gap_ms > 0 else 0.0, 1),
            # → sim camera runs at this rate, not the control loop rate
        },
    }

    json_path = os.path.join(run_dir, "noise_config.json")
    with open(json_path, "w") as f:
        json.dump(noise_cfg, f, indent=2)
    print(f"  Saved noise config      → {json_path}")

    # ── Markdown report ────────────────────────────────────────────────────────
    traj_note = (
        "Zero-trajectory run — motor stationary. Noise estimates are valid floor measurements."
        if is_zero_traj else
        "**WARNING:** Non-zero trajectory. Vision/motor noise includes motion signal — not a clean noise floor."
    )
    md_lines = [
        f"# Hardware Noise & Latency Report",
        f"",
        f"**Run:** `{run_name}`  ",
        f"**Mode:** {mode}  ",
        f"**Duration:** {total_duration_s:.1f} s ({n} steps @ {actual_hz:.1f} Hz)  ",
        f"**Clock offset (Pi − laptop):** {clock_offset_ms:+.1f} ms  ",
        f"",
        f"> {traj_note}",
        f"",
        f"---",
        f"",
        f"## Obs Noise Standard Deviations",
        f"",
        f"Input these into the RL training environment as `obs_noise_std`.",
        f"",
        f"| Obs index | Channel | Std | Unit | Note |",
        f"|-----------|---------|-----|------|------|",
        f"| 0, 1 | motor_pos (cos/sin) | {motor_pos_std:.5f} | rad | apply to angle before encoding |",
        f"| 2 | motor_vel | {motor_vel_std:.5f} | rad/s | |",
        f"| 3 | imu_wx | {imu_std['wx']:.5f} | rad/s | |",
        f"| 4 | imu_wy | {imu_std['wy']:.5f} | rad/s | |",
        f"| 5 | imu_wz | {imu_std['wz']:.5f} | rad/s | |",
        f"| 6 | dist_target | {vision_noise['dist_std']:.5f} | m | zero-traj only |",
        f"| 7 | vec_target_x | {vision_noise['vec_x_std']:.5f} | m | zero-traj only |",
        f"| 8 | vec_target_y | {vision_noise['vec_y_std']:.5f} | m | zero-traj only |",
        f"| 9 | vel_vec_x | {vision_noise['vvel_x_std']:.5f} | m/s | zero-traj only |",
        f"| 10 | vel_vec_y | {vision_noise['vvel_y_std']:.5f} | m/s | zero-traj only |",
        f"",
        f"## Obs Biases (systematic offsets)",
        f"",
        f"| Channel | Mean | Unit |",
        f"|---------|------|------|",
        f"| motor_vel (obs[2]) | {motor_vel_mean:+.5f} | rad/s |",
        f"| imu_wx (obs[3]) | {imu_mean['wx']:+.5f} | rad/s |",
        f"| imu_wy (obs[4]) | {imu_mean['wy']:+.5f} | rad/s |",
        f"| imu_wz (obs[5]) | {imu_mean['wz']:+.5f} | rad/s |",
        f"",
        f"## Vision Latency",
        f"",
        f"| Metric | Value | Threshold / guidance |",
        f"|--------|-------|----------------------|",
        f"| Mean age | {np.mean(vision_age):.1f} ms | — |",
        f"| p95 age | {_pct(vision_age, 95):.1f} ms | — |",
        f"| p99 age | {_pct(vision_age, 99):.1f} ms | — |",
        f"| Max age | {np.max(vision_age):.1f} ms | `vision_stale_stop_ms` should be > this |",
        f"| Frame drop rate | {drop_rate*100:.1f}% | use as vision dropout prob in sim |",
        f"| Camera rate | {1000/mean_arrival_gap_ms if mean_arrival_gap_ms>0 else 0:.1f} Hz | sim camera rate |",
        f"",
        f"## Control Loop",
        f"",
        f"| Metric | Value | Guidance |",
        f"|--------|-------|----------|",
        f"| Actual Hz | {actual_hz:.2f} | sim dt = 1/{actual_hz:.1f} = {1/actual_hz*1000:.2f} ms |",
        f"| Step dt std | {np.std(step_dt_ms):.2f} ms | add timing jitter in sim |",
        f"| Overrun fraction | {overrun_frac*100:.1f}% | steps > 30 ms |",
        f"",
        f"---",
        f"*Generated by `deploy/laptop/analyze_run.py`*",
    ]
    md_path = os.path.join(run_dir, "noise_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Saved noise report      → {md_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a Workflow B run.csv for latency, frequency, and noise"
    )
    parser.add_argument(
        "run",
        help="Path to run directory (containing run.csv) or directly to run.csv"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip saving the diagnostic PNG"
    )
    args = parser.parse_args()

    analyze(args.run, save_png=not args.no_plot)
