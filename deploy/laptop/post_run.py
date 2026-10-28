#!/usr/bin/env python3
"""
Master post-run analysis script for RL deployment runs.

Usage:
    python deploy/laptop/post_run.py data/deploy_runs/20260508_143022_pool_test_1
    python deploy/laptop/post_run.py data/deploy_runs/20260508_143022_pool_test_1 --render
    python deploy/laptop/post_run.py data/deploy_runs/20260508_143022_pool_test_1 --no-noise
    python deploy/laptop/post_run.py data/deploy_runs/20260508_143022_pool_test_1 --all
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import deploy.laptop.analyze_run as _analyze_run_mod
from deploy.laptop.analyze_run import analyze
from deploy.laptop.plot_run import plot_run


class _FakeArgs:
    """Stub for analyze_run's module-level ``args`` reference (line 167).

    analyze_run.analyze() references the module-global ``args`` that is only
    set when the script is run as __main__.  We inject a stub so that the
    ``hasattr(args, 'duration')`` guard resolves cleanly.
    """
    pass


_OBS_DIM_BY_ENV = {"cpg_12d": 12, "cpg_14d": 14, "cpg_24d": 24}


def _load_config(raw_dir: str) -> dict:
    cfg_path = os.path.join(raw_dir, "run_config.json")
    if not os.path.exists(cfg_path):
        sys.exit(f"[post_run] run_config.json not found in {raw_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def _load_rl_csv(raw_dir: str, env_type: str) -> pd.DataFrame:
    """Read run.csv with explicit column names based on obs_dim.

    Handles both old runs (14-col header for 24D obs, pandas misaligns) and
    new runs (correct obs-dim header) by always providing explicit names.
    """
    obs_dim = _OBS_DIM_BY_ENV.get(env_type, 14)
    all_cols = (
        ["pi_time", "laptop_time"]
        + [f"obs_{i:02d}" for i in range(obs_dim)]
        + [f"act_{i:02d}" for i in range(6)]
        + ["motor_setpoint", "reward", "done",
           "vision_age_ms", "vision_flags", "frame_seq", "cpg_phase"]
    )
    return pd.read_csv(
        os.path.join(raw_dir, "run.csv"),
        names=all_cols,
        skiprows=1,
    )


def _generate_summary_shadow_openloop(raw_dir: str, processed_dir: str, df: pd.DataFrame, mode: str) -> None:
    t = df["pi_time"].to_numpy() - df["pi_time"].iloc[0]
    motor_pos = np.arctan2(df["obs_01"].to_numpy(), df["obs_00"].to_numpy())
    traj_cmd = df["trajectory_cmd"].to_numpy()
    dist = df["obs_06"].to_numpy()
    vision_ages = df["vision_age_ms"].to_numpy()

    run_name = os.path.basename(os.path.dirname(raw_dir))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Run: {run_name} ({mode})", fontsize=12)

    axes[0, 0].plot(t, traj_cmd, lw=1, label="traj cmd")
    axes[0, 0].plot(t, motor_pos, lw=0.7, alpha=0.7, label="encoder")
    if mode == "shadow" and "policy_motor_setpoint" in df.columns:
        axes[0, 0].plot(t, df["policy_motor_setpoint"].to_numpy(), lw=1, ls="--", label="policy cmd")
    axes[0, 0].set_xlabel("Time [s]"); axes[0, 0].set_ylabel("rad")
    axes[0, 0].set_title("Motor Setpoint vs Encoder"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(t, dist, lw=1, color="tab:blue")
    axes[0, 1].set_xlabel("Time [s]"); axes[0, 1].set_ylabel("dist [m]")
    axes[0, 1].set_title("Distance to Target")

    axes[1, 0].hist(vision_ages, bins=60, color="tab:green", edgecolor="none")
    axes[1, 0].axvline(200, ls="--", lw=1, color="gold", label="stale 200ms")
    axes[1, 0].axvline(500, ls="--", lw=1, color="red", label="lost 500ms")
    axes[1, 0].set_xlabel("Vision age [ms]"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Vision Latency Distribution"); axes[1, 0].legend(fontsize=8)

    if mode == "shadow" and "policy_act_raw_04" in df.columns:
        axes[1, 1].plot(t, df["policy_act_raw_04"].to_numpy(), lw=1, color="tab:orange")
        axes[1, 1].set_xlabel("Time [s]"); axes[1, 1].set_ylabel("omega_raw [-1,1]")
        axes[1, 1].set_title("Shadow: Policy Omega Output")
    else:
        axes[1, 1].plot(t, df["obs_05"].to_numpy(), lw=1, color="tab:red")
        axes[1, 1].set_xlabel("Time [s]"); axes[1, 1].set_ylabel("wz [rad/s]")
        axes[1, 1].set_title("IMU Angular Velocity (wz)")

    fig.tight_layout()
    fig.savefig(os.path.join(processed_dir, "summary.png"), dpi=150)
    plt.close(fig)
    print("[post_run] Saved processed/summary.png")


def _generate_diagnostics_rl(raw_dir: str, processed_dir: str, df: pd.DataFrame) -> None:
    """6-panel diagnostics PNG for RL mode (no trajectory_cmd column)."""
    pi_times = df["pi_time"].to_numpy()
    t = pi_times - pi_times[0]
    step_dt_ms = np.diff(pi_times) * 1000.0
    vision_age = df["vision_age_ms"].to_numpy()
    motor_pos = np.arctan2(df["obs_01"].to_numpy(), df["obs_00"].to_numpy())
    motor_sp = df["motor_setpoint"].to_numpy()
    tracking_err = motor_pos - motor_sp
    tracking_rmse = float(np.sqrt(np.mean(tracking_err ** 2)))
    wx = df["obs_03"].to_numpy()
    wy = df["obs_04"].to_numpy()
    wz = df["obs_05"].to_numpy()
    frame_seq = df["frame_seq"].to_numpy(dtype=np.int64)
    expected = int(frame_seq.max()) - int(frame_seq.min()) + 1
    received = len(np.unique(frame_seq))
    drop_rate = (expected - received) / expected if expected > 0 else 0.0

    run_name = os.path.basename(os.path.dirname(raw_dir))
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Run diagnostics — {run_name}", fontsize=11)

    ax = axes[0, 0]
    ax.hist(step_dt_ms, bins=60, color="steelblue", edgecolor="none")
    ax.axvline(20.0, color="green", lw=1.5, label="target 20ms")
    ax.axvline(30.0, color="red", lw=1.5, ls="--", label="30ms threshold")
    ax.set_xlabel("Step Δt [ms]"); ax.set_ylabel("Count")
    ax.set_title(f"Control Loop Timing (mean {np.mean(step_dt_ms):.1f} ms)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.hist(vision_age, bins=60, color="darkorange", edgecolor="none")
    ax.axvline(200.0, color="gold", lw=1.5, ls="--", label="stale 200ms")
    ax.axvline(500.0, color="red", lw=1.5, ls="--", label="lost 500ms")
    ax.set_xlabel("Vision age [ms]"); ax.set_ylabel("Count")
    ax.set_title(f"Vision Latency (mean {np.mean(vision_age):.1f} ms)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, motor_sp, lw=1, color="steelblue", label="setpoint")
    ax.plot(t, motor_pos, lw=0.7, alpha=0.7, color="tomato", label="encoder")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position [rad]")
    ax.set_title(f"Motor Tracking (RMSE {tracking_rmse:.4f} rad)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, tracking_err, lw=0.7, color="purple")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Error [rad]")
    ax.set_title("Tracking Error (encoder − setpoint)")

    ax = axes[2, 0]
    ax.plot(t, wx, lw=0.6, label=f"wx  σ={np.std(wx):.3f}")
    ax.plot(t, wy, lw=0.6, label=f"wy  σ={np.std(wy):.3f}")
    ax.plot(t, wz, lw=0.6, label=f"wz  σ={np.std(wz):.3f}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Angular velocity [rad/s]")
    ax.set_title("IMU Gyroscope"); ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(t, vision_age, lw=0.7, color="darkorange")
    ax.axhline(200.0, color="gold", lw=1, ls="--", label="stale")
    ax.axhline(500.0, color="red", lw=1, ls="--", label="lost")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Vision age [ms]")
    ax.set_title(f"Vision Age Over Time (drop rate {drop_rate*100:.1f}%)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(processed_dir, "diagnostics.png"), dpi=130)
    plt.close(fig)
    print("[post_run] Saved processed/diagnostics.png")


def run_analysis(run_dir: str, render: bool = False, no_noise: bool = False) -> None:
    raw_dir = os.path.join(run_dir, "raw")
    if not os.path.isdir(raw_dir):
        sys.exit(f"[post_run] raw/ not found in {run_dir}")

    cfg = _load_config(raw_dir)
    if "mode" in cfg:
        mode = cfg["mode"]
    elif "model_hash" in cfg and "trajectory" not in cfg:
        mode = "rl"
    else:
        mode = "open_loop"
    is_zero_traj = cfg.get("trajectory") == "zero"

    processed_dir = os.path.join(run_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    csv_path = os.path.join(raw_dir, "run.csv")
    if not os.path.exists(csv_path):
        sys.exit(f"[post_run] run.csv not found in {raw_dir}")

    if mode == "rl":
        df = _load_rl_csv(raw_dir, cfg.get("env_type", "cpg_14d"))
    else:
        df = pd.read_csv(csv_path)

    # Summary PNG — mode-specific
    if mode == "rl":
        # plot_run works for RL mode; writes to raw_dir/summary.png then we move it
        plot_run(raw_dir, df=df)
        src = os.path.join(raw_dir, "summary.png")
        dst = os.path.join(processed_dir, "summary.png")
        if os.path.exists(src):
            shutil.move(src, dst)
            print("[post_run] Saved processed/summary.png")
    else:
        _generate_summary_shadow_openloop(raw_dir, processed_dir, df, mode)

    # Diagnostics PNG
    if mode == "rl":
        _generate_diagnostics_rl(raw_dir, processed_dir, df)
    else:
        # analyze_run writes to raw_dir; move outputs to processed/
        # Inject stub args so analyze_run.py's module-level ``args`` reference
        # (line 167) doesn't raise NameError when called programmatically.
        if not hasattr(_analyze_run_mod, "args"):
            _analyze_run_mod.args = _FakeArgs()
        analyze(raw_dir, save_png=True)
        for fname in ("diagnostics.png",):
            src = os.path.join(raw_dir, fname)
            if os.path.exists(src):
                shutil.move(src, os.path.join(processed_dir, fname))
        print("[post_run] Saved processed/diagnostics.png")

        # Noise files — always produced by analyze(); move or remove based on flags
        for fname in ("noise_config.json", "noise_report.md"):
            src = os.path.join(raw_dir, fname)
            if os.path.exists(src):
                if is_zero_traj and not no_noise:
                    shutil.move(src, os.path.join(processed_dir, fname))
                    print(f"[post_run] Saved processed/{fname}")
                else:
                    os.remove(src)

    # Render (optional)
    if render:
        video_path = os.path.join(raw_dir, "video.avi")
        if not os.path.exists(video_path):
            video_path = os.path.join(raw_dir, "video.mp4")
        if os.path.exists(video_path):
            env_type = cfg.get("env_type", "cpg_14d")
            env_config_path = os.path.join(
                _REPO_ROOT, "deploy", "configs", f"{env_type}.json"
            )
            subprocess.run(
                [sys.executable,
                 os.path.join(_REPO_ROOT, "deploy", "laptop", "render_run.py"),
                 "--session", raw_dir,
                 "--env-config", env_config_path],
                check=False,
            )
            # Move replay.mp4 from raw/ to processed/
            src = os.path.join(raw_dir, "replay.mp4")
            if os.path.exists(src):
                shutil.move(src, os.path.join(processed_dir, "replay.mp4"))
                print("[post_run] Saved processed/replay.mp4")
        else:
            print("[post_run] --render requested but no video found — skipping.")

    print(f"[post_run] Done. Outputs in {processed_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Post-run analysis for RL deployment runs")
    parser.add_argument("run_dir", help="Path to run directory (data/deploy_runs/YYYYMMDD_name)")
    parser.add_argument("--render", action="store_true", help="Generate replay.mp4 (~2 min)")
    parser.add_argument("--no-noise", action="store_true", help="Skip noise_config/report generation")
    parser.add_argument("--all", action="store_true", help="All outputs including render")
    args = parser.parse_args()
    run_analysis(
        run_dir=args.run_dir,
        render=args.render or args.all,
        no_noise=args.no_noise,
    )


if __name__ == "__main__":
    main()
