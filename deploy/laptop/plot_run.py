"""
Post-session visualization from run.csv.

Usage:
    python deploy/laptop/plot_run.py deploy/data/20260505_143022_pool_test_1
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_run(run_dir: str, df: pd.DataFrame = None):
    if df is None:
        csv_path = os.path.join(run_dir, "run.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"run.csv not found in {run_dir}")
        df = pd.read_csv(csv_path)

    t = df["pi_time"].to_numpy() - df["pi_time"].iloc[0]

    obs_cols   = [f"obs_{i:02d}" for i in range(14)]
    act_cols   = [f"act_{i:02d}" for i in range(6)]
    fish_x_proxy = df["obs_07"].to_numpy()   # vec_target_x
    dist = df["obs_06"].to_numpy()
    rewards = df["reward"].to_numpy()
    cum_reward = np.cumsum(rewards)
    vision_ages = df["vision_age_ms"].to_numpy()
    motor_setpt = df["motor_setpoint"].to_numpy()
    cpg_phase   = df["cpg_phase"].to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Run: {os.path.basename(run_dir)}", fontsize=12)

    axes[0, 0].plot(t, dist, lw=1, color="tab:blue")
    axes[0, 0].axhline(0.05, ls="--", lw=0.8, color="tab:red", label="success (0.05 m)")
    axes[0, 0].set_xlabel("Time [s]"); axes[0, 0].set_ylabel("dist_target [m]")
    axes[0, 0].set_title("Distance to Target"); axes[0, 0].legend()

    ax2 = axes[0, 1]
    ax2b = ax2.twinx()
    ax2.plot(t, cum_reward, lw=1, color="tab:orange", label="cumulative")
    ax2b.plot(t, rewards, lw=0.5, alpha=0.4, color="tab:gray", label="per-step")
    ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Cumulative reward", color="tab:orange")
    ax2b.set_ylabel("Per-step reward", color="tab:gray")
    ax2.set_title("Reward")

    axes[1, 0].hist(vision_ages, bins=60, color="tab:green", edgecolor="none")
    axes[1, 0].axvline(100, ls="--", lw=0.8, color="orange", label="warn 100ms")
    axes[1, 0].axvline(500, ls="--", lw=0.8, color="red",    label="stop 500ms")
    axes[1, 0].set_xlabel("Vision age [ms]"); axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Vision Latency Distribution"); axes[1, 0].legend()

    axes[1, 1].plot(t, motor_setpt, lw=1, label="setpoint [rad]")
    axes[1, 1].plot(t, np.cos(cpg_phase) * 0.5, lw=0.7, alpha=0.6, label="cos(cpg_phase)×0.5")
    axes[1, 1].set_xlabel("Time [s]"); axes[1, 1].set_ylabel("rad")
    axes[1, 1].set_title("Motor Setpoint & CPG Phase"); axes[1, 1].legend()

    fig.tight_layout()
    out_path = os.path.join(run_dir, "summary.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to run data directory")
    args = parser.parse_args()
    plot_run(args.run_dir)
