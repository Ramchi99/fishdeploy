import csv
import json
import os
import time

import numpy as np


class OpenLoopLogger:
    """Logs open-loop and shadow-mode runs.

    run.csv always contains:
      pi_time, laptop_time, obs_00..obs_13, trajectory_cmd,
      vision_age_ms, vision_flags, frame_seq

    Shadow mode appends after trajectory_cmd:
      policy_act_raw_00..05, policy_act_scaled_00..05, policy_motor_setpoint
    """

    _OBS_DIM = 14
    _ACT_DIM = 6

    def __init__(
        self,
        run_dir: str,
        config: dict,
        trajectory_name: str,
        trajectory_params: dict,
        shadow: bool,
        record: bool,
        git_hash: str,
        pi_ip: str,
        clock_offset_ms: float,
        model_hash: str = "",
    ):
        os.makedirs(run_dir, exist_ok=True)
        self._run_dir = run_dir
        self._shadow = shadow
        self._rows: list = []

        cfg = {
            "mode": "shadow" if shadow else "open_loop",
            "trajectory": trajectory_name,
            **trajectory_params,
            "model_hash": model_hash,
            "git_hash": git_hash,
            "pi_ip": pi_ip,
            "clock_offset_ms": clock_offset_ms,
            "record": record,
            "start_time": time.time(),
        }
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        obs_cols = [f"obs_{i:02d}" for i in range(self._OBS_DIM)]
        shadow_cols = (
            [f"policy_act_raw_{i:02d}" for i in range(self._ACT_DIM)]
            + [f"policy_act_scaled_{i:02d}" for i in range(self._ACT_DIM)]
            + ["policy_motor_setpoint"]
        ) if shadow else []

        self._cols = (
            ["pi_time", "laptop_time"]
            + obs_cols
            + ["trajectory_cmd"]
            + shadow_cols
            + ["vision_age_ms", "vision_flags", "frame_seq"]
        )

        # Index of the first shadow column (used in summary PNG)
        self._shadow_col_start = 2 + self._OBS_DIM + 1  # after pi_time,laptop_time,obs,traj_cmd

        run_path = os.path.join(run_dir, "run.csv")
        self._f = open(run_path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(self._cols)

    def log_step(
        self,
        pi_time: float,
        laptop_time: float,
        obs: np.ndarray,
        trajectory_cmd: float,
        vision_age_ms: float,
        vision_flags: int,
        frame_seq: int,
        policy_act_raw: np.ndarray = None,
        policy_act_scaled: np.ndarray = None,
        policy_motor_setpoint: float = None,
    ):
        if self._shadow:
            shadow_vals = (
                list(policy_act_raw)
                + list(policy_act_scaled)
                + [policy_motor_setpoint if policy_motor_setpoint is not None else 0.0]
            )
        else:
            shadow_vals = []

        row = (
            [pi_time, laptop_time]
            + obs.tolist()
            + [trajectory_cmd]
            + shadow_vals
            + [vision_age_ms, vision_flags, frame_seq]
        )
        self._writer.writerow(row)
        self._rows.append(row)

    def close(self):
        self._f.flush()
        self._f.close()
        self._write_summary_png()

    def _write_summary_png(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if not self._rows:
                return

            obs_start = 2  # after pi_time, laptop_time
            traj_col = obs_start + self._OBS_DIM

            pi_times = [r[0] for r in self._rows]
            t = [p - pi_times[0] for p in pi_times]
            traj_cmds = [r[traj_col] for r in self._rows]
            motor_pos = [np.arctan2(r[obs_start + 1], r[obs_start + 0]) for r in self._rows]
            dists = [r[obs_start + 6] for r in self._rows]

            # vision_age_ms is after shadow cols
            n_shadow_cols = (self._ACT_DIM * 2 + 1) if self._shadow else 0
            vision_age_col = traj_col + 1 + n_shadow_cols
            vision_ages = [r[vision_age_col] for r in self._rows]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0, 0].plot(t, traj_cmds, lw=1, label="setpoint")
            axes[0, 0].plot(t, motor_pos, lw=0.7, alpha=0.7, label="encoder")
            if self._shadow:
                # policy_motor_setpoint is last shadow col
                pol_setpoint_col = traj_col + 1 + self._ACT_DIM * 2
                pol_setpoints = [r[pol_setpoint_col] for r in self._rows]
                axes[0, 0].plot(t, pol_setpoints, lw=1, ls="--", label="policy cmd")
            axes[0, 0].set_xlabel("Time [s]")
            axes[0, 0].set_ylabel("rad")
            axes[0, 0].set_title("Motor Setpoint vs Encoder")
            axes[0, 0].legend()

            axes[0, 1].plot(t, dists, lw=1, color="tab:blue")
            axes[0, 1].set_xlabel("Time [s]")
            axes[0, 1].set_ylabel("dist [m]")
            axes[0, 1].set_title("Distance to Target")

            axes[1, 0].hist(vision_ages, bins=50, color="tab:green")
            axes[1, 0].set_xlabel("Vision age [ms]")
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Vision Latency Distribution")

            if self._shadow:
                # policy omega raw (dim 4 of raw action, col = shadow_col_start + 4)
                omega_col = self._shadow_col_start + 4
                policy_omega = [r[omega_col] for r in self._rows]
                axes[1, 1].plot(t, policy_omega, lw=1, color="tab:orange")
                axes[1, 1].set_xlabel("Time [s]")
                axes[1, 1].set_ylabel("omega_raw [-1, 1]")
                axes[1, 1].set_title("Shadow: Policy Omega Output")
            else:
                imu_wz = [r[obs_start + 5] for r in self._rows]
                axes[1, 1].plot(t, imu_wz, lw=1, color="tab:red")
                axes[1, 1].set_xlabel("Time [s]")
                axes[1, 1].set_ylabel("wz [rad/s]")
                axes[1, 1].set_title("IMU Angular Velocity (wz)")

            fig.tight_layout()
            fig.savefig(os.path.join(self._run_dir, "summary.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[OpenLoopLogger] summary.png failed: {e}")
