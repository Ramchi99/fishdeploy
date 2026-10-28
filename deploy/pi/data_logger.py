import csv
import json
import os
import time

import numpy as np


_OBS_DIM_BY_ENV = {"cpg_12d": 12, "cpg_14d": 14, "cpg_24d": 24, "da_rl_policy_40d": 40}


class DataLogger:
    """Writes run.csv, run_config.json, replay_buffer.csv (optional), summary.png (on close).

    run.csv columns:
      pi_time, laptop_time, obs_00..obs_<N-1>, act_00..act_05,
      motor_setpoint, reward, done, vision_age_ms, vision_flags, frame_seq, cpg_phase
    where N = obs_dim derived from env_type in config.
    """

    def __init__(
        self,
        run_dir: str,
        config: dict,
        record: bool,
        finetune: bool,
        model_hash: str,
        git_hash: str,
        pi_ip: str,
        clock_offset_ms: float,
    ):
        os.makedirs(run_dir, exist_ok=True)
        self._run_dir = run_dir
        self._finetune = finetune
        self._record = record
        self._rows: list = []
        self._rb_rows: list = []
        self._prev_obs: np.ndarray = None
        self._obs_dim = _OBS_DIM_BY_ENV.get(config.get("env_type", "cpg_14d"), 14)
        self._action_dim = int(config.get("action_dim", 6))

        cfg = {
            "env_type": config.get("env_type"),
            "model_path": config.get("model_path"),
            "model_hash": model_hash,
            "git_hash": git_hash,
            "pi_ip": pi_ip,
            "clock_offset_ms": clock_offset_ms,
            "record": record,
            "finetune": finetune,
            "start_time": time.time(),
        }
        with open(os.path.join(run_dir, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        obs_cols = [f"obs_{i:02d}" for i in range(self._obs_dim)]
        act_cols = [f"act_{i:02d}" for i in range(self._action_dim)]
        self._run_cols = (
            ["pi_time", "laptop_time"]
            + obs_cols + act_cols
            + ["motor_setpoint", "reward", "done",
               "vision_age_ms", "vision_flags", "frame_seq", "cpg_phase", "motor_load"]
        )
        run_path = os.path.join(run_dir, "run.csv")
        self._run_f = open(run_path, "w", newline="")
        self._run_writer = csv.writer(self._run_f)
        self._run_writer.writerow(self._run_cols)

        if finetune:
            s_cols = [f"s_{i:02d}" for i in range(self._obs_dim)]
            sp_cols = [f"sp_{i:02d}" for i in range(self._obs_dim)]
            a_cols = [f"a_{i:02d}" for i in range(6)]
            self._rb_cols = s_cols + a_cols + ["reward", "done"] + sp_cols
            rb_path = os.path.join(run_dir, "replay_buffer.csv")
            self._rb_f = open(rb_path, "w", newline="")
            self._rb_writer = csv.writer(self._rb_f)
            self._rb_writer.writerow(self._rb_cols)
        else:
            self._rb_f = None
            self._rb_writer = None

    def log_step(
        self,
        pi_time: float,
        laptop_time: float,
        obs: np.ndarray,
        action: np.ndarray,
        motor_setpoint: float,
        reward: float,
        done: bool,
        vision_age_ms: float,
        vision_flags: int,
        frame_seq: int,
        cpg_phase: float,
        motor_load: int = 0,
    ):
        row = (
            [pi_time, laptop_time]
            + obs.tolist()
            + action.tolist()
            + [motor_setpoint, reward, int(done),
               vision_age_ms, vision_flags, frame_seq, cpg_phase, motor_load]
        )
        self._run_writer.writerow(row)
        self._rows.append(row)

        if self._rb_writer is not None and self._prev_obs is not None:
            rb_row = (
                self._prev_obs.tolist()
                + action.tolist()
                + [reward, int(done)]
                + obs.tolist()
            )
            self._rb_writer.writerow(rb_row)

        self._prev_obs = obs.copy()

    def close(self):
        self._run_f.flush()
        self._run_f.close()
        if self._rb_f is not None:
            self._rb_f.flush()
            self._rb_f.close()
        self._write_summary_png()

    def _write_summary_png(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np

            if not self._rows:
                return

            pi_times = [r[0] for r in self._rows]
            obs_start = 2
            n = self._obs_dim
            rewards         = [r[obs_start + n + 6 + 1] for r in self._rows]
            dists           = [r[obs_start + 6]          for r in self._rows]
            vision_ages     = [r[obs_start + n + 6 + 3]  for r in self._rows]
            motor_setpoints = [r[obs_start + n + 6]       for r in self._rows]
            cpg_phases      = [r[-1]                       for r in self._rows]
            # Recover motor encoder position from cos/sin obs
            motor_pos = [np.arctan2(r[obs_start + 1], r[obs_start]) for r in self._rows]

            t = [p - pi_times[0] for p in pi_times]
            cum_reward = [sum(rewards[:i+1]) for i in range(len(rewards))]

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            axes[0, 0].plot(t, dists, lw=1)
            axes[0, 0].set_xlabel("Time [s]"); axes[0, 0].set_ylabel("dist_target [m]")
            axes[0, 0].set_title("Distance to Target")

            axes[0, 1].plot(t, cum_reward, lw=1, color="tab:orange")
            axes[0, 1].set_xlabel("Time [s]"); axes[0, 1].set_ylabel("Cumulative reward")
            axes[0, 1].set_title("Cumulative Reward")

            axes[1, 0].hist(vision_ages, bins=50, color="tab:green")
            axes[1, 0].axvline(200, ls="--", lw=1, color="gold", label="stale 200ms")
            axes[1, 0].axvline(500, ls="--", lw=1, color="red",  label="lost 500ms")
            axes[1, 0].set_xlabel("Vision age [ms]"); axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Vision Latency Distribution")
            axes[1, 0].legend(fontsize=8)

            # Motor tracking: setpoint (blue) and encoder (red) share the left axis;
            # CPG phase (grey) on a secondary right axis (different scale: 0–2π).
            ax = axes[1, 1]
            ax.plot(t, motor_setpoints, lw=1,   color="steelblue", label="setpoint")
            ax.plot(t, motor_pos,       lw=0.7,  color="tomato",    label="encoder", alpha=0.8)
            ax.set_xlabel("Time [s]"); ax.set_ylabel("Motor position [rad]")
            ax.set_title("Motor Tracking & CPG Phase")
            ax.legend(fontsize=8, loc="upper left")
            ax2 = ax.twinx()
            ax2.plot(t, cpg_phases, lw=0.6, color="grey", alpha=0.5, label="CPG phase")
            ax2.set_ylabel("CPG phase [rad]", color="grey")
            ax2.set_ylim(0, 2 * np.pi)
            ax2.tick_params(axis='y', labelcolor="grey")
            ax2.legend(fontsize=8, loc="upper right")

            fig.tight_layout()
            fig.savefig(os.path.join(self._run_dir, "summary.png"), dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[DataLogger] summary.png generation failed: {e}")
