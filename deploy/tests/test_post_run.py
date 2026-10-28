import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd
import pytest


def _write_shadow_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = {"mode": "shadow", "trajectory": "zero", "clock_offset_ms": 0.0}
    (raw / "run_config.json").write_text(json.dumps(cfg))
    n = 100
    df = pd.DataFrame({
        "pi_time": np.linspace(0, 2, n),
        "laptop_time": np.linspace(0, 2, n),
        **{f"obs_{i:02d}": np.zeros(n) for i in range(14)},
        "trajectory_cmd": np.zeros(n),
        "vision_age_ms": np.full(n, 50.0),
        "vision_flags": np.ones(n, dtype=int),
        "frame_seq": np.arange(n, dtype=int),
    })
    df["obs_00"] = 1.0  # cos(0)=1 → motor_pos = arctan2(0,1) = 0
    df.to_csv(raw / "run.csv", index=False)
    return tmp_path


def _write_rl_run(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = {"mode": "rl", "clock_offset_ms": 0.0}
    (raw / "run_config.json").write_text(json.dumps(cfg))
    n = 100
    df = pd.DataFrame({
        "pi_time": np.linspace(0, 2, n),
        "laptop_time": np.linspace(0, 2, n),
        **{f"obs_{i:02d}": np.zeros(n) for i in range(14)},
        **{f"act_{i:02d}": np.zeros(n) for i in range(6)},
        "motor_setpoint": np.zeros(n),
        "reward": np.full(n, -1.0),
        "done": np.zeros(n, dtype=int),
        "vision_age_ms": np.full(n, 50.0),
        "vision_flags": np.ones(n, dtype=int),
        "frame_seq": np.arange(n, dtype=int),
        "cpg_phase": np.linspace(0, 6.28, n),
    })
    df["obs_00"] = 1.0
    df.to_csv(raw / "run.csv", index=False)
    return tmp_path


def test_post_run_shadow_zero_traj(tmp_path):
    run_dir = _write_shadow_run(tmp_path)
    from deploy.laptop.post_run import run_analysis
    run_analysis(str(run_dir))
    proc = run_dir / "processed"
    assert (proc / "summary.png").exists()
    assert (proc / "diagnostics.png").exists()
    assert (proc / "noise_config.json").exists()
    assert (proc / "noise_report.md").exists()
    # raw/ should NOT have analysis artifacts after post_run
    assert not (run_dir / "raw" / "diagnostics.png").exists()
    assert not (run_dir / "raw" / "noise_config.json").exists()


def test_post_run_rl_mode(tmp_path):
    run_dir = _write_rl_run(tmp_path)
    from deploy.laptop.post_run import run_analysis
    run_analysis(str(run_dir))
    proc = run_dir / "processed"
    assert (proc / "summary.png").exists()
    assert (proc / "diagnostics.png").exists()
    assert not (proc / "noise_config.json").exists()  # RL runs don't produce noise files


def test_post_run_no_noise_flag(tmp_path):
    run_dir = _write_shadow_run(tmp_path)
    from deploy.laptop.post_run import run_analysis
    run_analysis(str(run_dir), no_noise=True)
    proc = run_dir / "processed"
    assert (proc / "summary.png").exists()
    assert (proc / "diagnostics.png").exists()
    assert not (proc / "noise_config.json").exists()
