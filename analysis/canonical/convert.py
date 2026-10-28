"""Build canonical per-segment CSVs from either data source.

field_test: read 03_field2sim_ready.csv (already 1 kHz, unified clock), scale
translations by `scale` (0.8 tag correction), slice [t_start,t_end], re-zero time.

rl_deploy: (Task 6) pose from AprilTag re-detection (already scaled), motor from
obs in run.csv, resampled to a common 1 kHz grid.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from . import schema
from . import pose_extraction


def _grid(t_start: float, t_end: float, sim_dt: float) -> np.ndarray:
    n = int(round((t_end - t_start) / sim_dt)) + 1
    return t_start + np.arange(n) * sim_dt


def _interp(src_t, src_v, grid):
    f = interp1d(src_t, src_v, kind="linear", bounds_error=False,
                 fill_value=(src_v[0], src_v[-1]))
    return f(grid)


def field_test_video_offset(run_dir: str) -> float:
    """Seconds between field_test data `Time`=0 (motor/IMU start) and the review
    video's first frame, i.e. ``data_Time = video.currentTime - offset``.

    Both clocks are Unix (the field_test "perfect offset"): the motor start is
    ``02_fish_log_trimmed.csv['Unix_Time'][0]`` and the video start is
    ``01_frame_times_unix.csv['Unix_Time'].min()``. Curation marks are made on the
    video axis; the review tool subtracts this so saved marks are in data `Time`
    (what ``field_test_segment`` slices). Returns 0.0 if either file is missing.
    """
    proc = os.path.join(run_dir, "processed")
    fish = os.path.join(proc, "02_fish_log_trimmed.csv")
    frames = os.path.join(proc, "01_frame_times_unix.csv")
    if not (os.path.isfile(fish) and os.path.isfile(frames)):
        return 0.0
    motor_t0 = float(pd.read_csv(fish)["Unix_Time"].iloc[0])
    vmin = float(pd.read_csv(frames)["Unix_Time"].min())
    return motor_t0 - vmin


def field_test_segment(run_dir: str, t_start: float, t_end: float,
                       scale: float, sim_dt: float, starts_at_rest: bool) -> pd.DataFrame:
    raw = pd.read_csv(os.path.join(run_dir, "processed", "03_field2sim_ready.csv"))
    t = raw["Time"].to_numpy()
    grid = _grid(t_start, t_end, sim_dt)

    pos_x = _interp(t, raw["Ghost_X"].to_numpy(), grid) * scale
    pos_y = _interp(t, raw["Ghost_Y"].to_numpy(), grid) * scale
    yaw = _interp(t, np.unwrap(raw["Ghost_Yaw"].to_numpy()), grid)
    motor = _interp(t, raw["Ghost_Alpha"].to_numpy(), grid)
    motor_vel = np.gradient(motor, sim_dt)
    vx = np.gradient(pos_x, sim_dt)
    vy = np.gradient(pos_y, sim_dt)
    omega = np.gradient(yaw, sim_dt)
    gyro_z = _interp(t, raw["GyrZ"].to_numpy(), grid) if "GyrZ" in raw else omega

    out = pd.DataFrame({
        "time": grid - t_start,
        "motor_pos_meas": motor, "motor_vel": motor_vel,
        "pos_x": pos_x, "pos_y": pos_y, "yaw": yaw,
        "vx": vx, "vy": vy, "omega": omega, "gyro_z": gyro_z,
        "source": "field_test", "scale_applied": scale,
        "starts_at_rest": starts_at_rest,
    })
    schema.validate_canonical(out)
    return out


def build_segment(seg: dict, cfg: dict) -> pd.DataFrame:
    """Build one canonical DataFrame for an accepted manifest segment."""
    conv = cfg["conversion"]
    if seg["source"] == "field_test":
        run_dir = os.path.join(cfg["paths"]["field_test_runs"], seg["run_id"])
        return field_test_segment(
            run_dir, seg["t_start"], seg["t_end"],
            scale=conv["field_test_scale"], sim_dt=conv["sim_dt"],
            starts_at_rest=seg["starts_at_rest"])
    if seg["source"] == "rl_deploy":
        run_dir = os.path.join(cfg["paths"]["rl_deploy_runs"], seg["run_id"])
        return rl_deploy_segment(
            run_dir, seg["t_start"], seg["t_end"], sim_dt=conv["sim_dt"],
            tag_size_m=cfg["apriltag"]["tag_size_m"],
            starts_at_rest=seg["starts_at_rest"])
    raise ValueError(f"unknown source {seg['source']!r}")


def rl_deploy_segment(run_dir: str, t_start: float, t_end: float, sim_dt: float,
                      tag_size_m: float, starts_at_rest: bool,
                      traj: dict | None = None) -> pd.DataFrame:
    """Pose from AprilTag re-detection (frame clock); motor from obs (run.csv clock).
    Both carry laptop_time; we put everything on the laptop_time master, zeroed to
    the first video frame, then resample to a 1 kHz grid over [t_start, t_end].

    `traj` may be passed in to avoid re-decoding the video (review cache reuse).
    """
    if traj is None:
        traj = pose_extraction.extract_trajectory(run_dir, tag_size_m=tag_size_m)

    ft = np.asarray(traj["frame_time"])           # already zeroed to first frame
    det = np.asarray(traj["detected"], dtype=bool)
    if det.sum() < 2:
        raise ValueError(
            f"rl_deploy_segment: fewer than 2 detected pose frames in {run_dir} "
            f"({int(det.sum())} detected of {det.size}); cannot interpolate trajectory")
    px = np.asarray(traj["pos_x"]); py = np.asarray(traj["pos_y"])
    yw = np.asarray(traj["yaw"])

    # Use only detected frames for pose interpolation (drop NaN gaps).
    ftd, pxd, pyd = ft[det], px[det], py[det]
    ywd = np.unwrap(yw[det])

    # Motor + gyro from run.csv on the Pi clock, converted to laptop-clock epoch
    # then zeroed to the first video frame.
    # pi_time + clock_offset_s gives the Pi sample time on the laptop clock.
    run = pd.read_csv(os.path.join(run_dir, "raw", "run.csv"))
    off = pose_extraction.clock_offset_s(run_dir)
    fts = pd.read_csv(os.path.join(run_dir, "raw", "frame_timestamps.csv"))
    t0_abs = fts["laptop_time"].min()
    run_t = run["pi_time"].to_numpy() + off - t0_abs

    motor = np.arctan2(run["obs_01"].to_numpy(), run["obs_00"].to_numpy())
    motor_vel_raw = run["obs_02"].to_numpy()
    gyro = run["obs_03"].to_numpy()  # imu_wz yaw rate (cpg_24d layout: obs[3])

    grid = _grid(t_start, t_end, sim_dt)
    pos_x = _interp(ftd, pxd, grid)
    pos_y = _interp(ftd, pyd, grid)
    yaw = _interp(ftd, ywd, grid)
    motor_i = _interp(run_t, motor, grid)
    motor_vel = _interp(run_t, motor_vel_raw, grid)
    gyro_z = _interp(run_t, gyro, grid)

    dt = sim_dt
    vx = np.gradient(pos_x, dt)
    vy = np.gradient(pos_y, dt)
    omega = np.gradient(yaw, dt)

    out = pd.DataFrame({
        "time": grid - t_start,
        "motor_pos_meas": motor_i, "motor_vel": motor_vel,
        "pos_x": pos_x, "pos_y": pos_y, "yaw": yaw,
        "vx": vx, "vy": vy, "omega": omega, "gyro_z": gyro_z,
        "source": "rl_deploy", "scale_applied": 1.0,
        "starts_at_rest": starts_at_rest,
    })
    schema.validate_canonical(out)
    return out
