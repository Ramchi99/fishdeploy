"""Canonical per-segment data schema for FluidID.

One CSV per accepted segment, resampled to sim dt (1 kHz). The optimizer reads
ONLY this schema. Velocity columns (vx, vy, omega) are derived and used for
validation metrics only, never in the training loss.
"""
from __future__ import annotations
import pandas as pd

SIM_DT = 0.001  # seconds; canonical resample step (matches MuJoCo sim dt)

CANONICAL_COLUMNS = [
    "time",            # seconds from segment start
    "motor_pos_meas",  # measured tail-motor angle [rad] (teleport target)
    "motor_vel",       # motor angular velocity [rad/s]
    "pos_x",           # fish x in world/anchor frame [m], correctly scaled
    "pos_y",           # fish y [m]
    "yaw",             # heading [rad]
    "vx",              # derived world vx [m/s] — VALIDATION ONLY
    "vy",              # derived world vy [m/s] — VALIDATION ONLY
    "omega",           # derived yaw rate [rad/s] — VALIDATION ONLY
    "gyro_z",          # measured yaw rate [rad/s] (validation metric C)
    "source",          # "rl_deploy" | "field_test"
    "scale_applied",   # translation scale factor used (1.0 or 0.8)
    "starts_at_rest",  # bool
]

VALID_SOURCES = {"rl_deploy", "field_test"}


def validate_canonical(df: pd.DataFrame) -> None:
    """Raise ValueError if df does not conform to the canonical schema."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"canonical frame missing columns: {missing}")
    bad_sources = set(df["source"].unique()) - VALID_SOURCES
    if bad_sources:
        raise ValueError(f"invalid source values: {bad_sources}")
    if len(df) and not df["time"].is_monotonic_increasing:
        raise ValueError("time column must be monotonically increasing")
