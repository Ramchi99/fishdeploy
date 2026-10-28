"""Convert Layer-2 curated parts → canonical 1 kHz CSVs.

Reads layer2_parts.yml. For each part calls rl_deploy_segment() / field_test_segment()
with per-run detection params (from data/detection/*.json), the session global anchor
(data/anchor_calibration.json), and applies any motor-zero correction
(data/motor_zeros.json). Writes to data/02_deploy_20260520/canonical/.

The raw videos are in the archive, not in this repository: copy a run's raw/video.avi next to its
run.csv before converting it. The curation itself (choosing the segments) was done interactively in the
private fluidID workspace; the manifest records its result.

Usage (from the repository root):
  python analysis/canonical/convert_segments.py
  python analysis/canonical/convert_segments.py --num 6
  python analysis/canonical/convert_segments.py --num 6 --part 2
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from paths import CANONICAL, DEPLOY, FIELD                # noqa: E402
from analysis.canonical import convert as C               # noqa: E402
from analysis.canonical import curation as CUR            # noqa: E402
from analysis.canonical import pose_extraction as PE      # noqa: E402

# Conversion constants (were fluidID/configs/fluidID.yml)
_SIM_DT = 0.001              # 1 kHz canonical grid
_TAG_SIZE_M = 0.080          # physical AprilTag side length
_FIELD_TEST_SCALE = 0.8      # 0.08 / 0.10 translation correction of the field-test pipeline


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _det_params(slug: str, det_dir: str) -> dict | None:
    """Per-run detection params written by 11_detect.py. None if not yet run."""
    p = os.path.join(det_dir, f"{slug}.json")
    if not os.path.isfile(p):
        return None
    d = _load_json(p)
    return {k: d[k] for k in ("alpha", "beta", "blur", "quad_decimate") if k in d}


def _resolve_motor_zero(run_id: str, saved: dict, computed: dict) -> float:
    """Motor-zero offset to subtract from encoder readings.

    Priority:
      1. motor_zeros.json (plural) — explicitly saved in the layer-2 UI.
      2. motor_zero.json  (singular) — auto-computed; used only when reliable=True.
      3. 0.0 — no correction.
    """
    if run_id in saved:
        return float(saved[run_id].get("value", 0.0))
    entry = computed.get(run_id, {})
    if entry.get("reliable") and entry.get("recommended") is not None:
        return float(entry["recommended"])
    return 0.0


def _apply_coord_correction(df) -> None:
    """Convert camera-frame detection output to MuJoCo convention in-place.

    The camera AprilTag system has Z pointing INTO the pool (downward); MuJoCo has
    Z pointing UP. This flips Y and negates rotation sense. The fish model also has
    its nose at −X (headRollZ=0), whereas extract_yaw returns 0 when the tag Y-axis
    points in the camera +X direction.

    Combined transform for rl_deploy data:
      pos_y  →  −pos_y
      vy     →  −vy
      yaw    →  π − yaw_raw     (negate rotation sense + nose-at-−X offset)
      omega  →  −omega           (d(π−yaw)/dt = −omega_raw)
      gyro_z →  −gyro_z          (Z-axis convention flip)
    pos_x, motor_pos_meas, motor_vel are unchanged.
    """
    df["pos_y"]  = -df["pos_y"]
    df["vy"]     = -df["vy"]
    df["yaw"]    = np.pi - df["yaw"]
    df["omega"]  = -df["omega"]
    df["gyro_z"] = -df["gyro_z"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num",  type=int, default=None,
                    help="Only convert run with this num (from layer2_parts.yml)")
    ap.add_argument("--part", type=int, default=None,
                    help="Only convert this part number (requires --num)")
    args = ap.parse_args()

    data_root        = str(CANONICAL)
    det_dir          = os.path.join(data_root, "detection")
    anchor_path      = os.path.join(data_root, "anchor_calibration.json")
    mz_saved_path    = os.path.join(data_root, "motor_zeros.json")   # UI-saved (plural)
    mz_computed_path = os.path.join(data_root, "motor_zero.json")    # auto-computed (singular)
    out_dir          = str(CANONICAL)
    os.makedirs(out_dir, exist_ok=True)

    # Session anchor (fallback only): used when per-run detection yields < 50% of frames.
    # 11_detect.py also used per-run anchor as primary, falling back to session anchor.
    session_anchor = _load_json(anchor_path) if os.path.isfile(anchor_path) else None
    mz_saved    = _load_json(mz_saved_path)    if os.path.isfile(mz_saved_path)    else {}
    mz_computed = _load_json(mz_computed_path) if os.path.isfile(mz_computed_path) else {}

    l2 = CUR.load_layer2(os.path.join(data_root, "layer2_parts.yml"))
    if args.num is not None:
        l2 = [e for e in l2 if e["num"] == args.num]
    if not l2:
        print("No matching entries found in layer2_parts.yml.")
        return

    sim_dt   = _SIM_DT
    tag_size = _TAG_SIZE_M
    rl_root  = str(DEPLOY)
    ft_root  = str(FIELD)

    for entry in l2:
        run_id = entry["run_id"]
        source = entry["source"]
        slug   = CUR.run_slug(entry["num"], entry["name"])
        parts  = entry["parts"]
        if args.part is not None:
            parts = [p for p in parts if p["part"] == args.part]
        if not parts:
            continue

        # Re-detect AprilTags at true variable frame rate using saved per-run params.
        # Per-run anchor (global_anchor=None): anchor detected fresh from each run's video,
        # giving maximum pose accuracy. The 30fps layer2 cache is NOT used here.
        traj = None
        if source == "rl_deploy":
            run_dir = os.path.join(rl_root, run_id)
            params  = _det_params(slug, det_dir)
            print(f"  [{slug}] detecting (per-run anchor, params={params}) …", flush=True)
            traj = PE.extract_trajectory(run_dir, tag_size_m=tag_size, params=params)
            n_det   = sum(traj["detected"])
            n_total = len(traj["detected"])
            pct     = 100 * n_det / n_total if n_total else 0
            print(f"  [{slug}] detected {n_det}/{n_total} frames ({pct:.0f}%)", end="")

            # Fallback: if per-run anchor barely detected anything, retry with
            # the session-wide anchor (mirrors 11_detect.py Phase-1 fallback logic).
            if pct < 50 and session_anchor is not None:
                print(f" ← LOW — retrying with session anchor …", flush=True)
                traj = PE.extract_trajectory(run_dir, tag_size_m=tag_size,
                                             params=params, global_anchor=session_anchor)
                n_det   = sum(traj["detected"])
                n_total = len(traj["detected"])
                pct     = 100 * n_det / n_total if n_total else 0
                print(f"  [{slug}] session anchor: {n_det}/{n_total} ({pct:.0f}%)", end="")
            print()

        mz_val = _resolve_motor_zero(run_id, mz_saved, mz_computed)
        mz_src = ("UI-saved" if run_id in mz_saved
                  else "computed-reliable" if mz_val != 0.0 else "none")
        if mz_val != 0.0:
            print(f"  [{slug}] motor_zero: {mz_val:+.4f} rad ({mz_src})")

        for p in parts:
            pid      = CUR.part_id(entry["num"], entry["name"], p["part"])
            out_path = os.path.join(out_dir, f"{pid}.csv")
            try:
                if source == "rl_deploy":
                    df = C.rl_deploy_segment(
                        run_dir, p["t_start"], p["t_end"],
                        sim_dt=sim_dt, tag_size_m=tag_size,
                        starts_at_rest=p["starts_at_rest"], traj=traj)
                else:
                    run_dir_ft = os.path.join(ft_root, run_id)
                    df = C.field_test_segment(
                        run_dir_ft, p["t_start"], p["t_end"],
                        scale=_FIELD_TEST_SCALE,
                        sim_dt=sim_dt, starts_at_rest=p["starts_at_rest"])

                if mz_val != 0.0:
                    df["motor_pos_meas"] = df["motor_pos_meas"] - mz_val

                # Camera Z points into pool; MuJoCo Z points up → Y flip + sign fix.
                if source == "rl_deploy":
                    _apply_coord_correction(df)

                df.to_csv(out_path, index=False)
                n = len(df)
                print(f"  → {pid}: {n} rows  rest={p['starts_at_rest']}  "
                      f"t=[{p['t_start']:.3f},{p['t_end']:.3f}]s  → {out_path}")
            except Exception as exc:
                print(f"  ERROR [{pid}]: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
