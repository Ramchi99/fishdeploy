import pandas as pd
import numpy as np
import os
import sys
from scipy.interpolate import CubicSpline


def prep_field2sim(run_dir, pulley_radius=0.0175, yaw_source='vision', yaw_blend_alpha=0.7):
    """
    Prepare field data for Field-to-Sim replay.

    yaw_source : 'vision' | 'imu' | 'blend'
        'vision' – AprilTag Yaw only (globally referenced, ~30 Hz).
        'imu'    – BNO055 RawY zeroed at first sample (~35-55 Hz, gyro-dominated
                   at 1 Hz so less affected by magnetometer drift/interference).
        'blend'  – weighted average: (1-alpha)*vision + alpha*imu,
                   where alpha = yaw_blend_alpha.
    yaw_blend_alpha : float in [0, 1]
        Weight on IMU when mode='blend'. 1.0 = full IMU, 0.0 = full Vision.
    """
    run_name = os.path.basename(run_dir)
    print(f"\n--- Field-to-Sim Data Preparation: {run_name} ---")
    print(f"    Pulley radius: {pulley_radius*1000:.1f} mm | Yaw source: '{yaw_source}'" +
          (f" (alpha={yaw_blend_alpha:.2f})" if yaw_source == 'blend' else ""))

    proc_dir = os.path.join(run_dir, "processed")
    vis_path = os.path.join(proc_dir, "02_clean_path.csv")
    imu_path = os.path.join(proc_dir, "02_fish_log_trimmed.csv")

    if not os.path.exists(vis_path) or not os.path.exists(imu_path):
        print(f"  ! Error: Required files not found in {proc_dir}")
        print("    Please run 02_process_swimming.py first.")
        return

    # 1. LOAD DATA
    df_vis = pd.read_csv(vis_path)
    df_imu = pd.read_csv(imu_path)

    # 2. ALIGN AND SHIFT TIME
    # Drop frames where AprilTag was not detected (NaN position/orientation).
    df_vis = df_vis.dropna(subset=['X', 'Y', 'Z', 'Yaw']).reset_index(drop=True)
    print(f"  - Vision: {len(df_vis)} valid detections after dropping NaN frames")

    # Per-run manual outlier corrections from vision_settings.json.
    # "drop_yaw_rows"      : 0-based row indices (post-dropna) where Yaw is bad but position is OK.
    # "drop_position_rows" : 0-based row indices (post-dropna) where X/Y/Z is bad but Yaw is OK.
    # Both null only the affected columns — the other column stays available for its own spline.
    import json
    _cfg_path = os.path.join(os.path.dirname(__file__), "..", "vision_settings.json")
    if os.path.exists(_cfg_path):
        with open(_cfg_path) as _f:
            _cfg = json.load(_f)
        _run_cfg = _cfg.get(run_name, {})
        _drop_yaw = _run_cfg.get("drop_yaw_rows", [])
        _drop_pos = _run_cfg.get("drop_position_rows", [])
        if _drop_yaw:
            df_vis.loc[_drop_yaw, 'Yaw'] = np.nan
            print(f"  - Nulled Yaw at rows: {_drop_yaw}")
        if _drop_pos:
            df_vis.loc[_drop_pos, ['X', 'Y', 'Z']] = np.nan
            print(f"  - Nulled Position at rows: {_drop_pos}")

    # Drop the last Vision sample — it is a known outlier spike in AprilTag tracking.
    if len(df_vis) > 1:
        df_vis = df_vis.iloc[:-1].copy()

    # t0 = motor/test start (first IMU sample), NOT the first AprilTag detection.
    # Using vision start would zero the motor encoder mid-swim, producing a spurious
    # constant offset in Ghost_Alpha when the tag appears after the motor has moved.
    t0 = df_imu['Unix_Time'].min()
    df_vis['T_rel'] = df_vis['Unix_Time'] - t0
    df_imu['T_rel'] = df_imu['Unix_Time'] - t0

    t_max = df_vis['T_rel'].max()
    df_imu = df_imu[(df_imu['T_rel'] >= 0) & (df_imu['T_rel'] <= t_max)].copy()

    # Cap t_max at the last IMU sample so the motor spline never extrapolates past its data.
    # Without this, CubicSpline (clamped BC) diverges quadratically beyond the last knot,
    # causing Ghost_dAlpha to shoot to 20-40 rad/s in the final milliseconds.
    t_max = df_imu['T_rel'].iloc[-1]

    # Split into separate position and yaw dataframes so partial nulls don't cross-contaminate.
    df_pos = df_vis.dropna(subset=['X', 'Y', 'Z']).copy()
    df_yaw = df_vis.dropna(subset=['Yaw']).copy()

    # 3. COORDINATE TRANSFORMS  (position dataframe only)
    # Vision Z (3m swim axis) → MuJoCo X
    # Vision X → MuJoCo Y, negated: World X = lateral LEFT, MuJoCo +Y = lateral LEFT,
    #   but the camera/world frame has X pointing in the opposite lateral direction vs
    #   MuJoCo's right-hand convention (confirmed by 04_plot_field_test.py which also
    #   negates World X for the correct pool-view plot).
    # Vision Y → MuJoCo Z (vertical; locked to 0 in sim)
    vis_z_shifted = df_pos['Z'].values - df_pos['Z'].iloc[0]
    vis_x_shifted = df_pos['X'].values - df_pos['X'].iloc[0]
    vis_y_shifted = df_pos['Y'].values - df_pos['Y'].iloc[0]

    mu_x_raw = vis_z_shifted
    mu_y_raw = -vis_x_shifted   # sign flip: matches 04_plot_field_test.py mu_y = -(World_X)
    mu_z_raw = vis_y_shifted

    # 4. YAW: compute from each source, then select  (yaw dataframe only)
    # --- Vision yaw (AprilTag) ---
    vis_yaw_rad       = np.radians(df_yaw['Yaw'].values)
    vis_yaw_unwrapped = np.unwrap(vis_yaw_rad)
    vis_yaw_zeroed    = vis_yaw_unwrapped - vis_yaw_unwrapped[0]  # zero at start

    # --- IMU yaw (BNO055 RawY, zeroed at first sample) ---
    # RawY is the fused heading (magnetometer+gyro). At 1 Hz flapping the gyro
    # dominates the fusion, so this is largely immune to magnetometer drift/EMI.
    # GyrZ is confirmed as the yaw-rate axis (largest range: ~3.8 rad/s).
    imu_yaw_deg       = df_imu['RawY'].values
    imu_yaw_rad       = np.radians(-(imu_yaw_deg - imu_yaw_deg[0]))  # negate: RawY decreases for CCW, MuJoCo needs positive=CCW
    imu_yaw_unwrapped = np.unwrap(imu_yaw_rad)

    # --- Select or blend ---
    if yaw_source == 'vision':
        mu_yaw_raw = vis_yaw_zeroed
        print(f"  - Vision Yaw amplitude: [{np.degrees(vis_yaw_zeroed.min()):.1f}°, "
              f"{np.degrees(vis_yaw_zeroed.max()):.1f}°]")

    elif yaw_source == 'imu':
        # Interpolate IMU yaw onto the yaw-valid vision time grid
        imu_spline_for_vis = CubicSpline(df_imu['T_rel'], imu_yaw_unwrapped, bc_type='clamped')
        mu_yaw_raw = imu_spline_for_vis(df_yaw['T_rel'].values)
        print(f"  - IMU Yaw amplitude:    [{np.degrees(imu_yaw_unwrapped.min()):.1f}°, "
              f"{np.degrees(imu_yaw_unwrapped.max()):.1f}°]")

    elif yaw_source == 'blend':
        imu_spline_for_vis = CubicSpline(df_imu['T_rel'], imu_yaw_unwrapped, bc_type='clamped')
        imu_on_vis_grid    = imu_spline_for_vis(df_yaw['T_rel'].values)
        mu_yaw_raw = (1.0 - yaw_blend_alpha) * vis_yaw_zeroed + yaw_blend_alpha * imu_on_vis_grid
        print(f"  - Blend Yaw amplitude:  [{np.degrees(mu_yaw_raw.min()):.1f}°, "
              f"{np.degrees(mu_yaw_raw.max()):.1f}°]  "
              f"(vis={np.degrees(vis_yaw_zeroed.min()):.1f}°/{np.degrees(vis_yaw_zeroed.max()):.1f}°, "
              f"imu={np.degrees(imu_yaw_unwrapped.min()):.1f}°/{np.degrees(imu_yaw_unwrapped.max()):.1f}°)")
    else:
        raise ValueError(f"Unknown yaw_source '{yaw_source}'. Use 'vision', 'imu', or 'blend'.")

    # 5. MOTOR ALPHA
    # Zero at first sample — robot is placed at mechanical neutral before each test.
    motor_center        = df_imu['Actual'].iloc[0]
    motor_rad           = (df_imu['Actual'].values - motor_center) * (2 * np.pi / 4096)
    motor_rad_unwrapped = np.unwrap(motor_rad)

    # 6. CUBIC SPLINE INTERPOLATION TO 1000 Hz
    target_dt = 0.001
    t_sim = np.arange(0, t_max, target_dt)
    print(f"  - Interpolating to {len(t_sim)} samples (1000 Hz)...")

    spline_x     = CubicSpline(df_pos['T_rel'], mu_x_raw)
    spline_y     = CubicSpline(df_pos['T_rel'], mu_y_raw)
    spline_z     = CubicSpline(df_pos['T_rel'], mu_z_raw)
    spline_yaw   = CubicSpline(df_yaw['T_rel'], mu_yaw_raw)
    # bc_type='clamped' forces derivative=0 at boundaries, eliminating the t=0 velocity spike
    spline_alpha = CubicSpline(df_imu['T_rel'], motor_rad_unwrapped, bc_type='clamped')

    # 7. GENERATE DATA COLUMNS
    sim_data = pd.DataFrame({'Time': t_sim})
    sim_data['Ghost_X']     = spline_x(t_sim)
    sim_data['Ghost_Y']     = spline_y(t_sim)
    sim_data['Ghost_Z']     = 0.0   # Z is constrained in sim — noisy vision depth omitted
    sim_data['Ghost_Yaw']   = spline_yaw(t_sim)
    sim_data['Ghost_Alpha'] = spline_alpha(t_sim)

    sim_data['Ghost_vX']     = spline_x.derivative()(t_sim)
    sim_data['Ghost_vY']     = spline_y.derivative()(t_sim)
    sim_data['Ghost_vZ']     = 0.0
    sim_data['Ghost_Omega']  = spline_yaw.derivative()(t_sim)
    sim_data['Ghost_dAlpha'] = spline_alpha.derivative()(t_sim)

    # Fill the gap between motor start (T=0) and first AprilTag detection with zeros.
    # Position/yaw are undefined before the tag is visible; 0 is the correct prior
    # because all positions are already zeroed relative to the first detection.
    t_vis_start = df_pos['T_rel'].min()
    if t_vis_start > 0:
        pre = t_sim < t_vis_start
        for col in ['Ghost_X', 'Ghost_Y', 'Ghost_Yaw', 'Ghost_vX', 'Ghost_vY', 'Ghost_Omega']:
            sim_data.loc[pre, col] = 0.0
        print(f"  - Pre-detection gap: {t_vis_start:.3f}s filled with zeros "
              f"(motor running, tag not yet visible)")

    # 7b. COMMANDED_ALPHA — ideal motor target, before real motor dynamics shape it.
    # Used by field2sim when motor.twin_actuation = 'commanded'.
    # Source 1: Goal column from Dynamixel log (actual position target sent each step).
    # Source 2: reconstructed pure sine from run name (e.g. "run_3_1.0Hz_9.0mm").
    commanded_alpha = None
    if 'Goal' in df_imu.columns:
        goal_rad        = (df_imu['Goal'].values - motor_center) * (2 * np.pi / 4096)
        goal_spline     = CubicSpline(df_imu['T_rel'], np.unwrap(goal_rad), bc_type='clamped')
        commanded_alpha = goal_spline(t_sim)
        print(f"  - Commanded_Alpha: from Goal column  "
              f"range [{commanded_alpha.min():.4f}, {commanded_alpha.max():.4f}] rad")
    else:
        import re
        m = re.search(r'_([\d.]+)Hz_([\d.]+)mm', run_name)
        if m:
            freq_hz  = float(m.group(1))
            amp_mm   = float(m.group(2))
            # Run-name mm was computed with r=12.5mm (openloop_swim.py default),
            # so convert ticks → rad via the same 12.5mm radius, not pulley_radius.
            FIELD_RADIUS_M = 0.0125
            amp_rad  = (amp_mm / 1000.0) / FIELD_RADIUS_M
            commanded_alpha = amp_rad * np.sin(2 * np.pi * freq_hz * t_sim)
            print(f"  - Commanded_Alpha: reconstructed sine {freq_hz} Hz, "
                  f"{amp_mm} mm @12.5mm radius → {amp_rad:.4f} rad amplitude")
        else:
            print(f"  - Commanded_Alpha: skipped (no Goal column, run name not parseable)")

    if commanded_alpha is not None:
        sim_data['Commanded_Alpha'] = commanded_alpha

    # 8. SPARSE MEASUREMENT MASKS
    # Vision: X, Y position + Yaw heading (from selected yaw source)
    sim_data['Vision_Update'] = False
    sim_data['Meas_X']        = np.nan
    sim_data['Meas_Y']        = np.nan
    sim_data['Meas_Yaw']      = np.nan

    # Position measurements from df_pos (excludes position-nulled rows)
    for i in range(len(df_pos)):
        idx = int(round(df_pos['T_rel'].iloc[i] / target_dt))
        if 0 <= idx < len(sim_data):
            sim_data.at[idx, 'Vision_Update'] = True
            sim_data.at[idx, 'Meas_X']        = mu_x_raw[i]
            sim_data.at[idx, 'Meas_Y']        = mu_y_raw[i]

    # Yaw measurements from df_yaw (excludes yaw-nulled rows)
    for i in range(len(df_yaw)):
        idx = int(round(df_yaw['T_rel'].iloc[i] / target_dt))
        if 0 <= idx < len(sim_data):
            meas_yaw_val = mu_yaw_raw[i]
            sim_data.at[idx, 'Meas_Yaw'] = (meas_yaw_val + np.pi) % (2 * np.pi) - np.pi

    # IMU: motor encoder + GyrZ (yaw rate)
    # GyrZ confirmed as yaw-rate axis (largest gyro range: ~3.8 rad/s for run_3)
    sim_data['IMU_Update'] = False
    sim_data['Meas_Alpha'] = np.nan
    sim_data['Meas_GyrZ']  = np.nan

    for i in range(len(df_imu)):
        idx = int(round(df_imu['T_rel'].iloc[i] / target_dt))
        if 0 <= idx < len(sim_data):
            sim_data.at[idx, 'IMU_Update'] = True
            sim_data.at[idx, 'Meas_Alpha'] = motor_rad[i]
            sim_data.at[idx, 'Meas_GyrZ']  = df_imu['GyrZ'].iloc[i]

    # 9. DIAGNOSTICS (dense interpolation for plotting/monitoring only)
    diag_cols = ['Volt', 'Temp', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ']
    for col in diag_cols:
        if col in df_imu.columns:
            sim_data[col] = np.interp(t_sim, df_imu['T_rel'], df_imu[col])

    # 10. SAVE
    out_path = os.path.join(proc_dir, "03_field2sim_ready.csv")
    sim_data.to_csv(out_path, index=False)
    print(f"  - SUCCESS: {out_path}")
    print(f"    Yaw source: '{yaw_source}' | "
          f"Ghost_Yaw range: [{sim_data['Ghost_Yaw'].min():.3f}, {sim_data['Ghost_Yaw'].max():.3f}] rad")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare field data for Field-to-Sim replay.")
    parser.add_argument("path",         type=str,   nargs='?',         help="Path to run directory")
    parser.add_argument("--radius",     type=float, default=17.5,      help="Pulley radius in mm (default: 17.5)")
    parser.add_argument("--yaw_source", type=str,   default='imu',
                        choices=['vision', 'imu', 'blend'],             help="Yaw measurement source (default: imu)")
    parser.add_argument("--blend_alpha",type=float, default=0.7,       help="IMU weight for blend mode (default: 0.7)")
    args = parser.parse_args()

    if args.path and os.path.isdir(args.path):
        prep_field2sim(
            args.path,
            pulley_radius=args.radius / 1000.0,
            yaw_source=args.yaw_source,
            yaw_blend_alpha=args.blend_alpha,
        )
    else:
        print("Please provide a valid run directory path.")
        print("Example: python3 03_prep_field2sim.py <run_dir> --yaw_source imu")
