import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d, CubicSpline
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import DATA  # noqa: E402

# --- PLOTTING STYLE (from test_freq.py) ---
plt.rcParams.update({"font.size": 7})
plt.rcParams.update({"pdf.fonttype": 42})
plt.rcParams.update({"ps.fonttype": 42})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"font.family": 'serif'})
mm = 1 / 25.4

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def plot_field_test(run_dir, show_removed=True):
    run_name = os.path.basename(run_dir)
    print(f"\n--- Plotting Field Test: {run_name} ---")

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

    # 2. PRE-PROCESS
    df_vis = df_vis.dropna(subset=['X', 'Y', 'Z', 'Yaw']).reset_index(drop=True)

    # Apply partial nulling from vision_settings.json
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

    df_pos = df_vis.dropna(subset=['X', 'Y', 'Z']).copy()
    df_yaw = df_vis.dropna(subset=['Yaw']).copy()

    if len(df_pos) > 1:
        df_pos = df_pos.iloc[:-1].copy()
    if len(df_yaw) > 1:
        df_yaw = df_yaw.iloc[:-1].copy()

    t0 = df_imu['Unix_Time'].min()   # motor/test start, not first vision detection
    df_pos['T_rel'] = df_pos['Unix_Time'] - t0
    df_yaw['T_rel'] = df_yaw['Unix_Time'] - t0
    df_imu['T_rel'] = df_imu['Unix_Time'] - t0

    # World Z = swim axis → plot X
    # World X = lateral LEFT (cross-product with gravity-DOWN convention): negate so that
    # positive plot-Y = fish going RIGHT/UP in the pool, matching the camera overhead view.
    mu_x = df_pos['Z'].values - df_pos['Z'].iloc[0]
    mu_y = -(df_pos['X'].values - df_pos['X'].iloc[0])
    
    # Speed in x direction (flipped finite difference of mu_x)
    # Using a sign flip instead of np.abs() to avoid rectification bias in the MAF
    dt = np.diff(df_pos['T_rel'].values)
    vx_raw = -(np.diff(mu_x) / dt)
    t_vx = df_pos['T_rel'].values[:-1] + dt/2
    
    # MAF for speed
    window = 30
    if len(vx_raw) > window:
        vx_maf = moving_average(vx_raw, window)
        t_vx_maf = t_vx[window-1:]
    else:
        vx_maf = vx_raw
        t_vx_maf = t_vx

    # --- PLOTTING ---
    
    # 1. Trajectory (Equal Scale, Full Box)
    # Size: 88x50mm
    # Compute data bounds and derive figure size to show full trajectory at equal scale.
    # The X axis is always 88mm wide; figure height is scaled to match the data aspect ratio.
    pad = 0.05
    x_min, x_max = mu_x.min() - pad, mu_x.max() + pad
    y_min, y_max = mu_y.min() - pad, mu_y.max() + pad
    x_span = x_max - x_min
    y_span = y_max - y_min
    fig_w = 88 * mm
    fig_h = fig_w * (y_span / x_span)
    fig_h = np.clip(fig_h, 30*mm, 160*mm)  # keep between 30–160mm tall

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.plot(mu_x, mu_y, color="tab:blue", label="Vision (AprilTag)", linewidth=1.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xlabel("X (m) [Swim Axis]")
    ax.set_ylabel("Y (m) [Lateral]")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(proc_dir, "04_trajectory.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(proc_dir, "04_trajectory.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 2. Yaw (IMU vs AprilTag)
    # Size: 88x50mm
    fig, ax = plt.subplots(figsize=(88*mm, 50*mm))
    try:
        vis_yaw_deg = df_yaw['Yaw'].values
        vis_yaw_zeroed = vis_yaw_deg - vis_yaw_deg[0]
        vis_yaw_zeroed = np.degrees(np.unwrap(np.radians(vis_yaw_zeroed)))
        ax.plot(df_yaw['T_rel'], vis_yaw_zeroed, color="tab:orange", label="Vision (AprilTag)", alpha=0.8, linewidth=1.0)
    except KeyError:
        print("  ! Warning: 'Yaw' not found in 02_clean_path.csv. Skipping AprilTag Yaw.")
        
    imu_yaw_deg = df_imu['RawY'].values
    imu_yaw_zeroed = imu_yaw_deg - imu_yaw_deg[0]
    imu_yaw_zeroed = np.degrees(np.unwrap(np.radians(imu_yaw_zeroed)))
    ax.plot(df_imu['T_rel'], imu_yaw_zeroed, color="tab:blue", label="IMU (BNO055)", alpha=0.8, linewidth=0.8)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Yaw (deg)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="lower left", frameon=True, fancybox=True, framealpha=0.8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(proc_dir, "04_yaw_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(proc_dir, "04_yaw_comparison.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 3. X over time
    # Size: 88x50mm
    fig, ax = plt.subplots(figsize=(88*mm, 50*mm))
    ax.plot(df_pos['T_rel'], mu_x, color="tab:green", label="Vision (AprilTag)", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X (m)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(proc_dir, "04_x_pos.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(proc_dir, "04_x_pos.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 4. Speed in x direction over time
    # Size: 88x50mm
    fig, ax = plt.subplots(figsize=(88*mm, 50*mm))
    ax.plot(t_vx, vx_raw, color="tab:gray", alpha=0.3, label="Vision Raw", linewidth=0.5)
    ax.plot(t_vx_maf, vx_maf, color="tab:red", label=f"Vision MAF (n={window})", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed in x direction (m/s)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(proc_dir, "04_speed_x.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(proc_dir, "04_speed_x.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 5. Total speed — Savitzky-Golay with deriv=1 (state of the art for motion capture data).
    # SG simultaneously smooths and differentiates by fitting a local polynomial to each window,
    # then taking its analytical derivative. This avoids:
    #   - noise amplification from finite differences
    #   - phase shift from filtering then differentiating
    #   - rectification bias from MAF(|v|) near zero
    # Requires regular sampling → interpolate to a uniform grid first.
    # Size: 88x50mm
    t_pos   = df_pos['T_rel'].values
    fps_est = 1.0 / np.median(np.diff(t_pos))          # estimate actual camera rate
    t_reg   = np.arange(t_pos[0], t_pos[-1], 1.0 / fps_est)
    # Asymmetric BC: clamped start (fish at rest → zero velocity), natural end
    # (zero second derivative, does NOT force zero velocity — fish is still swimming).
    x_reg   = CubicSpline(t_pos, mu_x, bc_type=((1, 0.0), (2, 0.0)))(t_reg)
    y_reg   = CubicSpline(t_pos, mu_y, bc_type=((1, 0.0), (2, 0.0)))(t_reg)

    # Window: 21 samples ≈ 0.7 s at 30 Hz — resolves swimming harmonics, rejects noise.
    # Must be odd and > polyorder (3). Increase window for more smoothing (not polyorder —
    # higher polyorder = more flexible fit = LESS smooth).
    #
    # Asymmetric edge handling:
    #   Start — manually pad with the first position (constant → zero velocity) so the SG
    #            window sees a flat signal and estimates ~0 m/s at t=0.
    #   End   — mode='interp' (scipy default): fits a polynomial to available data,
    #            freely estimates the actual velocity — fish hasn't stopped.
    sg_window  = 51
    sg_poly    = 3
    _pad = sg_window // 2
    x_padded = np.concatenate([np.full(_pad, x_reg[0]), x_reg])
    y_padded = np.concatenate([np.full(_pad, y_reg[0]), y_reg])
    vx_sg = savgol_filter(x_padded, sg_window, sg_poly,
                          deriv=1, delta=1.0 / fps_est, mode='interp')[_pad:]
    vy_sg = savgol_filter(y_padded, sg_window, sg_poly,
                          deriv=1, delta=1.0 / fps_est, mode='interp')[_pad:]
    speed_sg = np.sqrt(vx_sg**2 + vy_sg**2)

    # Raw speed (finite difference) shown for reference only
    dt_raw    = np.diff(t_pos)
    speed_raw = np.sqrt((np.diff(mu_x) / dt_raw)**2 + (np.diff(mu_y) / dt_raw)**2)
    t_raw     = t_pos[:-1] + dt_raw / 2

    fig, ax = plt.subplots(figsize=(88*mm, 50*mm))
    ax.plot(t_raw, speed_raw, color="tab:gray", alpha=0.25, linewidth=0.5, label="Raw (finite diff)")
    ax.plot(t_reg, speed_sg,  color="tab:red",  linewidth=1.0,
            label=f"SG deriv=1 (w={sg_window}, p={sg_poly})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8, ncol=2)
    plt.tight_layout()
    fig.savefig(os.path.join(proc_dir, "04_speed_total.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(proc_dir, "04_speed_total.pdf"), bbox_inches="tight")
    plt.close(fig)

    # 7. Motor: Commanded (Goal) vs Measured (Actual)
    # Size: 88x50mm
    if 'Goal' in df_imu.columns and 'Actual' in df_imu.columns:
        motor_center = df_imu['Actual'].iloc[0]
        goal_rad   = (df_imu['Goal'].values   - motor_center) * (2 * np.pi / 4096)
        actual_rad = (df_imu['Actual'].values - motor_center) * (2 * np.pi / 4096)
        error_rad  = goal_rad - actual_rad

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(88*mm, 70*mm),
                                        sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08},
                                        constrained_layout=True)

        ax1.plot(df_imu['T_rel'], goal_rad,   color="tab:blue",   label="Commanded (Goal)",   linewidth=0.9)
        ax1.plot(df_imu['T_rel'], actual_rad, color="tab:orange", label="Measured (Actual)", linewidth=0.9, alpha=0.85)
        ax1.set_ylabel(r"$\alpha$ (rad)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8, ncol=2)

        ax2.plot(df_imu['T_rel'], error_rad, color="tab:red", linewidth=0.8, label="Error (Cmd $-$ Meas)")
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel(r"$\Delta\alpha$ (rad)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.8)

        fig.savefig(os.path.join(proc_dir, "04_motor_tracking.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(proc_dir, "04_motor_tracking.pdf"), bbox_inches="tight")
        plt.close(fig)
    else:
        print("  ! Warning: Goal/Actual columns not found — skipping motor tracking plot.")

    print(f"  - Plots saved to {proc_dir}")

def main():
    base_data_path = str(DATA)
    
    # 1. Select Session (Day)
    sessions = sorted([d for d in os.listdir(base_data_path) if glob.glob(os.path.join(base_data_path, d, "run_*", "raw")) and d not in ['old', 'apriltag_recording']])
    
    if not sessions:
        print("No session folders (data/<session>/run_*/raw) found in data/")
        return

    print("\nAvailable Sessions:")
    for i, s in enumerate(sessions):
        print(f"  [{i+1}] {s}")
    
    s_choice = input("\nSelect Session Number: ").strip()
    try:
        session_name = sessions[int(s_choice)-1]
    except:
        print("Invalid selection."); return

    # 2. Select Run
    session_path = os.path.join(base_data_path, session_name)
    runs = sorted([d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))])
    
    print(f"\nRuns in {session_name}:")
    for i, r in enumerate(runs):
        print(f"  [{i+1}] {r}")
    
    r_choice = input("\nEnter Run Number (or 'all'): ").strip().lower()
    
    if r_choice == 'all':
        for r in runs: 
            run_dir = os.path.join(session_path, r)
            plot_field_test(run_dir)
    else:
        try:
            run_idx = int(r_choice) - 1
            if not (0 <= run_idx < len(runs)):
                raise ValueError(f"Run number must be 1–{len(runs)}")
            run_dir = os.path.join(session_path, runs[run_idx])
            plot_field_test(run_dir)
        except (ValueError, IndexError) as e:
            print(f"Invalid selection: {e}")

if __name__ == "__main__":
    main()
