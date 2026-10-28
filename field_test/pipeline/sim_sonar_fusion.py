import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def run_imu_fusion_simulation():
    print("\n--- Digital Twin: Internal IMU-Only Fusion (No Sonar) ---")
    
    # 1. SELECT DATA
    base_data = "data"
    sessions = sorted([d for d in os.listdir(base_data) if os.path.isdir(os.path.join(base_data, d)) and d not in ['old', 'apriltag_recording']])
    print("\nSelect Session:"); [print(f"  [{i+1}] {s}") for i, s in enumerate(sessions)]
    try:
        s_idx = int(input("> ")) - 1
        session_path = os.path.join(base_data, sessions[s_idx])
    except: return

    runs = sorted([d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))])
    print("\nSelect Run:"); [print(f"  [{i+1}] {r}") for i, r in enumerate(runs)]
    try:
        r_idx = int(input("> ")) - 1
        run_path = os.path.join(session_path, runs[r_idx])
    except: return
    
    proc_dir = os.path.join(run_path, "processed")
    df_vis = pd.read_csv(os.path.join(proc_dir, "02_clean_path.csv"))
    df_imu = pd.read_csv(os.path.join(proc_dir, "02_fish_log_trimmed.csv"))

    # 2. PREPARE DATA
    t = df_imu['Unix_Time'].values
    dt = np.diff(t, prepend=t[0])
    
    # Fast Dynamics (Gyro)
    gyro_y = np.degrees(df_imu['GyrY'].values)
    
    # Absolute Anchor (Raw Magnetometer-Fused Yaw from log)
    raw_mag_yaw = df_imu['RawY'].values
    # Normalize to start at 0
    raw_mag_yaw = (raw_mag_yaw - raw_mag_yaw[0] + 180) % 360 - 180
    
    # GROUND TRUTH (Vision)
    true_yaw = np.interp(t, df_vis['Unix_Time'], df_vis['Yaw'])
    true_yaw = (true_yaw - true_yaw[0] + 180) % 360 - 180

    # 3. APPLY YOUR THEORY (Low-Pass Filter the Magnetometer)
    # 2-second window (200 samples) removes the motor oscillations
    mag_anchor = pd.Series(raw_mag_yaw).rolling(200, center=True).mean().ffill().bfill().values

    # 4. PERFORM FUSION (Internal Only)
    fused_yaw = np.zeros(len(t))
    fused_yaw[0] = mag_anchor[0]
    
    # Complementary Filter Gain
    # We trust the 'Anchor' (Magnetometer) very slowly to stop drift
    alpha = 0.005 

    print(f"Fusing 100Hz Gyro with 0.5Hz Filtered Magnetometer...")
    for i in range(1, len(t)):
        # Predict with Gyro
        prediction = fused_yaw[i-1] + gyro_y[i] * dt[i]
        
        # Correct with Filtered Magnetometer (User Theory)
        diff = (mag_anchor[i] - prediction + 180) % 360 - 180
        fused_yaw[i] = prediction + (alpha * diff)

    # Final Prep
    raw_gyro_integrated = np.cumsum(gyro_y * dt)
    raw_gyro_integrated = (raw_gyro_integrated - raw_gyro_integrated[0] + 180) % 360 - 180
    fused_plot = (fused_yaw - fused_yaw[0] + 180) % 360 - 180

    # 5. PLOT
    plt.figure(figsize=(12, 7))
    
    plt.plot(t - t[0], true_yaw, label='Vision (Ground Truth)', color='green', linewidth=3)
    plt.plot(t - t[0], raw_gyro_integrated, label='Integrated Gyro (Drifts)', color='purple', linestyle=':', alpha=0.5)
    plt.plot(t - t[0], raw_mag_yaw, label='Raw BNO055 Yaw (Noisy)', color='blue', alpha=0.2)
    plt.plot(t - t[0], mag_anchor, label='Averaged Magnetometer Anchor', color='orange', linewidth=2, alpha=0.8)
    plt.plot(t - t[0], fused_plot, label='FINAL FUSED ESTIMATE', color='red', linewidth=2)
    
    plt.title(f"Internal State Estimation (No External Sensors): {runs[r_idx]}")
    plt.xlabel("Time (s)")
    plt.ylabel("Yaw (Degrees)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_imu_fusion_simulation()
