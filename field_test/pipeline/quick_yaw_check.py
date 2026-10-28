import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

def check_yaw():
    print("\n--- Sofi Sensor Comparison Tool ---")
    print(" [1] Compare Yaw (Fused IMU vs Vision)")
    print(" [2] Compare Yaw Rate (Gyro vs Vision Derivative)")
    print(" [3] Compare Integrated Gyro (Pure Gyro vs Vision)")
    mode = input("> ").strip()

    base_data = "field_test/data"
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
    vis_path = os.path.join(proc_dir, "02_clean_path.csv")
    imu_path = os.path.join(proc_dir, "02_fish_log_trimmed.csv")

    if not os.path.exists(vis_path) or not os.path.exists(imu_path):
        print("Error: Missing files. Run 02_process_swimming.py first!")
        return

    df_vis = pd.read_csv(vis_path)
    df_imu = pd.read_csv(imu_path)

    plt.figure(figsize=(12, 6))

    # Helper to wrap angles
    def wrap(x): return (x + 180) % 360 - 180

    if mode == "1":
        # --- YAW POSITION (FUSED) ---
        vis_times = df_vis['Unix_Time']
        vis_yaw_raw = df_vis['Yaw']
        if len(vis_yaw_raw) > 1:
            vis_times = vis_times.iloc[:-1]; vis_yaw_raw = vis_yaw_raw.iloc[:-1]

        vis_yaw = wrap(vis_yaw_raw - vis_yaw_raw.dropna().iloc[0])
        imu_yaw = wrap(df_imu['RawY'] - df_imu['RawY'].dropna().iloc[0])

        plt.plot(vis_times, vis_yaw, label='Vision Yaw', color='green', linewidth=2)
        plt.plot(df_imu['Unix_Time'], imu_yaw, label='IMU Fused Yaw (BNO055)', color='blue', alpha=0.6)
        plt.ylabel("Degrees")
        plt.title(f"Fused Yaw Comparison: {runs[r_idx]}")

    elif mode == "2":
        # --- YAW RATE ---
        dt = np.diff(df_vis['Unix_Time'])
        dyaw = np.diff(df_vis['Yaw'])
        dyaw = (dyaw + 180) % 360 - 180
        vis_rate = dyaw / dt
        imu_rate = np.degrees(df_imu['GyrY']) 

        if len(vis_rate) > 1:
            vis_rate = vis_rate[:-1]; vis_times = df_vis['Unix_Time'].iloc[1:-1]
        else:
            vis_times = df_vis['Unix_Time'].iloc[1:]

        plt.plot(vis_times, vis_rate, label='Vision Yaw Rate', color='green', alpha=0.8)
        plt.plot(df_imu['Unix_Time'], imu_rate, label='IMU Gyro Y', color='blue', alpha=0.5)
        
        all_rates = np.concatenate([vis_rate, imu_rate])
        limit = np.percentile(np.abs(all_rates), 99) * 1.5
        plt.ylim(-limit, limit)
        plt.ylabel("Degrees/s")
        plt.title(f"Yaw Rate Comparison: {runs[r_idx]}")

    elif mode == "3":
        # --- INTEGRATED GYRO ---
        # 1. Prepare Vision
        vis_times = df_vis['Unix_Time']
        vis_yaw_raw = df_vis['Yaw']
        if len(vis_yaw_raw) > 1:
            vis_times = vis_times.iloc[:-1]; vis_yaw_raw = vis_yaw_raw.iloc[:-1]
        vis_yaw = wrap(vis_yaw_raw - vis_yaw_raw.dropna().iloc[0])

        # 2. Integrate IMU Gyro
        imu_times = df_imu['Unix_Time'].values
        imu_gyr_deg = np.degrees(df_imu['GyrY'].values)
        # Calculate dt between samples
        imu_dt = np.diff(imu_times, prepend=imu_times[0])
        # Cumulative sum of (velocity * time)
        integrated_yaw = np.cumsum(imu_gyr_deg * imu_dt)
        # Normalize to start at 0
        integrated_yaw = wrap(integrated_yaw - integrated_yaw[0])

        plt.plot(vis_times, vis_yaw, label='Vision Yaw', color='green', linewidth=2)
        plt.plot(imu_times, integrated_yaw, label='Integrated Gyro Yaw', color='purple', alpha=0.8)
        plt.ylabel("Degrees")
        plt.title(f"Integrated Gyro vs Vision: {runs[r_idx]}")
        print("Tip: If the purple line drifts away from the green, that is your Gyro Bias!")

    plt.xlabel("Unix Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    check_yaw()
