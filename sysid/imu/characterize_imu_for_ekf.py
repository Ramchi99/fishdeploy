import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import os
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))   # repository root
from paths import IMU_NOISE  # noqa: E402

def characterize_imu(csv_files, output_pdf="sysid/imu/plots/imu_ekf_characterization.pdf", config_json="sysid/imu/data/imu_params.json"):
    all_stats = []
    
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    os.makedirs(os.path.dirname(config_json), exist_ok=True)
    
    with PdfPages(output_pdf) as pdf:
        for csv_file in csv_files:
            print(f"Processing {csv_file}...")
            df = pd.read_csv(csv_file)
            
            # 1. Crop data (12s start, 15s end) to ensure we analyze stable motor-running state
            start_t = df['Timestamp_s'].iloc[0] + 12
            end_t = df['Timestamp_s'].iloc[-1] - 15
            df = df[(df['Timestamp_s'] >= start_t) & (df['Timestamp_s'] <= end_t)].copy()
            
            if df.empty:
                print(f"Skipping {csv_file}: Dataframe empty after cropping.")
                continue

            # 2. Calculate Sampling Frequency
            dt = df['Timestamp_s'].diff().dropna()
            fs = 1.0 / dt.mean()
            fs_std = (1.0 / dt).std()

            # 3. Analyze Fused Data (Quaternions and Euler)
            quat_cols = ['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']
            quat_var = df[quat_cols].var()
            
            euler_cols = ['Euler_Roll', 'Euler_Pitch', 'Euler_Yaw']
            euler_stats = df[euler_cols].agg(['mean', 'std', 'var'])

            # 4. Analyze Prediction Inputs (Gyro and Lin Accel)
            gyro_cols = ['Raw_Gyro_X', 'Raw_Gyro_Y', 'Raw_Gyro_Z']
            accel_cols = ['Lin_Accel_X', 'Lin_Accel_Y', 'Lin_Accel_Z']
            
            gyro_stats = df[gyro_cols].agg(['mean', 'std', 'var'])
            accel_stats = df[accel_cols].agg(['mean', 'std', 'var'])

            # Store stats for JSON
            run_stats = {
                "file": os.path.basename(csv_file),
                "sampling_rate_hz": float(fs),
                "sampling_jitter_std": float(fs_std),
                "quat_variance": quat_var.to_dict(),
                "euler_variance": euler_stats.loc['var'].to_dict(),
                "gyro_variance": gyro_stats.loc['var'].to_dict(),
                "accel_variance": accel_stats.loc['var'].to_dict(),
                "gyro_bias": gyro_stats.loc['mean'].to_dict(),
                "accel_bias": accel_stats.loc['mean'].to_dict(),
                "euler_bias": euler_stats.loc['mean'].to_dict()
            }
            all_stats.append(run_stats)

            # --- PLOTTING FOR PDF ---
            fig = plt.figure(figsize=(11.69, 8.27)) # A4 Landscape
            plt.suptitle(f"IMU Characterization: {os.path.basename(csv_file)}\n(Motor Running, Cropped 12s-15s)", fontsize=16)

            # Subplot 1: Sampling Jitter
            plt.subplot(2, 2, 1)
            plt.hist(1.0/dt, bins=30, color='skyblue', edgecolor='black')
            plt.axvline(fs, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {fs:.2f}Hz')
            plt.title("Sampling Rate Distribution")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Samples")
            plt.legend()

            # Subplot 2: Gyro Noise
            plt.subplot(2, 2, 2)
            for col in gyro_cols:
                plt.plot(df['Timestamp_s'] - start_t, df[col] - df[col].mean(), label=f"{col} (σ={gyro_stats.loc['std', col]:.4f})", alpha=0.7)
            plt.title("Gyroscope Noise (Zero-Centered)")
            plt.xlabel("Time (s)")
            plt.ylabel("Rad/s")
            plt.legend(fontsize='small')

            # Subplot 3: Linear Accel Noise
            plt.subplot(2, 2, 3)
            for col in accel_cols:
                plt.plot(df['Timestamp_s'] - start_t, df[col] - df[col].mean(), label=f"{col} (σ={accel_stats.loc['std', col]:.4f})", alpha=0.7)
            plt.title("Linear Accel Noise (Zero-Centered)")
            plt.xlabel("Time (s)")
            plt.ylabel("m/s^2")
            plt.legend(fontsize='small')

            # Subplot 4: Euler Angles Noise
            plt.subplot(2, 2, 4)
            for col in euler_cols:
                plt.plot(df['Timestamp_s'] - start_t, df[col] - df[col].mean(), label=f"{col} (σ={euler_stats.loc['std', col]:.4f})", alpha=0.7)
            plt.title("Euler Angles Noise (Zero-Centered)")
            plt.xlabel("Time (s)")
            plt.ylabel("Degrees")
            plt.legend(fontsize='small')

            # Summary Text at the bottom
            summary_text = (
                f"EKF PARAMETERS: Rate: {fs:.2f} Hz | Sys Calib: {df['Calib_Sys'].mode()[0]} | Mag Calib: {df['Calib_Mag'].mode()[0]}\n"
                f"Gyro Var: {gyro_stats.loc['var'].mean():.2e} | Accel Var: {accel_stats.loc['var'].mean():.2e}\n"
                f"Quat Var: {quat_var.mean():.2e} | Euler Var (deg^2): {euler_stats.loc['var'].mean():.2e}"
            )
            fig.text(0.5, 0.02, summary_text, ha='center', va='bottom', fontsize=11, family='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

            plt.tight_layout(rect=[0, 0.08, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

    # Save finalized JSON config
    with open(config_json, 'w') as f:
        json.dump(all_stats, f, indent=4)
    
    print(f"\n✅ Analysis Complete!")
    print(f"📄 PDF Report: {output_pdf}")
    print(f"⚙️  EKF Config: {config_json}")

if __name__ == "__main__":
    data_files = [
        str(IMU_NOISE / 'imu_9DOF_data_20260320_185648.csv'),
        str(IMU_NOISE / 'imu_9DOF_data_20260320_191340.csv')
    ]
    characterize_imu(data_files)
