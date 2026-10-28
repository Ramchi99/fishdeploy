import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import argparse
import os

def analyze_motor_noise(csv_file):
    print(f"Loading benchmark data from: {csv_file}")
    df = pd.read_csv(csv_file, skiprows=2)  # Skip the two metadata header rows
    
    # Check if necessary columns exist
    if 'Time (s)' not in df.columns or 'Velocity' not in df.columns or 'Position' not in df.columns:
        print("Error: CSV does not have 'Time (s)', 'Position', or 'Velocity' columns. Ensure it is a dynamic benchmark file.")
        return

    time = df['Time (s)'].values
    raw_vel = df['Velocity'].values
    raw_pos = df['Position'].values
    
    dt = np.mean(np.diff(time))
    fps = 1.0 / dt
    print(f"Data sampling rate: ~{fps:.1f} Hz")
    
    # ---------------------------------------------------------
    # 1. POSITION NOISE (Mostly Quantization)
    # ---------------------------------------------------------
    # Position noise is so low we just filter it lightly to find the tiny variance
    window_length = min(15, len(raw_pos) - 1)
    if window_length % 2 == 0: window_length -= 1
    
    smooth_pos = savgol_filter(raw_pos, window_length, 3)
    pos_residuals = raw_pos - smooth_pos
    pos_variance_ticks = np.var(pos_residuals)
    pos_std_ticks = np.std(pos_residuals)
    
    # ---------------------------------------------------------
    # 2. VELOCITY NOISE (The big problem)
    # ---------------------------------------------------------
    # We apply a heavy low-pass filter to find the "True Physical Velocity" of the tail mass
    vel_window = min(31, len(raw_vel) - 1)
    if vel_window % 2 == 0: vel_window -= 1
    
    smooth_vel = savgol_filter(raw_vel, vel_window, 3)
    vel_residuals = raw_vel - smooth_vel
    vel_variance = np.var(vel_residuals)
    vel_std = np.std(vel_residuals)

    print("\n" + "="*40)
    print("🎯 DYNAMIXEL NOISE ESTIMATION")
    print("="*40)
    print("These are the values you should hardcode into your EKF's R matrix:")
    print("\n[Motor Position Noise]")
    print(f"Standard Deviation : {pos_std_ticks:.3f} ticks")
    print(f"Variance (R_pos)   : {pos_variance_ticks:.3f} ticks^2")
    
    # Conversion to radians: 1 tick = (2 * pi) / 4096 rads
    tick_to_rad = (2 * np.pi) / 4096
    pos_var_rads = pos_variance_ticks * (tick_to_rad ** 2)
    print(f"Variance (R_pos)   : {pos_var_rads:.8f} rads^2")

    print("\n[Motor Velocity Noise]")
    print(f"Standard Deviation : {vel_std:.3f} ticks/sec")
    print(f"Variance (R_vel)   : {vel_variance:.3f} (ticks/sec)^2")
    
    # Dynamixel velocity units are weird: 1 unit ~ 0.229 RPM
    # 0.229 RPM = 0.229 * (2*pi/60) rad/sec = 0.02398 rad/sec
    rpm_to_radsec = 0.229 * (2 * np.pi / 60)
    vel_var_radsec = vel_variance * (rpm_to_radsec ** 2)
    print(f"Variance (R_vel)   : {vel_var_radsec:.6f} (rad/s)^2")

    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    plot_dir = os.path.join(os.path.dirname(csv_file), "../motor_noise/plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: The raw vs smooth velocity
    ax1.plot(time, raw_vel, color='lightgray', label='Raw Dynamixel Velocity', linewidth=1.5)
    ax1.plot(time, smooth_vel, color='red', label='True Physical Velocity (Filtered)', linewidth=2)
    ax1.set_title(f'Dynamixel Velocity Noise Analysis\n({base_name})')
    ax1.set_ylabel('Velocity (Units)')
    ax1.legend()
    ax1.grid(True)
    
    # Bottom Plot: The Residuals (The Noise)
    ax2.plot(time, vel_residuals, color='blue', alpha=0.7, label='Extracted Noise Residuals')
    ax2.axhline(0, color='black', linestyle='--')
    ax2.axhline(vel_std, color='red', linestyle=':', label=f'+1 Std Dev ({vel_std:.1f})')
    ax2.axhline(-vel_std, color='red', linestyle=':')
    ax2.set_title('Extracted Internal Velocity Noise')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (Units)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, f"{base_name}_velocity_noise.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"\nVisualization saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate Motor EKF Variance from Benchmark Data")
    parser.add_argument("csv_file", help="Path to a Dynamic Benchmark CSV file")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found.")
    else:
        analyze_motor_noise(args.csv_file)