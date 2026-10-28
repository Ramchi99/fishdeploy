import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def analyze_imu_data(csv_file):
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Crop data: remove first 10s and last 5s
    start_time = df['Timestamp_s'].iloc[0] + 10
    end_time = df['Timestamp_s'].iloc[-1] - 5
    print(f"Cropping data from {start_time:.2f}s to {end_time:.2f}s...")
    df = df[(df['Timestamp_s'] >= start_time) & (df['Timestamp_s'] <= end_time)]
    
    if df.empty:
        print("Error: Cropped data is empty. The file may be too short.")
        return

    # Basic info
    duration = df['Timestamp_s'].iloc[-1] - df['Timestamp_s'].iloc[0]
    num_samples = len(df)
    sample_rate = num_samples / duration if duration > 0 else 0
    
    print("-" * 30)
    print("📊 DATASET SUMMARY")
    print("-" * 30)
    print(f"Total Samples : {num_samples}")
    print(f"Duration      : {duration:.2f} seconds")
    print(f"Average Rate  : {sample_rate:.2f} Hz")
    
    # Check calibration status (taking the mode/most common value)
    print("\nCalibration Status (0=Uncalibrated, 3=Fully Calibrated):")
    calib_vals = []
    for col in ['Calib_Sys', 'Calib_Gyro', 'Calib_Accel', 'Calib_Mag']:
        if col in df.columns:
            calib_vals.append(f"{col.split('_')[1]}: {df[col].mode()[0]}")
    print(", ".join(calib_vals))
    
    # Calculate some basic statistics
    print("\n📈 SIGNAL STATISTICS")
    stats_cols = ['Euler_Roll', 'Euler_Pitch', 'Euler_Yaw', 'Lin_Accel_X', 'Lin_Accel_Y', 'Lin_Accel_Z']
    for col in stats_cols:
        if col in df.columns:
            print(f"{col:15s}: Min={df[col].min():8.3f}, Max={df[col].max():8.3f}, Mean={df[col].mean():8.3f}, Std={df[col].std():8.3f}")

    # Plotting
    print("\nGenerating plots...")
    
    # Create output directory for plots based on csv name
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    time = df['Timestamp_s']
    
    # Plot 1: Euler Angles (Orientation)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time, df['Euler_Roll'], label='Roll (X-axis rotation)', color='r')
    ax1.plot(time, df['Euler_Pitch'], label='Pitch (Y-axis rotation)', color='g')
    ax1.plot(time, df['Euler_Yaw'], label='Yaw (Z-axis rotation)', color='b')
    ax1.set_title('IMU Orientation (Euler Angles)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angle (Degrees)')
    ax1.legend()
    ax1.grid(True)
    fig1_path = os.path.join(plot_dir, f"{base_name}_euler.png")
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)
    
    # Plot 2: Linear Acceleration (Movement excluding Gravity)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(time, df['Lin_Accel_X'], label='Accel X (Forward/Back)', color='r')
    ax2.plot(time, df['Lin_Accel_Y'], label='Accel Y (Left/Right)', color='g')
    ax2.plot(time, df['Lin_Accel_Z'], label='Accel Z (Up/Down)', color='b')
    ax2.set_title('IMU Linear Acceleration (Gravity Removed)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s^2)')
    ax2.legend()
    ax2.grid(True)
    fig2_path = os.path.join(plot_dir, f"{base_name}_lin_accel.png")
    fig2.savefig(fig2_path, dpi=200)
    plt.close(fig2)

    # Plot 3: Raw Gyroscope (Angular Velocity)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(time, df['Raw_Gyro_X'], label='Gyro X', color='r')
    ax3.plot(time, df['Raw_Gyro_Y'], label='Gyro Y', color='g')
    ax3.plot(time, df['Raw_Gyro_Z'], label='Gyro Z', color='b')
    ax3.set_title('IMU Angular Velocity (Raw Gyroscope)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (rad/s)')
    ax3.legend()
    ax3.grid(True)
    fig3_path = os.path.join(plot_dir, f"{base_name}_gyro.png")
    fig3.savefig(fig3_path, dpi=200)
    plt.close(fig3)

    # Plot 4: Magnetometer (Raw Magnetic Field)
    mag_cols = ['Raw_Mag_X', 'Raw_Mag_Y', 'Raw_Mag_Z']
    if all(col in df.columns for col in mag_cols):
        # Skip plot if all values are zero (common in 6DOF logs)
        if not (df[mag_cols] == 0).all().all():
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.plot(time, df['Raw_Mag_X'], label='Mag X', color='r')
            ax4.plot(time, df['Raw_Mag_Y'], label='Mag Y', color='g')
            ax4.plot(time, df['Raw_Mag_Z'], label='Mag Z', color='b')
            ax4.set_title('IMU Magnetic Field (Magnetometer)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Magnetic Field (uT)')
            ax4.legend()
            ax4.grid(True)
            fig4_path = os.path.join(plot_dir, f"{base_name}_mag.png")
            fig4.savefig(fig4_path, dpi=200)
            plt.close(fig4)
            print(f" - {os.path.basename(fig4_path)}")

    print(f"Plots saved to: {plot_dir}/")
    print(f" - {os.path.basename(fig1_path)}")
    print(f" - {os.path.basename(fig2_path)}")
    print(f" - {os.path.basename(fig3_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze BNO055 IMU Data")
    parser.add_argument("csv_file", help="Path to the IMU CSV file")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found.")
    else:
        analyze_imu_data(args.csv_file)