# IMU EKF Noise Analysis

This module is dedicated to determining the **Measurement Noise Covariance ($R$ matrix)** for the BNO055 IMU, specifically for use in Extended Kalman Filters.

It contains scripts to measure both **Static Electrical Noise** (when the robot is perfectly still) and **Dynamic Mechanical Noise** (when the motors are vibrating the chassis).

## Directory Structure
- `analyze_imu.py`: Reads any IMU CSV file, calculates the Standard Deviations/Variances, and generates noise distribution plots.
- `vibration_benchmark.py`: A hardware-in-the-loop script that synchronously flaps the Dynamixel motor while logging the IMU data to measure dynamic vibration noise.
- `data/`: Contains the raw CSV logs (like the 37-minute static test).
- `plots/`: Contains the generated Euler, Acceleration, and Gyroscope variance plots.

## Workflows

### 1. Static Noise Analysis
Run the analyzer on a static log (e.g., the 37-minute test) to find the baseline silicon noise of the sensor.
```bash
python3 analyze_imu.py data/imu_6DOF_data_20260319_162517.csv
```

### 2. Dynamic Noise Analysis (Tethered Test)
To find the realistic noise during swimming, clamp the fish to a table, and run the vibration benchmark:
```bash
# Run on the Pi
python3 vibration_benchmark.py --max_tendon_travel 15.0 --freq 2.0 --amp_frac 1.0
```
Then, pull the data back to your laptop and run `analyze_imu.py` on the resulting CSV. The new, higher standard deviations should be used in your active EKF matrix.