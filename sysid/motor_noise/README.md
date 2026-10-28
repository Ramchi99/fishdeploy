# Motor EKF Noise Analysis

This module is specifically dedicated to finding the **Measurement Noise Covariance ($R$ matrix)** for the Dynamixel motor encoders, which is required for an Extended Kalman Filter (EKF).

Unlike the `tail` folder (which measures how well the motor tracks a command under physical load), this folder purely isolates the **internal electrical noise** of the sensor. 

## How it works
The `analyze_motor_noise.py` script takes existing CSV data where the motor is moving continuously (like a dynamic sine wave benchmark). It applies a heavy low-pass filter to strip away the sensor noise, revealing the "True Physical Velocity". It then calculates the variance of the residuals between the raw sensor data and that true physical velocity.

## Directory Structure
- `analyze_motor_noise.py`: The analysis script.
- `plots/`: Contains generated graphs showing the noise separation.
- `data/`: Placeholder if you want to store copies of specific benchmark files here.

## Usage
Run the analyzer on any of your existing benchmark CSV files. It will calculate the variances and save a plot in the `plots/` directory.

```bash
# Example from the root of fishdeploy:
python3 sysid/motor_noise/analyze_motor_noise.py sysid/tail/data/YOUR_BENCHMARK.csv
```