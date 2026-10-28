# Sofi Robotic Fish - Field Test Suite

This folder contains the unified testing and logging environment for the Sofi robotic fish. The primary tool is `unified_test.py`, designed for high-performance data collection during water trials.

## 🚀 Quick Start

Run a basic 1Hz, 5mm amplitude test for 10 seconds:
```bash
python3 field_test/unified_test.py --name "first_swim" --freq 1.0 --amp 5.0 --duration 10.0
```

Run with full 9-DOF IMU (Magnetometer ON) and custom PID gains:
```bash
python3 field_test/unified_test.py --imu-mode 9dof --p 1000 --d 50 --name "stiff_tail_test"
```

## 📦 Installation (Raspberry Pi 5)

Install the required system packages and Python libraries:
```bash
sudo apt update
sudo apt install python3-numpy python3-matplotlib python3-pip
sudo pip3 install adafruit-circuitpython-bno055 dynamixel-sdk --break-system-packages
```

## 🛠 Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--name` | `test_run` | Label for the experiment (used in filenames) |
| `--freq` | `1.0` | Swimming frequency in Hz |
| `--amp` | `5.0` | Peak-to-peak tendon travel in mm |
| `--offset` | `0.0` | Tendon length offset in mm |
| `--duration`| `10.0` | Run time in seconds |
| `--p, --i, --d`| `800, 0, 0` | Dynamixel XL430 PID gains |
| `--imu-mode` | `6dof` | `6dof` (Mag OFF) or `9dof` (Mag ON) |
| `--rate` | `50` | Logging frequency in Hz |
| `--skip-calib`| `False` | Skip the 60s IMU calibration wait |

## 📊 Output & Data
- **Logs:** Saved as CSV in `field_test/data/`. Includes timestamps, motor state (Pos, Vel, Load, Volt, Temp), and full IMU fusion data (Euler, Quaternions, Accel, Gyro, Mag).
- **Plots:** A summary PNG is generated in `field_test/plots/` after every run, showing motor tracking accuracy, orientation, and system health (Voltage/Temp).

## ⚠️ Safety Notes
- **Motor Temp:** Monitor the temperature in the summary plots. If it exceeds **60°C**, stop the tests and let the motor cool.
- **Battery:** If voltage drops below **10.0V** (for a 3S LiPo), recharge immediately to avoid damage.
- **IMU Mode:** Use `6dof` if you notice the heading (Yaw) drifting wildly near the motor. Use `9dof` only if you have calibrated the magnetometer in the field.
