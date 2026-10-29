"""Log the BNO055 on the Pi with the fish still, for the IMU noise floor.

    python3 sysid/imu/log_imu.py --mode 6dof        # IMUPLUS: gyro + accelerometer fusion, magnetometer off (the mode the deployment uses)
    python3 sysid/imu/log_imu.py --mode 9dof        # NDOF: magnetometer on

Waits for the sensor's own calibration (move it in figure-eights, rest it on its faces), then logs raw
and fused readings at about 100 Hz into data/03_sysid/imu_noise/imu_<mode>_data_<timestamp>.csv until
Ctrl+C. Analyse with sysid/imu/analyze_imu.py.
"""
import argparse
import csv
import datetime
import os
import sys
import time

import board
import busio
import adafruit_bno055

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import IMU_NOISE  # noqa: E402

HEADERS = [
    "Timestamp_s", "Calib_Sys", "Calib_Gyro", "Calib_Accel", "Calib_Mag",
    "Raw_Accel_X", "Raw_Accel_Y", "Raw_Accel_Z",
    "Raw_Gyro_X", "Raw_Gyro_Y", "Raw_Gyro_Z",
    "Raw_Mag_X", "Raw_Mag_Y", "Raw_Mag_Z",
    "Lin_Accel_X", "Lin_Accel_Y", "Lin_Accel_Z",
    "Quat_W", "Quat_X", "Quat_Y", "Quat_Z",
    "Euler_Yaw", "Euler_Roll", "Euler_Pitch",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["6dof", "9dof"], default="6dof")
    ap.add_argument("--prep", type=float, default=15.0, help="seconds between calibration and logging, to tape the sensor down")
    args = ap.parse_args()
    use_mag = args.mode == "9dof"

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    sensor.mode = adafruit_bno055.NDOF_MODE if use_mag else adafruit_bno055.IMUPLUS_MODE

    os.makedirs(IMU_NOISE, exist_ok=True)
    filename = os.path.join(IMU_NOISE, f"imu_{args.mode.upper()}_data_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")

    print(f"--- BNO055 logger, {args.mode} ({'NDOF' if use_mag else 'IMUPLUS'}) ---")
    print("1. Calibration: move the sensor in figure-eights and rest it on different faces.")
    while True:
        sys_c, gyro, accel, mag = sensor.calibration_status
        print(f"   calibration -> sys {sys_c}, gyro {gyro}, accel {accel}, mag {mag if use_mag else 'ignored'}", end="\r")
        if gyro == 3 and accel == 3 and (not use_mag or (sys_c == 3 and mag == 3)):
            print("\n   calibrated")
            break
        time.sleep(0.1)

    print(f"2. Logging in {args.prep:.0f} s: tape the sensor down and do not touch it.")
    for i in range(int(args.prep), 0, -1):
        print(f"   starting in {i} s   ", end="\r")
        time.sleep(1)
    print("\n   logging; Ctrl+C to stop")

    t0 = time.time()
    n = 0
    try:
        with open(filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADERS)
            while True:
                t = time.time() - t0
                sys_c, gyro, accel, mag = sensor.calibration_status
                a_raw, g_raw, a_lin, quat, euler = (sensor.acceleration, sensor.gyro, sensor.linear_acceleration,
                                                    sensor.quaternion, sensor.euler)
                m_raw = sensor.magnetic if use_mag else (sensor.magnetic or (0.0, 0.0, 0.0))
                if all(v is not None for v in (a_raw, g_raw, m_raw, a_lin, quat, euler)):
                    w.writerow([round(t, 4), sys_c, gyro, accel, mag, *a_raw, *g_raw, *m_raw, *a_lin, *quat, *euler])
                    n += 1
                    if n % 500 == 0:
                        print(f"   {n} rows ({t:.0f} s)")
                time.sleep(0.01)
    except KeyboardInterrupt:
        print(f"\nsaved {n} rows to {filename}")


if __name__ == "__main__":
    main()
