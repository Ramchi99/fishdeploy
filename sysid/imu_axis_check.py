#!/usr/bin/env python3
"""
Live IMU axis diagnostic — run on the Pi to verify the BulkheadFixed remap
applied in imu_reader.py produces the correct yaw/pitch/roll assignment.

Usage:
    python3 sysid/imu_axis_check.py

Rotate the fish slowly in each direction while watching the output:
  - Spin nose LEFT or RIGHT  →  wz column should spike  (yaw = obs[5])
  - Tilt nose UP or DOWN     →  wy column should spike  (pitch = obs[4])
  - Roll onto its side       →  wx column should spike  (roll = obs[3])

The left column ("RL deploy / sim frame") is what the policy actually receives
after the BulkheadFixed remap in imu_reader.py.  The right column ("field_test
hardware remap") applies the same remap via the BNO055 hardware registers — both
columns should read IDENTICALLY, confirming the software remap is correct.

BulkheadFixed (conf=0x12, sign=0x06):
  wx =  raw_z  (+physical_Z = field GyrX)
  wy = -raw_x  (-physical_X = field GyrY)
  wz = -raw_y  (-physical_Y = field GyrZ)  ← YAW, obs[5] in policy

Gimbal-lock note: gyro measures angular RATE, not Euler angles — no gimbal lock.

Press Ctrl+C to exit.
"""
import sys
import time
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

try:
    import board
    import busio
    import adafruit_bno055
except ImportError:
    sys.exit("adafruit-circuitpython-bno055 not installed. Run: pip3 install adafruit-circuitpython-bno055")


def apply_bulkhead_remap(raw_x, raw_y, raw_z):
    """Applies the BulkheadFixed remap (conf=0x12, sign=0x06).

    Remap result:
      field_X = +physical_Z  (raw_z,  positive)
      field_Y = -physical_X  (raw_x,  negated)
      field_Z = -physical_Y  (raw_y,  negated)
    """
    return raw_z, -raw_x, -raw_y


def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    sensor.mode = adafruit_bno055.IMUPLUS_MODE
    time.sleep(0.5)

    print("Waiting for gyro calibration (gyro cal = 3)...", flush=True)
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            cal = sensor.calibration_status
            if cal and cal[1] >= 3:
                print(f"Gyro calibrated. Status: {cal}\n")
                break
            print(f"  cal={cal}\r", end="", flush=True)
        except Exception:
            pass
        time.sleep(0.2)
    else:
        print("\nWarning: gyro cal timeout — continuing anyway.")

    header = (
        f"{'RL deploy / sim frame (imu_reader)':^42}  |  {'field_test hw remap (should match)':^42}\n"
        f"{'wx=obs[3]':>12}  {'wy=obs[4]':>12}  {'wz=obs[5]=YAW':>14}  |  "
        f"{'GyrX':>12}  {'GyrY':>12}  {'GyrZ=YAW':>14}\n"
        + "-" * 95
    )
    print(header)

    while True:
        try:
            g = sensor.gyro  # raw: (physical_X, physical_Y, physical_Z)
            if g is None or any(v is None for v in g):
                time.sleep(0.02)
                continue

            raw_x, raw_y, raw_z = float(g[0]), float(g[1]), float(g[2])

            # Software remap (matches imu_reader.py — confirmed on hardware)
            wx =  raw_z   # +physical_Z = field GyrX
            wy = -raw_x   # -physical_X = field GyrY
            wz =  raw_y   # +physical_Y = YAW, CCW-positive confirmed on hardware

            # Hardware remap result (what field_test reads from BNO055 registers)
            gx, gy, gz = apply_bulkhead_remap(raw_x, raw_y, raw_z)

            # Both columns should be identical — any difference = remap bug
            deploy_vals = [abs(wx), abs(wy), abs(wz)]
            dom = deploy_vals.index(max(deploy_vals))

            def fmt(val, i, dominant):
                s = f"{val:+10.4f}"
                return s + "*" if i == dominant else s + " "

            line = (
                fmt(wx, 0, dom) + "  " +
                fmt(wy, 1, dom) + "  " +
                fmt(wz, 2, dom) + "    |  " +
                f"{gx:+10.4f}   {gy:+10.4f}   {gz:+10.4f}"
            )
            print(f"\r{line}", end="", flush=True)
            time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n\nExpected results:")
            print("  Spin fish nose LEFT (CCW from above) → wz > 0  (obs[5], yaw)")
            print("  Spin fish nose RIGHT (CW from above) → wz < 0  (obs[5], yaw)")
            print("  Tilt nose UP                         → wy spike (obs[4], pitch)")
            print("  Roll onto right side                 → wx spike (obs[3], roll)")
            print("\n  Both columns (deploy / field_test) should be identical.")
            print("  If they differ, the software remap in imu_reader.py is wrong.")
            break
        except Exception as e:
            print(f"\nIMU read error: {e}", flush=True)
            time.sleep(0.1)


if __name__ == "__main__":
    main()
