#!/usr/bin/env python3
"""
SOFI Pi Sensor Server
=====================
Runs on the Raspberry Pi.  Broadcasts live IMU (BNO055) and motor
(Dynamixel XL430) data over UDP so that obs_verifier.py on the
laptop can show IMU + motor observation dimensions during the test
protocol (tests 6-10).

Usage (on Pi, from repo root)
------------------------------
    python3 deploy/obs_verify/pi_sensor_server.py

    # Send to a specific laptop IP (faster than broadcast on some networks):
    python3 deploy/obs_verify/pi_sensor_server.py --laptop-ip <your-laptop-ip>

    # Read motor without enabling torque (safe for manual positioning):
    python3 deploy/obs_verify/pi_sensor_server.py --no-torque

Packet format (matches obs_verifier.py SENSOR_FMT)
----------------------------------------------------
    "!fffffff" big-endian:
      [0] wx         rad/s  (BulkheadFixed remap, same as imu_reader.py)
      [1] wy         rad/s
      [2] wz         rad/s  (yaw rate, CCW+ from above)
      [3] pos_rad    rad    (motor position, zeroed at start)
      [4] vel_rad_s  rad/s  (motor velocity)
      [5] _reserved  0.0
      [6] timestamp  s      (laptop time.time()-compatible)
"""
import argparse
import socket
import struct
import sys
import time
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

SENSOR_FMT  = "!fffffff"
SENSOR_PORT = 5007

# Motor unit conversions (same as motor_runner.py)
TICKS_PER_REV  = 4096
RAD_PER_TICK   = 2 * 3.14159265358979 / TICKS_PER_REV
VEL_UNIT_TO_RPM = 0.229
RPM_TO_RAD_S   = 2 * 3.14159265358979 / 60


def _init_imu():
    try:
        import board
        import busio
        import adafruit_bno055
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_bno055.BNO055_I2C(i2c)
        sensor.mode = adafruit_bno055.IMUPLUS_MODE
        time.sleep(0.5)
        print("[IMU] BNO055 ready in IMUPLUS_MODE")
        return sensor
    except Exception as e:
        print(f"[IMU] Not available: {e}  (will broadcast zeros)")
        return None


def _init_motor(no_torque: bool):
    try:
        from hardware.dxlSetup.portAndPackets import openPortObject, packetHandler
        from hardware.dxlSetup.XL430 import XL430
        openPortObject()
        servo = XL430(id=1, zeroPos=0)
        if not no_torque:
            servo.setOpMode(4)
        zero_ticks = servo.readPos()
        print(f"[Motor] XL430 ready  zero_ticks={zero_ticks}  torque={'off' if no_torque else 'on'}")
        return servo, zero_ticks
    except Exception as e:
        print(f"[Motor] Not available: {e}  (will broadcast zeros)")
        return None, 0


def _read_imu(sensor):
    """Returns (wx, wy, wz) with BulkheadFixed remap (same as imu_reader.py)."""
    if sensor is None:
        return 0.0, 0.0, 0.0
    try:
        g = sensor.gyro
        if g is None or any(v is None for v in g):
            return 0.0, 0.0, 0.0
        raw_x, raw_y, raw_z = float(g[0]), float(g[1]), float(g[2])
        return raw_z, -raw_x, raw_y   # wx, wy, wz
    except Exception:
        return 0.0, 0.0, 0.0


def _read_motor(servo, zero_ticks):
    if servo is None:
        return 0.0, 0.0
    try:
        pos_ticks, vel_raw = servo.readPosVel()
        pos_rad   = (pos_ticks - zero_ticks) * RAD_PER_TICK
        vel_rad_s = vel_raw * VEL_UNIT_TO_RPM * RPM_TO_RAD_S
        return pos_rad, vel_rad_s
    except Exception:
        return 0.0, 0.0


def main():
    ap = argparse.ArgumentParser(description="SOFI Pi Sensor Server")
    ap.add_argument("--laptop-ip",  default="255.255.255.255",
                    help="Laptop IP or broadcast address (default: 255.255.255.255)")
    ap.add_argument("--port",       type=int, default=SENSOR_PORT)
    ap.add_argument("--rate",       type=float, default=50.0, help="Broadcast rate in Hz")
    ap.add_argument("--no-torque",  action="store_true",
                    help="Open motor without enabling torque (safe for manual positioning)")
    args = ap.parse_args()

    imu           = _init_imu()
    servo, zero_t = _init_motor(args.no_torque)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    dest = (args.laptop_ip, args.port)
    dt   = 1.0 / args.rate

    print(f"[Server] Broadcasting to {args.laptop_ip}:{args.port} @ {args.rate:.0f} Hz")
    print("[Server] Press Ctrl+C to stop\n")

    n = 0
    try:
        while True:
            t0 = time.time()

            wx, wy, wz       = _read_imu(imu)
            pos_rad, vel_rad = _read_motor(servo, zero_t)

            pkt = struct.pack(SENSOR_FMT,
                              wx, wy, wz, pos_rad, vel_rad, 0.0, time.time())
            try:
                sock.sendto(pkt, dest)
            except OSError:
                pass

            n += 1
            if n % int(args.rate) == 0:
                print(f"\r  wx={wx:+.3f}  wy={wy:+.3f}  wz={wz:+.3f}"
                      f"  pos={pos_rad:+.3f} rad  vel={vel_rad:+.3f} rad/s     ",
                      end="", flush=True)

            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[Server] Stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
