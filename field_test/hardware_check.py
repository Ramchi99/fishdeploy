import sys
import os
import time
import math
import numpy as np
from datetime import datetime
import glob

# --- PATHING ---
__script_dir = os.path.dirname(os.path.abspath(__file__))
__repo_root = os.path.abspath(os.path.join(__script_dir, '..'))
__dxl_dir = os.path.join(__repo_root, 'hardware')
if __repo_root not in sys.path: sys.path.insert(0, __repo_root)
if __dxl_dir not in sys.path: sys.path.insert(0, __dxl_dir)

# --- IMPORTS ---
try:
    from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
    from dxlSetup.XL430 import XL430
    from dynamixel_sdk import COMM_SUCCESS
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False

try:
    import board
    import busio
    import adafruit_bno055
    IMU_AVAILABLE = True
except ImportError:
    IMU_AVAILABLE = False

try:
    import smbus
    from hardware.battery_check.ina219 import INA219
    BATTERY_AVAILABLE = True
except ImportError:
    BATTERY_AVAILABLE = False

def scan_i2c():
    print("\n[Scanning I2C Buses...]")
    found_buses = {}
    i2c_paths = glob.glob("/dev/i2c-*")
    bus_ids = sorted([int(p.split('-')[-1]) for p in i2c_paths])

    for bus_id in bus_ids:
        found_on_bus = []
        try:
            bus = smbus.SMBus(bus_id)
            for address in range(0x03, 0x78):
                try:
                    bus.write_quick(address)
                    found_on_bus.append(hex(address))
                except OSError:
                    pass
            if 0 < len(found_on_bus) < 10:
                found_buses[bus_id] = found_on_bus
                print(f"Bus {bus_id}: {', '.join(found_on_bus)}")
            elif len(found_on_bus) >= 10:
                print(f"Bus {bus_id}: Ignored (Ghosting)")
        except Exception:
            pass
    return found_buses

def test_imu():
    print("\n--- IMU TEST (BNO055) ---")
    if not IMU_AVAILABLE:
        print("IMU Library not found. Skipping.")
        return
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        sensor = adafruit_bno055.BNO055_I2C(i2c)
        print("IMU Connected.")
        print("Waiting for calibration (Sys, Gyro, Accel, Mag)...")
        print("Calibration Tips:")
        print("  - Gyro:  Keep sensor perfectly still.")
        print("  - Accel: Rotate sensor into 6 stable positions.")
        print("  - Mag:   Move in a figure-8 in the air.")
        
        start_time = time.time()
        timeout = 60
        while time.time() - start_time < timeout:
            euler = sensor.euler
            calib = sensor.calibration_status # (sys, gyro, accel, mag)
            
            if euler is not None:
                # \033[K clears to the end of the line to prevent "ghost" brackets
                output = f"Yaw:{euler[0]:5.1f} | R:{euler[1]:5.1f} | P:{euler[2]:5.1f} | Calib (S,G,A,M): {calib} \033[K"
                print(output, end='\r')
            
            # Check if fully calibrated (all values == 3)
            if calib == (3, 3, 3, 3):
                print(f"\n\nIMU FULLY CALIBRATED in {int(time.time() - start_time)}s!")
                break
            time.sleep(0.1)
        else:
            print(f"\n\nIMU Timeout reached ({timeout}s). Continuing without full calibration.")
            
        print("IMU Test OK.")
    except Exception as e:
        print(f"\nIMU Test Failed: {e}")

def test_motor_and_power(found_buses):
    print("\n--- MOTOR & POWER TEST ---")
    if not MOTOR_AVAILABLE:
        print("Dynamixel SDK not found. Skipping.")
        return

    try:
        openPortObject()
        servo = XL430(id=1, zeroPos=0)
        
        # 1. BATTERY CHECK (Try UPS first, then Motor)
        ups_targets = ['0x40', '0x41', '0x42', '0x43', '0x44', '0x45']
        target_bus, target_addr = None, None
        for b_id, addrs in found_buses.items():
            for a in addrs:
                if a in ups_targets:
                    target_bus, target_addr = b_id, int(a, 16)
                    break
            if target_bus: break

        voltage = 0.0
        source = "None"
        
        if target_addr:
            try:
                ina = INA219(i2c_bus=target_bus, addr=target_addr)
                voltage = ina.getBusVoltage_V()
                source = f"UPS (I2C {hex(target_addr)})"
            except: pass
        
        if voltage < 1.0: # Fallback to Motor
            raw_v, _, _ = packetHandler.read2ByteTxRx(portHandler, servo.dxlID, 144)
            voltage = raw_v / 10.0
            source = f"Dynamixel (ID {servo.dxlID})"

        # Calculate percentage (Assuming 3S LiPo: 9.0V to 12.6V)
        percent = max(0, min(100, (voltage - 9.0) / 3.6 * 100))
        print(f"Battery Voltage: {voltage:.2f} V (Source: {source})")
        print(f"Estimated Charge: {percent:.1f}%")
        if percent < 20:
            print("WARNING: Battery low! Please charge.")

        # 2. MOTOR SINE TEST
        err, _, _ = packetHandler.read1ByteTxRx(portHandler, servo.dxlID, 70)
        if err != 0:
            print(f"MOTOR ERROR: {err}. Please fix limits and reboot.")
        else:
            amplitude_ticks = int((3.0 / (2 * math.pi * 12.5)) * 4096)
            print(f"Moving 1Hz Sine Wave (3mm)...")
            servo.setOpMode(4)
            servo.torqueEnable()
            center_pos = servo.readPos()
            start_time = time.time()
            while (time.time() - start_time) < 4.0:
                elapsed = time.time() - start_time
                target = int(center_pos + amplitude_ticks * math.sin(2 * math.pi * 1.0 * elapsed))
                packetHandler.write4ByteTxOnly(portHandler, servo.dxlID, 116, target)
                time.sleep(0.01)
            servo.move(goal_pos=int(center_pos), prof_vel=266)
            time.sleep(1)
            print("Motor Sine Test Done.")

        servo.torqueDisable()
        closePortObject()
        print("System Test OK.")
    except Exception as e:
        print(f"Test Failed: {e}")

def main():
    print("========================================")
    print("      SOFI ROBOTIC FISH FIELD TEST      ")
    print("========================================")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    found_buses = scan_i2c()
    test_imu()
    test_motor_and_power(found_buses)
    print("\n========================================")
    print("         FIELD TEST COMPLETE            ")
    print("========================================")

if __name__ == "__main__":
    main()
