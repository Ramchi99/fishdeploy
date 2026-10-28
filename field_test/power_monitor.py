#!/usr/bin/env python3
import time
import os
import sys
from datetime import datetime

# --- PATHING SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DXL_DIR = os.path.join(REPO_ROOT, 'hardware')
sys.path.append(REPO_ROOT)
sys.path.append(DXL_DIR)

try:
    from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    print("Error: Dynamixel SDK not found.")
    sys.exit(1)

def monitor_power():
    print("--- SOFI POWER MONITOR ---")
    print("Reading voltage from Dynamixel ID: 1")
    print("Log will be saved to: power_crash_log.txt")
    
    try:
        openPortObject()
    except Exception as e:
        print(f"Failed to open port: {e}")
        return

    servo_id = 1
    ADDR_PRESENT_VOLTAGE = 144 # 2 bytes, units of 0.1V
    
    # Open a log file with immediate flushing
    with open("power_crash_log.txt", "w") as log:
        log.write("Unix_Time, Voltage_V\n")
        
        print("\n[MONITORING START] Unplug the cable now...")
        print("Time (s) | Voltage (V)")
        print("----------------------")
        
        try:
            while True:
                now = time.time()
                # Read 2 bytes from the voltage register
                val, result, error = packetHandler.read2ByteTxRx(portHandler, servo_id, ADDR_PRESENT_VOLTAGE)
                
                if result == 0: # COMM_SUCCESS
                    voltage = val / 10.0
                    output = f"{now:.3f} | {voltage:.2f}V"
                    print(output)
                    
                    # Write to file and FORCE flush to disk
                    log.write(f"{now:.6f}, {voltage:.2f}\n")
                    log.flush()
                    os.fsync(log.fileno()) # Ensure it's physically written
                else:
                    print(f"Comm Error: {result}")
                
                # High-frequency sampling
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            closePortObject()

if __name__ == "__main__":
    monitor_power()
