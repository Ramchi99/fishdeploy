import time, math, os, csv
import numpy as np
from datetime import datetime
import argparse
import sys

# --- BULLETPROOF PATHING ---
__script_dir = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(__script_dir, "..", "..")))   # repository root
from paths import IMU_NOISE  # noqa: E402
__repo_root = os.path.abspath(os.path.join(__script_dir, '../..'))
__dxl_dir = os.path.join(__repo_root, 'hardware')
if __repo_root not in sys.path: sys.path.insert(0, __repo_root)
if __dxl_dir not in sys.path: sys.path.insert(0, __dxl_dir)

# --- DYNAMIXEL IMPORTS ---
from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
from dxlSetup.XL430 import XL430
from dynamixel_sdk import GroupSyncWrite, GroupBulkRead
from dynamixel_sdk import DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD

# --- IMU IMPORTS ---
import board
import busio
import adafruit_bno055

def twos_complement(val, bits):
    if (val & (1 << (bits - 1))) != 0: val = val - (1 << bits)
    return val

def main():
    parser = argparse.ArgumentParser(description="Tethered Vibration Benchmark (Motor + IMU)")
    parser.add_argument("--pulley_radius", type=float, default=12.5, help="Pulley radius in mm")
    parser.add_argument("--max_tendon_travel", type=float, required=True, help="Absolute max tendon travel (amplitude) in mm")
    parser.add_argument("--amp_frac", type=float, default=1.0, help="Fraction of max travel (0.0 to 1.0)")
    parser.add_argument("--freq", type=float, default=2.0, help="Sine wave frequency in Hz")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds (default: 30)")
    parser.add_argument("--delay", type=int, default=30, help="Countdown delay before starting in seconds (default: 30)")
    parser.add_argument("--dataset", type=str, default="vibration_test", help="Dataset name prefix")
    args = parser.parse_args()

    # --- 1. SETUP IMU ---
    print("\n--- 1. Initializing IMU ---")
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    sensor.mode = adafruit_bno055.IMUPLUS_MODE
    
    print("Please gently tilt the sensor to calibrate the Gyro and Accel (No figure-8 needed for IMUPLUS mode).")
    while True:
        sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
        print(f"Calibration -> Sys: {sys_cal}, Gyro: {gyro_cal}, Accel: {accel_cal}", end="\r")
        if gyro_cal == 3 and accel_cal == 3:
            print("\nSensor fully calibrated for 6DOF!")
            break
        time.sleep(0.1)

    # --- 2. SETUP MOTOR ---
    print("\n--- 2. Initializing Motor ---")
    max_ticks_theta = int((args.max_tendon_travel / (2 * math.pi * args.pulley_radius)) * 4096)
    amplitude_ticks = int(max_ticks_theta * args.amp_frac)
    
    openPortObject()
    servo = XL430(id=1, zeroPos=0)
    servo.setOpMode(4) # Mode 4: Extended Position Control
    servo.torqueEnable()
    
    motorVel = 266
    packetHandler.write4ByteTxRx(portHandler, servo.dxlID, servo.addrTable['ADDR_PROF_VEL'], motorVel)
    groupWrite = GroupSyncWrite(portHandler, packetHandler, servo.addrTable['ADDR_GOAL_POSITION'], 4)
    
    ADDR_START_READ = 126
    LEN_READ_TOTAL = 10
    groupBulkRead = GroupBulkRead(portHandler, packetHandler)
    groupBulkRead.addParam(servo.dxlID, ADDR_START_READ, LEN_READ_TOTAL)

    center_pos = servo.readPos()
    print(f"Motor Centered at {center_pos}. Amplitude: {amplitude_ticks} ticks. Frequency: {args.freq} Hz.")
    
    print(f"\n--- 3. Running Benchmark in {args.delay} seconds ---")
    print("Please rigidly mount (Tether) the fish now. Step away before it starts!")
    for i in range(args.delay, 0, -1):
        print(f"Starting in {i} seconds...  ", end="\r")
        time.sleep(1)
    print("\nGO!")

    # Pre-allocate data lists for speed
    data_log = []
    
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time() - start_time
            if current_time > args.duration:
                break
                
            # Motor Sine Wave Command
            target_pos = int(round(center_pos + amplitude_ticks * math.sin(2 * math.pi * args.freq * current_time)))
            param_goal_pos = [DXL_LOBYTE(DXL_LOWORD(target_pos)), DXL_HIBYTE(DXL_LOWORD(target_pos)), 
                              DXL_LOBYTE(DXL_HIWORD(target_pos)), DXL_HIBYTE(DXL_HIWORD(target_pos))]
            groupWrite.clearParam()
            groupWrite.addParam(servo.dxlID, param_goal_pos)
            groupWrite.txPacket()

            # Read Motor
            m_pos, m_vel, m_load = 0, 0, 0
            groupBulkRead.txRxPacket()
            if groupBulkRead.isAvailable(servo.dxlID, ADDR_START_READ, LEN_READ_TOTAL):
                m_load = twos_complement(groupBulkRead.getData(servo.dxlID, 126, 2), 16)
                m_vel = twos_complement(groupBulkRead.getData(servo.dxlID, 128, 4), 32)
                m_pos = twos_complement(groupBulkRead.getData(servo.dxlID, 132, 4), 32)

            # Read IMU
            sys_cal, gyro_cal, accel_cal, mag_cal = sensor.calibration_status
            a_lin = sensor.linear_acceleration
            g_raw = sensor.gyro
            euler = sensor.euler

            if a_lin is not None and g_raw is not None and euler is not None:
                data_log.append([
                    round(current_time, 4), target_pos, m_pos, m_vel, m_load,
                    a_lin[0], a_lin[1], a_lin[2],
                    g_raw[0], g_raw[1], g_raw[2],
                    euler[0], euler[1], euler[2]
                ])

    except KeyboardInterrupt:
        print("\nAborted by user.")

    print("\n--- 4. Cleaning up ---")
    servo.move(goal_pos=int(center_pos), prof_vel=motorVel)
    time.sleep(1)
    servo.torqueDisable()
    closePortObject()

    # --- 5. SAVING DATA ---
    save_dir = str(IMU_NOISE)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = os.path.join(save_dir, f"{timestamp}_{args.dataset}_{args.freq}Hz_{args.amp_frac}Amp_vibration.csv")
    
    print(f"Saving {len(data_log)} samples to {filename}...")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time_s", "Motor_TargetPos", "Motor_Pos", "Motor_Vel", "Motor_Load",
                         "IMU_LinAccel_X", "IMU_LinAccel_Y", "IMU_LinAccel_Z",
                         "IMU_Gyro_X", "IMU_Gyro_Y", "IMU_Gyro_Z",
                         "IMU_Euler_Yaw", "IMU_Euler_Roll", "IMU_Euler_Pitch"])
        writer.writerows(data_log)
        
    print("Done!")

if __name__ == "__main__":
    main()