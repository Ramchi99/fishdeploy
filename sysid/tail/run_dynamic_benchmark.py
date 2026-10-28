import time, math, os, csv, platform
import numpy as np
from datetime import datetime
import argparse
import sys

# --- BULLETPROOF PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))


import sys
import os
__script_dir = os.path.dirname(os.path.abspath(__file__))
__repo_root = os.path.abspath(os.path.join(__script_dir, '../..'))
__dxl_dir = os.path.join(__repo_root, 'hardware')
if __repo_root not in sys.path: sys.path.insert(0, __repo_root)
if __dxl_dir not in sys.path: sys.path.insert(0, __dxl_dir)

from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
from dxlSetup.XL430 import XL430
from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, GroupBulkRead, GroupBulkWrite
from dynamixel_sdk import DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD, COMM_SUCCESS

parser = argparse.ArgumentParser(description="Automated Dynamic Benchmarking Grid Search")
parser.add_argument("--env", type=str, required=True, choices=["air", "water"], help="Testing environment")
parser.add_argument("--pulley_radius", type=float, default=12.5, help="Pulley radius in mm (default: 12.5)")
parser.add_argument("--max_tendon_travel", type=float, required=True, help="Absolute max tendon travel (amplitude) in mm based on stiffness calibration")
parser.add_argument("--dataset", type=str, default="default", help="Dataset name to prefix the output files")
parser.add_argument("--motor_vel", type=int, default=266, help="Motor velocity limit in Dynamixel units (266 ~ 61 RPM)")
args = parser.parse_args()

## Matrix Parameters ##
frequencies = [1.0, 2.0, 3.0, 4.0]
amplitude_fractions = [0.2, 0.4, 0.6, 0.8, 1.0] # 20% to 100% of max_tendon_travel
runDuration = 6.0 # Seconds per test run
motorVel = args.motor_vel

# Calculate Max Ticks from Max Tendon Travel
# travel = (ticks / 4096) * (2 * pi * radius) => ticks = travel / (2 * pi * radius) * 4096
max_ticks_theta = int((args.max_tendon_travel / (2 * math.pi * args.pulley_radius)) * 4096)

## Setup ##
openPortObject()
servo = XL430(id=1, zeroPos=0)
servo.setOpMode(4) # Mode 4: Extended Position Control
servo.torqueEnable()

packetHandler.write4ByteTxRx(portHandler, servo.dxlID, servo.addrTable['ADDR_PROF_VEL'], motorVel)
groupWrite = GroupSyncWrite(portHandler, packetHandler, servo.addrTable['ADDR_GOAL_POSITION'], 4)

ADDR_START_READ = 126
LEN_READ_TOTAL = 10
groupBulkRead = GroupBulkRead(portHandler, packetHandler)
groupBulkRead.addParam(servo.dxlID, ADDR_START_READ, LEN_READ_TOTAL)

def twos_complement(val, bits):
    if (val & (1 << (bits - 1))) != 0: val = val - (1 << bits)
    return val

print(f"--- Starting Dynamic Benchmark ({args.env.upper()}) ---")
print(f"Dataset: {args.dataset}")
print(f"Pulley Radius: {args.pulley_radius} mm")
print(f"Max Tendon Travel: {args.max_tendon_travel} mm")
print(f"Calculated Absolute Max Theta: {max_ticks_theta} ticks")
print("This will run a matrix of tests. Please stand by...")

save_dir = os.path.join(SCRIPT_DIR, "data")
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    for freq in frequencies:
        for frac in amplitude_fractions:
            amplitude_travel = args.max_tendon_travel * frac
            amplitude_ticks = int(max_ticks_theta * frac)
            print(f"\nRunning: {freq} Hz | Amplitude: {frac*100}% ({amplitude_travel:.1f} mm, {amplitude_ticks} ticks)")
            
            # Read center position
            center_pos = servo.readPos()
            time.sleep(0.5)

            max_steps = int(runDuration * 500) + 100 
            position_data = np.zeros(max_steps)
            goal_pos_data = np.zeros(max_steps)
            vel_data = np.zeros(max_steps)
            load_data = np.zeros(max_steps)
            time_data = np.zeros(max_steps)

            step = 0
            start_time = time.time()
            
            # RUN THE SINE WAVE
            while (time.time() - start_time) < runDuration:
                current_time = time.time() - start_time
                target_pos = int(round(center_pos + amplitude_ticks * math.sin(2 * math.pi * freq * current_time)))
                
                param_goal_pos = [DXL_LOBYTE(DXL_LOWORD(target_pos)), DXL_HIBYTE(DXL_LOWORD(target_pos)), 
                                  DXL_LOBYTE(DXL_HIWORD(target_pos)), DXL_HIBYTE(DXL_HIWORD(target_pos))]
                groupWrite.clearParam()
                groupWrite.addParam(servo.dxlID, param_goal_pos)
                groupWrite.txPacket()

                groupBulkRead.txRxPacket()
                
                if groupBulkRead.isAvailable(servo.dxlID, ADDR_START_READ, LEN_READ_TOTAL):
                    raw_load = groupBulkRead.getData(servo.dxlID, 126, 2)
                    raw_vel = groupBulkRead.getData(servo.dxlID, 128, 4)
                    raw_pos = groupBulkRead.getData(servo.dxlID, 132, 4)
                    
                    if step < max_steps:
                        load_data[step] = twos_complement(raw_load, 16)
                        vel_data[step] = twos_complement(raw_vel, 32)
                        position_data[step] = twos_complement(raw_pos, 32)
                        goal_pos_data[step] = target_pos
                        time_data[step] = current_time
                        step += 1

            # Save the run
            servo.move(goal_pos=int(center_pos), prof_vel=motorVel)
            time.sleep(1) # Let it settle
            
            if step > 0:
                filename = os.path.join(save_dir, f"{timestamp}_{args.dataset}_Benchmark_{args.env}_{freq}Hz_{frac*100}Amp.csv")
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Env', 'Dataset', 'PulleyRadius_mm', 'Freq', 'AmpFrac', 'CmdAmpTravel_mm', 'CmdAmpTicks'])
                    writer.writerow([args.env, args.dataset, args.pulley_radius, freq, frac, amplitude_travel, amplitude_ticks])
                    writer.writerow(['Time (s)', 'Position', 'GoalPosition', 'Velocity', 'Load'])
                    for i in range(step):
                        writer.writerow([round(time_data[i], 4), int(position_data[i]), int(goal_pos_data[i]), int(vel_data[i]), int(load_data[i])])
            
            print("Waiting 4 seconds for water to settle...")
            time.sleep(4)
                        
except KeyboardInterrupt:
    print("\nBenchmark aborted by user.")

## Clean up ##
servo.move(goal_pos=int(center_pos), prof_vel=motorVel)
time.sleep(1)
servo.torqueDisable()
closePortObject()
print(f"\nBenchmark Complete! Data saved to {save_dir}/")
