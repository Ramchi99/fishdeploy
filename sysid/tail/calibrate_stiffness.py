import time, math, os, csv
import numpy as np
from datetime import datetime
import argparse
import sys

# --- BULLETPROOF PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '../..')))

# --- DYNAMIXEL IMPORTS ---

import sys
import os
__script_dir = os.path.dirname(os.path.abspath(__file__))
__repo_root = os.path.abspath(os.path.join(__script_dir, '../..'))
__dxl_dir = os.path.join(__repo_root, 'hardware')
if __repo_root not in sys.path: sys.path.insert(0, __repo_root)
if __dxl_dir not in sys.path: sys.path.insert(0, __dxl_dir)

from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
from dxlSetup.XL430 import XL430
from dynamixel_sdk import DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD, COMM_SUCCESS

parser = argparse.ArgumentParser(description="Bilateral Quasi-Static Stiffness Calibration")
parser.add_argument("--max_current", type=int, default=1000, help="Max current in mA (default: 1000, max: 1193)")
parser.add_argument("--pulley_radius", type=float, default=12.5, help="Pulley radius in mm (default: 12.5)")
parser.add_argument("--dataset", type=str, default="default", help="Dataset name to prefix the output file")
args = parser.parse_args()

## Parameters ##
MAX_CURRENT_mA = args.max_current
CURRENT_STEP_mA = 25
SETTLE_TIME_S = 1.0
PULLEY_RADIUS_MM = args.pulley_radius

## Setup ##
openPortObject()
servo = XL430(id=1, zeroPos=0)

servo.setOpMode(0) 
servo.torqueEnable()

print("\n--- Starting Bilateral Stiffness Calibration ---")
print(f"Max Current: {MAX_CURRENT_mA}mA | Step: {CURRENT_STEP_mA}mA")
print(f"Pulley Radius: {PULLEY_RADIUS_MM}mm | Dataset: {args.dataset}")
print("If the tendon creeps or stretches, tighten it and run this script again!\n")

time.sleep(2)

def set_current(motor, target_current):
    current_unit = 2.69
    goal_current_ticks = int(round(target_current / current_unit))
    packetHandler.write2ByteTxRx(portHandler, motor.dxlID, 102, goal_current_ticks & 0xFFFF)

def twos_complement(val, bits):
    if (val & (1 << (bits - 1))) != 0: 
        val = val - (1 << bits)
    return val

current_targets = []
current_targets.extend(range(0, MAX_CURRENT_mA + CURRENT_STEP_mA, CURRENT_STEP_mA))
current_targets.extend(range(MAX_CURRENT_mA - CURRENT_STEP_mA, -1, -CURRENT_STEP_mA))
current_targets.extend(range(-CURRENT_STEP_mA, -MAX_CURRENT_mA - CURRENT_STEP_mA, -CURRENT_STEP_mA))
current_targets.extend(range(-MAX_CURRENT_mA + CURRENT_STEP_mA, 1, CURRENT_STEP_mA))

currents_tested = []
positions_measured = []

try:
    print("Recording true center point (0mA)...")
    set_current(servo, 0)
    time.sleep(1)
    center_pos = servo.readPos()
    print(f"Center Position Ticks: {center_pos}\n")
    
    for i, current_mA in enumerate(current_targets):
        if current_mA == 0 and i == 0:
            print("--- Sweeping POSITIVE Side ---")
        elif current_mA == -CURRENT_STEP_mA and current_targets[i-1] == 0:
            print("\n--- Sweeping NEGATIVE Side ---")
            
        set_current(servo, current_mA)
        time.sleep(SETTLE_TIME_S)
        
        meas_pos = servo.readPos()
        actual_load = servo.readLoad()
        
        delta_ticks = meas_pos - center_pos
        delta_revs = delta_ticks / 4096.0
        tendon_travel_mm = delta_revs * (2 * math.pi * PULLEY_RADIUS_MM)
        
        print(f"Cmd: {current_mA:5d} mA | Pos: {delta_ticks:5d} ticks | Tendon Pull: {tendon_travel_mm:5.1f} mm | Load: {actual_load}")
        
        currents_tested.append(current_mA)
        positions_measured.append(delta_ticks)

    print("\nRelaxing...")
    set_current(servo, 0)
    time.sleep(1)

except KeyboardInterrupt:
    print("\nAborted early! Relaxing motor...")
    set_current(servo, 0)
    pass

servo.torqueDisable()
closePortObject()

if len(currents_tested) > 0:
    save_dir = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{timestamp}_{args.dataset}_Stiffness_Calibration.csv")
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PulleyRadius_mm', 'SettleTime_s', 'MaxCurrent_mA'])
        writer.writerow([PULLEY_RADIUS_MM, SETTLE_TIME_S, MAX_CURRENT_mA])
        writer.writerow(['Commanded_Current_mA', 'Measured_Delta_Ticks'])
        
        for i in range(len(currents_tested)):
            writer.writerow([currents_tested[i], positions_measured[i]])
            
    print(f"\nCalibration data saved to: {filename}")
