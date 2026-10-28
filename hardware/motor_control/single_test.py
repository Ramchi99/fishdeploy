import time, math, os, csv
import numpy as np
from datetime import datetime
import argparse
import sys

# --- BULLETPROOF PATHING ---
__script_dir = os.path.dirname(os.path.abspath(__file__))
__repo_root = os.path.abspath(os.path.join(__script_dir, '..', '..'))
__dxl_dir = os.path.join(__repo_root, 'hardware')
if __repo_root not in sys.path: sys.path.insert(0, __repo_root)
if __dxl_dir not in sys.path: sys.path.insert(0, __dxl_dir)

from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
from dxlSetup.XL430 import XL430
from dynamixel_sdk import GroupSyncWrite, GroupBulkRead
from dynamixel_sdk import DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD

def twos_complement(val, bits):
    if (val & (1 << (bits - 1))) != 0: val = val - (1 << bits)
    return val

def main():
    parser = argparse.ArgumentParser(description="Run a single frequency/amplitude motor test")
    parser.add_argument("--freq", type=float, required=True, help="Sine wave frequency in Hz")
    parser.add_argument("--amp_mm", type=float, required=True, help="Amplitude in mm")
    parser.add_argument("--pulley_radius", type=float, default=12.5, help="Pulley radius in mm")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds")
    parser.add_argument("--motor_vel", type=int, default=266, help="Motor velocity limit")
    parser.add_argument("--dataset", type=str, default="single_test", help="Dataset name prefix")
    args = parser.parse_args()

    # Calculate Ticks from Amplitude
    amplitude_ticks = int((args.amp_mm / (2 * math.pi * args.pulley_radius)) * 4096)

    ## Setup ##
    openPortObject()
    servo = XL430(id=1, zeroPos=0)
    servo.setOpMode(4) # Mode 4: Extended Position Control
    servo.torqueEnable()

    packetHandler.write4ByteTxRx(portHandler, servo.dxlID, servo.addrTable['ADDR_PROF_VEL'], args.motor_vel)
    groupWrite = GroupSyncWrite(portHandler, packetHandler, servo.addrTable['ADDR_GOAL_POSITION'], 4)

    ADDR_START_READ = 126
    LEN_READ_TOTAL = 10
    groupBulkRead = GroupBulkRead(portHandler, packetHandler)
    groupBulkRead.addParam(servo.dxlID, ADDR_START_READ, LEN_READ_TOTAL)

    center_pos = servo.readPos()
    print(f"\n--- Running Single Test ---")
    print(f"Freq: {args.freq} Hz | Amp: {args.amp_mm} mm ({amplitude_ticks} ticks)")
    print(f"Duration: {args.duration} s")
    
    data_log = []
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < args.duration:
            current_time = time.time() - start_time
            target_pos = int(round(center_pos + amplitude_ticks * math.sin(2 * math.pi * args.freq * current_time)))
            
            param_goal_pos = [DXL_LOBYTE(DXL_LOWORD(target_pos)), DXL_HIBYTE(DXL_LOWORD(target_pos)), 
                              DXL_LOBYTE(DXL_HIWORD(target_pos)), DXL_HIBYTE(DXL_HIWORD(target_pos))]
            groupWrite.clearParam()
            groupWrite.addParam(servo.dxlID, param_goal_pos)
            groupWrite.txPacket()

            groupBulkRead.txRxPacket()
            if groupBulkRead.isAvailable(servo.dxlID, ADDR_START_READ, LEN_READ_TOTAL):
                raw_load = twos_complement(groupBulkRead.getData(servo.dxlID, 126, 2), 16)
                raw_vel = twos_complement(groupBulkRead.getData(servo.dxlID, 128, 4), 32)
                raw_pos = twos_complement(groupBulkRead.getData(servo.dxlID, 132, 4), 32)
                data_log.append([round(current_time, 4), raw_pos, target_pos, raw_vel, raw_load])

    except KeyboardInterrupt:
        print("\nAborted.")

    ## Clean up ##
    servo.move(goal_pos=int(center_pos), prof_vel=args.motor_vel)
    time.sleep(1)
    servo.torqueDisable()
    closePortObject()

    # Save data
    save_dir = os.path.join(__repo_root, "sysid/tail/data")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{timestamp}_{args.dataset}_{args.freq}Hz_{args.amp_mm}mm.csv")
    
    print(f"Saving to {filename}...")
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Env', 'Dataset', 'PulleyRadius_mm', 'Freq', 'AmpFrac', 'CmdAmpTravel_mm', 'CmdAmpTicks'])
        writer.writerow(['single', args.dataset, args.pulley_radius, args.freq, 1.0, args.amp_mm, amplitude_ticks])
        writer.writerow(['Time (s)', 'Position', 'GoalPosition', 'Velocity', 'Load'])
        writer.writerows(data_log)
    print("Done!")

if __name__ == "__main__":
    main()