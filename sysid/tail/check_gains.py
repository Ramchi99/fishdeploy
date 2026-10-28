import sys
import os
import time

# --- PATHING ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
DXL_DIR = os.path.join(REPO_ROOT, 'hardware')
sys.path.append(REPO_ROOT)
sys.path.append(DXL_DIR)

from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
from dxlSetup.XL430 import XL430

def check_gains():
    openPortObject()
    servo_id = 1
    
    # XL430 Control Table Addresses
    ADDR_POS_D_GAIN = 80
    ADDR_POS_I_GAIN = 82
    ADDR_POS_P_GAIN = 84
    ADDR_FEEDFORWARD_1ST_GAIN = 88
    ADDR_FEEDFORWARD_2ND_GAIN = 90
    
    p_gain, _, _ = packetHandler.read2ByteTxRx(portHandler, servo_id, ADDR_POS_P_GAIN)
    i_gain, _, _ = packetHandler.read2ByteTxRx(portHandler, servo_id, ADDR_POS_I_GAIN)
    d_gain, _, _ = packetHandler.read2ByteTxRx(portHandler, servo_id, ADDR_POS_D_GAIN)
    ff1_gain, _, _ = packetHandler.read2ByteTxRx(portHandler, servo_id, ADDR_FEEDFORWARD_1ST_GAIN)
    ff2_gain, _, _ = packetHandler.read2ByteTxRx(portHandler, servo_id, ADDR_FEEDFORWARD_2ND_GAIN)
    
    print(f"--- Servo ID {servo_id} Gains ---")
    print(f"Position P Gain: {p_gain}")
    print(f"Position I Gain: {i_gain}")
    print(f"Position D Gain: {d_gain}")
    print(f"Feedforward 1st Gain: {ff1_gain}")
    print(f"Feedforward 2nd Gain: {ff2_gain}")
    
    closePortObject()

if __name__ == "__main__":
    check_gains()
