import time, numpy as np

import sys
import os
__script_dir = os.path.dirname(os.path.abspath(__file__))
__repo_root = os.path.abspath(os.path.join(__script_dir, '..', '..'))
__dxl_dir = os.path.join(__repo_root, 'hardware')
if __repo_root not in sys.path: sys.path.insert(0, __repo_root)
if __dxl_dir not in sys.path: sys.path.insert(0, __dxl_dir)

from dxlSetup.portAndPackets import openPortObject,closePortObject
from dxlSetup.groupSyncFuncs import *
from dxlSetup.XL430 import XL430

## Servo Setup ##
openPortObject()
servo = XL430(id=1,zeroPos=0)
servo.setOpMode(1) # Velocity Control mode
servo.torqueEnable()

## Trial Adjustable Parameters ##
desiredFreq = 2 #Hz or rev/s, max of 4.84335

## Trial Constants ##
gearRatio = 5.4
velScale = 0.229
goalVel = int(round(desiredFreq/(gearRatio*velScale)*60))
runFlag = True
try:
    servo.velMove(goalVel)
    while (runFlag):
        time.sleep(0.1)

except KeyboardInterrupt:
    servo.velMove(0)
    runFlag = False
    pass

## Clean up ##
time.sleep(1)
servo.torqueDisable()
closePortObject()