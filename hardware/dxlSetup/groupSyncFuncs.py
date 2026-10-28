import numpy as np
from dynamixel_sdk import *
from .portAndPackets import portHandler,packetHandler
from .DxlServoBase import DxlServoBase, twos_complement_to_decimal

LEN_PRO_GOAL_POSITION = 4
LEN_PRO_PRESENT_POSITION = 4
LEN_PRO_PRESENT_LOAD = 2

def cvt2ByteArray(position:int):
    paramGoal = [DXL_LOBYTE(DXL_LOWORD(position)), DXL_HIBYTE(DXL_LOWORD(position)), DXL_LOBYTE(DXL_HIWORD(position)), DXL_HIBYTE(DXL_HIWORD(position))]
    return paramGoal

def initializeSyncs(servo:DxlServoBase):
    groupReadPosition = GroupSyncRead(portHandler, packetHandler, servo.addrTable['ADDR_PRESENT_POSITION'], LEN_PRO_PRESENT_POSITION)
    groupReadLoad = GroupSyncRead(portHandler, packetHandler, servo.addrTable['ADDR_PRESENT_LOAD'], LEN_PRO_PRESENT_LOAD)
    groupWrite = GroupSyncWrite(portHandler, packetHandler,  servo.addrTable['ADDR_GOAL_POSITION'], LEN_PRO_GOAL_POSITION)
    return groupReadPosition, groupReadLoad, groupWrite

def addRead(groupRead:GroupSyncRead, servo:DxlServoBase):
    dxl_addparam_result = groupRead.addParam(servo.dxlID)
    if not dxl_addparam_result:
        print("[ID:%03d] groupSyncRead addparam failed" % servo.dxlID)

def addWrite(groupWrite:GroupSyncWrite,servo:DxlServoBase,param):
    newParam = cvt2ByteArray(int(np.round(param)))
    dxl_addparam_result = groupWrite.addParam(servo.dxlID,newParam)
    if not dxl_addparam_result:
        print("[ID:%03d] groupSyncWrite addparam failed" % servo.dxlID)

def clearReadParam(groupRead:GroupSyncRead):
    groupRead.clearParam()

def clearWriteParam(groupWrite:GroupSyncWrite):
    groupWrite.clearParam()

def writeGroup(groupWrite:GroupSyncWrite):
    dxl_comm_result = groupWrite.txPacket()
    if dxl_comm_result != COMM_SUCCESS:
        print("Error during write")
        print("%s" % packetHandler.getTxRxResult(dxl_comm_result))

def groupMove(groupWrite:GroupSyncWrite,*servos:DxlServoBase):
    for servo in servos:
        addWrite(groupWrite,servo,servo.goalPos)
    writeGroup(groupWrite)
    clearWriteParam(groupWrite)

def groupReadSetup(groupRead:GroupSyncRead,*servos:DxlServoBase):
    for servo in servos:
        addRead(groupRead,servo)

def readGroupPos(groupRead:GroupSyncRead,*servos:DxlServoBase):
    dxl_comm_result = groupRead.txRxPacket()
    if dxl_comm_result != COMM_SUCCESS:
        readSuccess = False
        print("Error during position read")
        #print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    else:
        readSuccess = True
    for servo in servos:
        presPos = groupRead.getData(servo.dxlID,servo.addrTable['ADDR_PRESENT_POSITION'],LEN_PRO_PRESENT_POSITION)
        servo.presPos = twos_complement_to_decimal(presPos)
        #print(f"Servo {servo.dxlID} Position = {servo.presPos}")
    #clearReadParam(groupRead)
    return readSuccess

def readGroupLoad(groupRead:GroupSyncRead,*servos:DxlServoBase):
    dxl_comm_result = groupRead.txRxPacket()
    if dxl_comm_result != COMM_SUCCESS:
        readSuccess = False
        print("Error during load read")
        #print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
    else:
        readSuccess = True
    for servo in servos:
        presLoad = groupRead.getData(servo.dxlID,servo.addrTable['ADDR_PRESENT_LOAD'],LEN_PRO_PRESENT_LOAD)
        servo.presLoad = twos_complement_to_decimal(presLoad)
        #print(f"Servo {servo.dxlID} Load = {servo.presLoad}")
    #clearReadParam(groupRead)
    return readSuccess