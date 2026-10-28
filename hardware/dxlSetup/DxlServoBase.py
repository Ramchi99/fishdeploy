from dynamixel_sdk import COMM_SUCCESS
from .portAndPackets import portHandler, packetHandler
import time

class DxlServoBase:
    def __init__(self, id, zeroPos, addr_table):
        self.addrTable = addr_table
        self.dxlID = id
        self.baudRateNum = 3 #1Mbps
        self.torqEn = 1
        self.torqDis = 0
        self.zeroPos = zeroPos
        self.goalPos = None
        self.presCurrent = None
        self.presPos = None
        self.presLoad = None
        self.presVel = None
        self.highTorqueCounter = 0
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_TORQUE_ENABLE'], self.torqDis)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("Dynamixel has been successfully connected")

    def setOpMode(self, opNum):
        packetHandler.write1ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_OP_MODE'], opNum)
        
    def move(self, goal_pos, prof_vel):
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_PROF_VEL'], prof_vel)
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_GOAL_POSITION'], goal_pos)

    def velMove(self, goal_vel):
        goal_raw = goal_vel & 0xFFFFFFFF
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_GOAL_VEL'], goal_raw)

    def pwmMove(self, goal_PWM):
        goal_raw = goal_PWM & 0xFFFFFFFF
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_GOAL_PWM'], goal_raw)

    def moveAndWait(self, goal_pos, prof_vel):
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_PROF_VEL'], prof_vel)
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_GOAL_POSITION'], goal_pos)
        servoMoving = True
        time.sleep(0.1)
        while servoMoving:
                presVel = self.readVel()
                if 0 <= abs(presVel) <= 1:
                    servoMoving = False

    def readPos(self):
        dxl_present_position, _, _ = packetHandler.read4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_PRESENT_POSITION'])
        self.presPos = twos_complement_to_decimal(dxl_present_position)
        return self.presPos  # Convert to decimal

    def readVel(self):
        dxl_present_velocity, _, _ = packetHandler.read4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_PRES_VEL'])
        self.presVel = twos_complement_to_decimal(dxl_present_velocity)
        return self.presVel  # Convert to decimal

    def readPosVel(self):
        """Read position and velocity in one serial round-trip.

        ADDR_PRES_VEL=128 and ADDR_PRESENT_POSITION=132 are contiguous, so 8
        bytes starting at 128 gives both in a single packet exchange.
        Returns (pos_ticks: int, vel_raw: int) — same units as readPos/readVel.
        Falls back to cached values if the read fails (communication error).
        """
        data, result, _ = packetHandler.readTxRx(
            portHandler, self.dxlID, self.addrTable['ADDR_PRES_VEL'], 8
        )
        if result != 0 or len(data) < 8:
            # Communication error — return last known values
            return (self.presPos if self.presPos is not None else 0,
                    self.presVel if self.presVel is not None else 0)
        vel_u = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
        pos_u = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24)
        self.presVel = twos_complement_to_decimal(vel_u)
        self.presPos = twos_complement_to_decimal(pos_u)
        return self.presPos, self.presVel

    def readLoadVelPos(self):
        """Read Present_Load + Present_Velocity + Present_Position in ONE packet.

        On X-series the registers are contiguous: Present_Load (126, 2B),
        Present_Velocity (128, 4B), Present_Position (132, 4B) → 10 bytes from 126.
        One serial round-trip for all three (vs 3 separate reads).
        Returns (pos_ticks, vel_raw, load_raw) — same units as readPos/readVel/readLoad.
        Falls back to cached values on a communication error.
        """
        data, result, _ = packetHandler.readTxRx(
            portHandler, self.dxlID, self.addrTable['ADDR_PRESENT_LOAD'], 10
        )
        if result != 0 or len(data) < 10:
            return (self.presPos if self.presPos is not None else 0,
                    self.presVel if self.presVel is not None else 0,
                    self.presLoad if self.presLoad is not None else 0)
        load_u = data[0] | (data[1] << 8)
        vel_u  = data[2] | (data[3] << 8) | (data[4] << 16) | (data[5] << 24)
        pos_u  = data[6] | (data[7] << 8) | (data[8] << 16) | (data[9] << 24)
        self.presLoad = twos_complement_to_decimal(load_u)   # 2B signed: -1000..1000 = ±100% torque
        self.presVel  = twos_complement_to_decimal(vel_u)
        self.presPos  = twos_complement_to_decimal(pos_u)
        return self.presPos, self.presVel, self.presLoad

    def setProfileVelocity(self, prof_vel: int):
        """Write ADDR_PROF_VEL once at startup; avoids repeating it every move."""
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_PROF_VEL'], prof_vel)

    def moveGoalOnly(self, goal_pos: int):
        """Write GOAL_POSITION only — use after setProfileVelocity() at startup."""
        packetHandler.write4ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_GOAL_POSITION'], goal_pos)
    
    def readLoad (self):
        dxl_present_current, _, _ =packetHandler.read2ByteTxRx(portHandler, self.dxlID, self.addrTable["ADDR_PRESENT_LOAD"])
        return twos_complement_to_decimal(dxl_present_current) #Convert to decimal

    def torqueDisable(self):
        packetHandler.write1ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_TORQUE_ENABLE'], self.torqDis)

    def torqueEnable(self):
        packetHandler.write1ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_TORQUE_ENABLE'], self.torqEn)

    def checkHardwareError(self):
        hardware_error, _, _ = packetHandler.read1ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_HARDWARE_ERROR_STATUS'])
        if hardware_error & 0x20:
            print(f"Servo ID {self.dxlID}: Overload Error")
            return True
        elif hardware_error & 0x10:
            print(f"Servo ID {self.dxlID}: Electrical Shock Error")
            return True
        elif hardware_error & 0x04:
            print(f"Servo ID {self.dxlID}: Overheating Error")
            return True
        elif hardware_error & 0x01:
            print(f"Servo ID {self.dxlID}: Input Voltage Error")
            return True
        else:
            return False
        
    def clearMultiTurn(self):
        packetHandler.clearMultiTurn(portHandler,self.dxlID)

    def reboot(self):
        self.presPos = self.readPos()
        dxl_comm_result, dxl_error = packetHandler.reboot(portHandler, self.dxlID)
        print("[ID:%03d] rebooted" % self.dxlID)

    def checkVoltage(self):
        present_voltage, _, _ = packetHandler.read2ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_PRES_VOLT'])
        print(f'Voltage = {present_voltage/10+0.5} V')


def twos_complement_to_decimal(value):
    if value & (1 << 31):
        return -((value ^ 0xFFFFFFFF) + 1)
    elif value & (1 << 15):
        return -((value ^ 0xFFFF) + 1)
    else:
        return value

#function not working, unable to write to this address?
'''    def changeBaudRate(self,baudRateNum):
        dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, self.dxlID, self.addrTable['ADDR_BAUD_RATE'], baudRateNum )
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % packetHandler.getRxPacketError(dxl_error))
        else:
            print("DXL Baudrate Changed!")'''