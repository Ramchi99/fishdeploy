from .DxlServoBase import DxlServoBase

class XC330(DxlServoBase):
    def __init__(self, id, zeroPos, shortBool=False):
        addr_table = {
            'ADDR_TORQUE_ENABLE': 64,
            'ADDR_GOAL_POSITION': 116,
            'ADDR_PRESENT_POSITION': 132,
            'ADDR_PROF_VEL': 112,
            'ADDR_PRES_VEL': 128,
            'ADDR_OP_MODE': 11,
            'ADDR_HARDWARE_ERROR_STATUS': 70,
            'ADDR_PRESENT_LOAD': 126,
            'ADDR_BAUD_RATE': 8,
            'ADDR_PRES_VOLT': 144
        }
        super().__init__(id,zeroPos,addr_table)
        self.actuationDistance = 3300 if shortBool else 7000
        self.newActDist = 7000 if shortBool else 9500
