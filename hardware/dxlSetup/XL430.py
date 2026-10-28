from .DxlServoBase import DxlServoBase

class XL430(DxlServoBase):
    def __init__(self, id, zeroPos):
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
            'ADDR_PRES_VOLT': 144,
            'ADDR_GOAL_VEL': 104,
            'ADDR_GOAL_PWM': 100
        }
        super().__init__(id,zeroPos,addr_table)
