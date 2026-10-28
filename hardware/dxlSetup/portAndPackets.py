from dynamixel_sdk import *
from configparser import ConfigParser
import os, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'portNum.ini')
config = ConfigParser()
config.read(config_path)
port = config.get('DxlConfig', 'port')
print(f"Configured port: {port}")

BAUDRATE = 1000000
PROTOCOL_VERSION = 2.0
portHandler = PortHandler(port)
packetHandler = PacketHandler(PROTOCOL_VERSION)

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()
else:
    import tty, termios
    fd = sys.stdin.fileno()
    #print(fd)
    #old_settings = termios.tcgetattr(fd)
    #print(old_settings)
    #tty.setraw(sys.stdin.fileno())
    def getch():
        return sys.stdin.read(1)

def openPortObject():
    if portHandler.openPort():
        print("Succeeded to open the port")
    else:
        print("Failed to open the port")
        print("Press any key to terminate...")
        getch()
        quit()
    if portHandler.setBaudRate(BAUDRATE):
        print("Succeeded to set the baudrate")
    else:
        print("Failed to set the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

def changeBaudRate(newBaud):
    if portHandler.setBaudRate(newBaud):
        print("Succeeded to change the baudrate")
    else:
        print("Failed to change the baudrate")
        print("Press any key to terminate...")
        getch()
        quit()

def closePortObject():
    portHandler.closePort()