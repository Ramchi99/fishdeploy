#!/usr/bin/env python3
import time, math, os, csv, argparse, sys, select
from datetime import datetime

# --- PATHING SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(SCRIPT_DIR, "..")))   # repository root
from paths import FIELD_LOGS  # noqa: E402
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
DXL_DIR = os.path.join(REPO_ROOT, 'hardware')
sys.path.append(REPO_ROOT); sys.path.append(DXL_DIR)

try:
    from dxlSetup.portAndPackets import openPortObject, closePortObject, portHandler, packetHandler
    from dxlSetup.XL430 import XL430
    from dynamixel_sdk import GroupSyncWrite, GroupBulkRead, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD
    MOTOR_AVAILABLE = True
except ImportError: MOTOR_AVAILABLE = False

try:
    import board; import busio; import adafruit_bno055; IMU_AVAILABLE = True
except ImportError: IMU_AVAILABLE = False

try:
    import smbus
    UPS_AVAILABLE = True
except ImportError: UPS_AVAILABLE = False

try:
    import numpy as np; import matplotlib.pyplot as plt; ANALYSIS_AVAILABLE = True
except ImportError: ANALYSIS_AVAILABLE = False

# --- UPS MONITORING CLASS (INA219) ---
class UPS_INA219:
    def __init__(self, i2c_bus=0, addr=0x41):
        self.bus = smbus.SMBus(i2c_bus)
        self.addr = addr
        # Set calibration for 16V 5A
        self.bus.write_i2c_block_data(self.addr, 0x05, [0x68, 0xF4]) # CALIBRATION
        self.bus.write_i2c_block_data(self.addr, 0x00, [0x39, 0x9F]) # CONFIG

    def get_bus_voltage_v(self):
        data = self.bus.read_i2c_block_data(self.addr, 0x02, 2)
        val = (data[0] * 256) + data[1]
        return (val >> 3) * 0.004

    def get_current_ma(self):
        data = self.bus.read_i2c_block_data(self.addr, 0x04, 2)
        val = (data[0] * 256) + data[1]
        if val > 32767: val -= 65535
        return val * 0.1524 # LSB based on your sample script

def twos_complement(val, bits):
    if (val & (1 << (bits - 1))) != 0: val = val - (1 << bits)
    return val

class SofiUnifiedTest:
    def __init__(self, args):
        self.args = args
        self.data_log = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_dir = str(FIELD_LOGS); self.plot_dir = str(FIELD_LOGS / "plots")
        os.makedirs(self.data_dir, exist_ok=True); os.makedirs(self.plot_dir, exist_ok=True)
        self.filename = os.path.join(self.data_dir, f"{self.timestamp}_{args.name}.csv")
        self.servo = None; self.sensor = None; self.ups = None; self.motor_enabled = False; self.baseline_euler = [0.0, 0.0, 0.0]

    def set_imu_map(self, conf, sign):
        if not self.sensor: return
        self.sensor.mode = adafruit_bno055.CONFIG_MODE; time.sleep(0.2)
        self.sensor._write_register(0x41, conf); self.sensor._write_register(0x42, sign); time.sleep(0.2)
        self.sensor.mode = adafruit_bno055.NDOF_MODE if self.args.imu_mode == "9dof" else adafruit_bno055.IMUPLUS_MODE
        time.sleep(1.0) 

    def setup_hardware(self):
        print("\n[1/4] Initializing Hardware...")
        if MOTOR_AVAILABLE:
            openPortObject(); self.servo = XL430(id=1, zeroPos=0); self.servo.setOpMode(4); self.servo.torqueDisable()
            self.groupWrite = GroupSyncWrite(portHandler, packetHandler, self.servo.addrTable['ADDR_GOAL_POSITION'], 4)
            self.groupRead = GroupBulkRead(portHandler, packetHandler); self.groupRead.addParam(self.servo.dxlID, 126, 21)
        if IMU_AVAILABLE:
            try:
                i2c = busio.I2C(board.SCL, board.SDA); self.sensor = adafruit_bno055.BNO055_I2C(i2c)
                self.set_imu_map(0x12, 0x06) # Bulkhead Fixed
                print(f"  - IMU Ready.")
            except Exception as e: print(f"  ! IMU Failed: {e}"); self.sensor = None
        if UPS_AVAILABLE:
            try:
                self.ups = UPS_INA219(i2c_bus=0, addr=0x41)
                print(f"  - UPS Module Detected.")
            except Exception as e: print(f"  ! UPS Init Failed: {e}"); self.ups = None

    def calibrate_imu(self):
        if not self.sensor or self.args.skip_calib: return
        print("\n[2/4] CALIBRATION PHASE\n Commands: 't'+Enter (Motor Toggle) | 's'+Enter (Skip to Test)")
        stable_start = None
        while True:
            now_unix = time.time()
            sys_s, gyro, accel, mag = self.sensor.calibration_status
            abs_e = self.sensor.euler
            p_val, r_val = (abs_e[2], abs_e[1]) if abs_e and abs_e[2] is not None else (0,0)
            
            # Read UPS status for dashboard
            bat_v = self.ups.get_bus_voltage_v() if self.ups else 0
            bat_ma = self.ups.get_current_ma() if self.ups else 0
            
            is_full = (gyro == 3 and accel == 3 and sys_s == 3)
            if self.args.imu_mode == "9dof": is_full = is_full and (mag == 3)
            if is_full:
                if stable_start is None: stable_start = time.time()
                elapsed = time.time() - stable_start
                if elapsed >= 3.0: print(f"\n  - Ready! Proceeding..."); break
                msg = f"STABLE ({elapsed:.1f}s)"
            else: stable_start = None; msg = "Waiting..."

            print(f"  [TIME:{now_unix:17.6f}] [BAT:{bat_v:5.2f}V {bat_ma/1000:5.2f}A] S:{sys_s} G:{gyro} A:{accel} M:{mag} | {msg} \033[K", end="\r")
            
            i, _, _ = select.select([sys.stdin], [], [], 0.1)
            if i:
                line = sys.stdin.readline().strip().lower()
                if line == 't':
                    if self.motor_enabled: self.servo.torqueDisable(); self.motor_enabled = False
                    else: self.servo.torqueEnable(); self.motor_enabled = True
                elif line == 's' or line == '': break
            time.sleep(0.02)

    def run_test(self):
        if not self.servo: return
        print("\n" + "="*50 + "\n [READY] Place fish in water. Press ENTER to start.\n" + "="*50); input()
        if not self.motor_enabled: self.servo.torqueEnable(); self.motor_enabled = True; time.sleep(0.5)
        if self.sensor:
            e = self.sensor.euler
            if e and all(v is not None for v in e): self.baseline_euler = list(e)

        print(f"\n[3/4] Running: {self.args.freq}Hz | {self.args.duration}s")
        amp_ticks = int((self.args.amp / (2 * math.pi * self.args.radius)) * 4096)
        offset_ticks = int((self.args.offset / (2 * math.pi * self.args.radius)) * 4096)
        center_pos = self.servo.readPos(); target_center = center_pos + offset_ticks
        
        start_t = time.time(); next_b = start_t; period = 1.0 / self.args.rate; prev_t = start_t
        try:
            while (time.time() - start_t) < self.args.duration:
                now = time.time(); elapsed = now - start_t
                hz = 1.0 / (now - prev_t) if now > prev_t else 0; prev_t = now
                target = int(target_center + amp_ticks * math.sin(2 * math.pi * self.args.freq * elapsed))
                param = [DXL_LOBYTE(DXL_LOWORD(target)), DXL_HIBYTE(DXL_LOWORD(target)), DXL_LOBYTE(DXL_HIWORD(target)), DXL_HIBYTE(DXL_HIWORD(target))]
                self.groupWrite.clearParam(); self.groupWrite.addParam(self.servo.dxlID, param); self.groupWrite.txPacket()
                
                self.groupRead.txRxPacket()
                m_data = [0]*5
                if self.groupRead.isAvailable(self.servo.dxlID, 126, 21):
                    m_data = [twos_complement(self.groupRead.getData(self.servo.dxlID, 132, 4), 32),
                              twos_complement(self.groupRead.getData(self.servo.dxlID, 128, 4), 32),
                              twos_complement(self.groupRead.getData(self.servo.dxlID, 126, 2), 16),
                              self.groupRead.getData(self.servo.dxlID, 144, 2)/10.0,
                              self.groupRead.getData(self.servo.dxlID, 146, 1)]
                
                imu_raw = [0.0]*16
                if self.sensor:
                    q, e, a, g, m = self.sensor.quaternion, self.sensor.euler, self.sensor.linear_acceleration, self.sensor.gyro, self.sensor.magnetic
                    if all(v is not None for v in [q, e, a, g, m]): imu_raw = list(q) + list(e) + list(a) + list(g) + list(m)

                # Read UPS (Battery)
                ups_data = [0.0, 0.0, 0.0]
                if self.ups:
                    v = self.ups.get_bus_voltage_v()
                    ma = self.ups.get_current_ma()
                    pct = (v - 9)/3.6*100
                    ups_data = [round(v, 2), round(ma/1000.0, 3), round(max(0, min(100, pct)), 1)]

                self.data_log.append([now, round(elapsed, 4), round(hz, 1), target] + m_data + imu_raw + ups_data)
                next_b += period; slp = next_b - time.time()
                if slp > 0: time.sleep(slp)
        except KeyboardInterrupt: print("\n  - Interrupted.")
        
        if len(self.data_log) > 0:
            avg_hz = sum([row[2] for row in self.data_log]) / len(self.data_log)
            print(f"  - Test Finished. Average Logging Rate: {avg_hz:.1f} Hz")
        self.servo.move(goal_pos=int(center_pos), prof_vel=266); self.servo.torqueDisable(); closePortObject()

    def save_and_viz(self):
        print("\n[4/4] Saving..."); headers = ['Unix_Time', 'Time_s', 'Hz', 'Goal', 'Actual', 'Vel', 'Load', 'Volt', 'Temp', 
                   'QW','QX','QY','QZ', 'RawY','RawR','RawP', 'RelY','RelR','RelP', 'AccX','AccY','AccZ', 'GyrX','GyrY','GyrZ', 'MagX','MagY','MagZ',
                   'Battery_V', 'Battery_A', 'Battery_Pct']
        final = []
        for r in self.data_log:
            y, roll, p = r[13], r[14], r[15]
            ry = (y - self.baseline_euler[0] + 180) % 360 - 180
            rr = roll - self.baseline_euler[1]
            rp = (p - self.baseline_euler[2] + 180) % 360 - 180
            final.append([f"{r[0]:.6f}"] + r[1:16] + [ry, rr, rp] + r[16:])
            
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f); writer.writerow(['# Metadata:', f"Remap=BulkheadFixed, Mode={self.args.imu_mode}"]); writer.writerow(headers); writer.writerows(final)

        if ANALYSIS_AVAILABLE and len(final) > 0:
            d = np.array(final, dtype=float); t = d[:, 1]
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            ax1.plot(t, d[:, 3], 'r--', label='Goal'); ax1.plot(t, d[:, 4], 'b-', label='Actual'); ax1.set_ylabel('Motor'); ax1.grid(True)
            ax2.plot(t, d[:, 16], 'k-', label='Yaw'); ax2.set_ylabel('Yaw (Deg)'); ly = max(np.abs(d[:, 16]).max()*1.1, 5); ax2.set_ylim(-ly, ly); ax2.grid(True)
            ax3.plot(t, d[:, 17], 'b-', label='Roll'); ax3.plot(t, d[:, 18], 'g-', label='Rel Pitch'); ax3.set_ylabel('R/P (Deg)'); ax3.set_xlabel('Time (s)')
            lrp = max(np.abs(d[:, 17:19]).max()*1.1, 5); ax3.set_ylim(-lrp, lrp); ax3.grid(True); plt.tight_layout(); plt.savefig(os.path.join(self.plot_dir, f"{self.timestamp}_{self.args.name}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="test_run")
    parser.add_argument("--freq", type=float, default=1.0); parser.add_argument("--amp", type=float, default=5.0); parser.add_argument("--offset", type=float, default=0.0); parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--p", type=int, default=800); parser.add_argument("--rate", type=int, default=100); parser.add_argument("--radius", type=float, default=12.5)
    parser.add_argument("--imu-mode", type=str, default="6dof", choices=["6dof", "9dof"]); parser.add_argument("--skip-calib", action="store_true")
    args = parser.parse_args(); test = SofiUnifiedTest(args)
    test.setup_hardware(); test.calibrate_imu(); test.run_test(); test.save_and_viz()
