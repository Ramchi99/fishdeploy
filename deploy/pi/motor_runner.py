import math
import queue
import threading
import time

import numpy as np

TICKS_PER_REV = 4096
RAD_PER_TICK = 2 * math.pi / TICKS_PER_REV

VEL_UNIT_TO_RPM = 0.229
RPM_TO_RAD_S = 2 * math.pi / 60

MOTOR_LIMIT_RAD = 1.0

TEMP_STOP_C = 60
VOLT_WARN_RAW = 105
VOLT_STOP_RAW = 100

ADDR_PRESENT_TEMPERATURE = 146
ADDR_PRESENT_INPUT_VOLTAGE = 144


class MotorRunner:
    """Background thread that reads Dynamixel state and writes goal positions.

    Usage:
        runner = MotorRunner(servo, packetHandler)
        runner.start()              # reads zero_ticks, enables torque, starts thread
        state = runner.get_state()  # {"pos_rad": float, "vel_rad_s": float}
        runner.set_goal(0.3)        # rad, clipped to ±MOTOR_LIMIT_RAD
        runner.stop()               # torque off, thread join
    """

    def __init__(self, servo, packet_handler, port_handler=None, state_queue_size: int = 2):
        self._servo = servo
        self._ph = packet_handler
        self._port = port_handler
        self._state_q: queue.Queue = queue.Queue(maxsize=state_queue_size)
        self._cmd_q: queue.Queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run, daemon=True, name="motor_runner"
        )
        self._zero_ticks: int = 0

    def start(self):
        self._zero_ticks = self._servo.readPos()
        self._servo.torqueEnable()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._servo.torqueDisable()

    def rezero(self):
        """Re-read current encoder position as the new zero reference."""
        self._zero_ticks = self._servo.readPos()

    def set_goal(self, setpoint_rad: float):
        clipped = self._clip_setpoint(setpoint_rad)
        goal_ticks = self._zero_ticks + self._rad_to_ticks(clipped)
        try:
            self._cmd_q.put_nowait(goal_ticks)
        except queue.Full:
            self._cmd_q.get_nowait()
            self._cmd_q.put_nowait(goal_ticks)

    def get_state(self) -> dict:
        try:
            return self._state_q.get_nowait()
        except queue.Empty:
            return {"pos_rad": 0.0, "vel_rad_s": 0.0, "load_raw": 0, "load_pct": 0.0}

    def _read_state(self) -> dict:
        # Single packet: Present_Load + Present_Velocity + Present_Position (10 bytes @126).
        pos_ticks, vel_raw, load_raw = self._servo.readLoadVelPos()
        pos_rad = (pos_ticks - self._zero_ticks) * RAD_PER_TICK
        vel_rad_s = vel_raw * VEL_UNIT_TO_RPM * RPM_TO_RAD_S
        # Present_Load is -1000..1000 (0.1% units) = ±100% of max torque (proxy for torque).
        return {"pos_rad": pos_rad, "vel_rad_s": vel_rad_s,
                "load_raw": int(load_raw), "load_pct": load_raw / 10.0}

    def _check_safety(self):
        temp, _, _ = self._ph.read1ByteTxRx(
            self._port, self._servo.dxlID, ADDR_PRESENT_TEMPERATURE
        )
        if temp > TEMP_STOP_C:
            raise RuntimeError(f"[SAFETY] Motor temperature {temp}°C > {TEMP_STOP_C}°C. Stopping.")
        volt, _, _ = self._ph.read2ByteTxRx(
            self._port, self._servo.dxlID, ADDR_PRESENT_INPUT_VOLTAGE
        )
        if volt < VOLT_STOP_RAW:
            raise RuntimeError(
                f"[SAFETY] Battery {volt / 10:.1f} V < {VOLT_STOP_RAW / 10:.1f} V. Stopping."
            )
        if volt < VOLT_WARN_RAW:
            print(f"[WARN] Battery low: {volt / 10:.1f} V")

    def _clip_setpoint(self, rad: float) -> float:
        return float(np.clip(rad, -MOTOR_LIMIT_RAD, MOTOR_LIMIT_RAD))

    def _rad_to_ticks(self, rad: float) -> int:
        return round(rad / RAD_PER_TICK)

    def _run(self):
        safety_counter = 0
        while not self._stop_event.is_set():
            state = self._read_state()
            try:
                self._state_q.put_nowait(state)
            except queue.Full:
                self._state_q.get_nowait()
                self._state_q.put_nowait(state)

            try:
                goal_ticks = self._cmd_q.get_nowait()
                self._servo.move(goal_ticks, prof_vel=0)
            except queue.Empty:
                pass

            safety_counter += 1
            if safety_counter >= 50:
                safety_counter = 0
                try:
                    self._check_safety()
                except RuntimeError as e:
                    print(e)
                    self._stop_event.set()
                    self._servo.torqueDisable()
                    return

            time.sleep(0.005)
