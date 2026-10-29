import queue
import threading
import time


class ImuReader:
    """Background thread that reads BNO055 gyro at ~200 Hz.

    The main 50 Hz loop calls get_imu() which returns the latest reading.
    BNO055 must be in IMUPLUS mode (6DOF) to avoid magnetometer interference
    near the motor — caller is responsible for setting mode before passing sensor.

    Axis remap — BulkheadFixed (conf=0x12, sign=0x06):
      The raw adafruit gyro tuple is (physical_X, physical_Y, physical_Z).
      We apply the same remap that field_test/openloop_swim.py writes to BNO055
      registers 0x41/0x42, but in software here so the IMUPLUS sensor mode is
      preserved (hardware remap requires CONFIG_MODE which resets fusion state).

        wx_out = +physical_Z   → field GyrX  → obs[3]  (roll / minor axis)
        wy_out = -physical_X   → field GyrY  → obs[4]  (pitch / minor axis)
        wz_out = -physical_Y   → field GyrZ  → obs[5]  YAW  ✓

      This aligns with the MuJoCo training environment (env_sofi.py line 101):
        imu_angular_vel = data.qvel[3:6]  # headRollX, headRollY, headRollZ rates
      where qvel[5] = headRollZ = yaw rate (positive = CCW from above).

      The field-to-sim EKF (run_field2sim.py line 277) also feeds Meas_GyrZ
      (= field GyrZ = -physical_Y) into EKF IMU slot [2], which maps to the
      yaw channel — confirming this is the right sign and axis.

    Usage:
        reader = ImuReader(sensor)   # sensor = adafruit_bno055.BNO055_I2C instance
        reader.start()
        imu = reader.get_imu()       # {"wx": rad/s, "wy": rad/s, "wz": rad/s}
        reader.stop()
    """

    def __init__(self, sensor, state_queue_size: int = 2):
        self._sensor = sensor
        self._q: queue.Queue = queue.Queue(maxsize=state_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="imu_reader")
        self._last_imu = {"wx": 0.0, "wy": 0.0, "wz": 0.0}

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)

    def get_imu(self) -> dict:
        try:
            self._last_imu = self._q.get_nowait()
        except queue.Empty:
            pass
        return self._last_imu

    def _run(self):
        while not self._stop_event.is_set():
            try:
                gyro = self._sensor.gyro  # raw: (physical_X, physical_Y, physical_Z)
                if gyro is None:
                    time.sleep(0.005)
                    continue
                raw_x, raw_y, raw_z = float(gyro[0]), float(gyro[1]), float(gyro[2])
                # BulkheadFixed remap — keeps axes consistent with sim and field_test:
                #   wx =  raw_z  (+physical_Z = field GyrX)
                #   wy = -raw_x  (-physical_X = field GyrY)
                #   wz = +raw_y  (+physical_Y = yaw, CCW-positive confirmed on hardware)
                data = {"wx": raw_z, "wy": -raw_x, "wz": raw_y}
                try:
                    self._q.put_nowait(data)
                except queue.Full:
                    self._q.get_nowait()
                    self._q.put_nowait(data)
            except Exception:
                pass
            time.sleep(0.005)
