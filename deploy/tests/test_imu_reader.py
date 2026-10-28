import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import time
from unittest.mock import MagicMock
from deploy.pi.imu_reader import ImuReader

def _make_mock_sensor(gyro=(0.1, 0.2, 0.3)):
    sensor = MagicMock()
    sensor.gyro = gyro
    return sensor

def test_get_imu_returns_dict():
    sensor = _make_mock_sensor(gyro=(0.1, 0.2, 0.3))
    reader = ImuReader(sensor)
    reader._zero_gyro = (0.0, 0.0, 0.0)
    data = reader.get_imu()
    assert "wx" in data and "wy" in data and "wz" in data

def test_gyro_values_correct():
    sensor = _make_mock_sensor(gyro=(0.1, 0.2, 0.3))
    reader = ImuReader(sensor)
    reader._zero_gyro = (0.0, 0.0, 0.0)
    reader._last_imu = {"wx": 0.1, "wy": 0.2, "wz": 0.3}
    data = reader.get_imu()
    assert abs(data["wx"] - 0.1) < 1e-6
    assert abs(data["wy"] - 0.2) < 1e-6
    assert abs(data["wz"] - 0.3) < 1e-6

def test_thread_starts_and_stops():
    sensor = _make_mock_sensor()
    reader = ImuReader(sensor)
    reader.start()
    time.sleep(0.05)
    reader.stop()
    assert not reader._thread.is_alive()

def test_none_gyro_returns_zeros():
    sensor = _make_mock_sensor(gyro=None)
    reader = ImuReader(sensor)
    reader._zero_gyro = (0.0, 0.0, 0.0)
    reader._last_imu = {"wx": 0.0, "wy": 0.0, "wz": 0.0}
    data = reader.get_imu()
    assert data["wx"] == 0.0
