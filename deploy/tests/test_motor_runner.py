import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import time
import threading
from unittest.mock import MagicMock, patch
from deploy.pi.motor_runner import MotorRunner, MOTOR_LIMIT_RAD

def _make_mock_servo(pos_ticks=2048, vel_raw=10, temp=35, voltage_raw=120):
    servo = MagicMock()
    servo.readPos.return_value = pos_ticks
    servo.readVel.return_value = vel_raw
    servo.readLoadVelPos.return_value = (pos_ticks, vel_raw, 0)
    servo.dxlID = 1
    return servo

def _make_mock_packethandler(temp=35, voltage_raw=120):
    ph = MagicMock()
    ph.read1ByteTxRx.return_value = (temp, 0, 0)
    ph.read2ByteTxRx.return_value = (voltage_raw, 0, 0)
    return ph

def test_motor_state_pos_rad():
    import math
    servo = _make_mock_servo(pos_ticks=2048)
    runner = MotorRunner(servo, MagicMock(), MagicMock())
    runner._zero_ticks = 2048
    state = runner._read_state()
    assert abs(state["pos_rad"]) < 1e-6   # at zero

def test_motor_state_vel_rad_s():
    servo = _make_mock_servo(vel_raw=100)
    runner = MotorRunner(servo, MagicMock(), MagicMock())
    runner._zero_ticks = 0
    state = runner._read_state()
    # 100 units × 0.229 RPM/unit × 2π/60 ≈ 2.398 rad/s
    assert abs(state["vel_rad_s"] - 100 * 0.229 * 2 * 3.14159 / 60) < 0.01

def test_setpoint_clipped_to_limit():
    servo = _make_mock_servo()
    runner = MotorRunner(servo, MagicMock(), MagicMock())
    runner._zero_ticks = 0
    # Should clip to ±MOTOR_LIMIT_RAD
    clipped = runner._clip_setpoint(10.0)
    assert clipped == MOTOR_LIMIT_RAD
    clipped = runner._clip_setpoint(-10.0)
    assert clipped == -MOTOR_LIMIT_RAD

def test_setpoint_to_ticks():
    import math
    servo = _make_mock_servo(pos_ticks=1000)
    runner = MotorRunner(servo, MagicMock(), MagicMock())
    runner._zero_ticks = 1000
    ticks = runner._rad_to_ticks(math.pi / 2)
    assert abs(ticks - round(math.pi / 2 * 4096 / (2 * math.pi))) < 2

def test_thread_starts_and_stops():
    servo = _make_mock_servo()
    runner = MotorRunner(servo, MagicMock(), MagicMock())
    runner._zero_ticks = 2048
    runner.start()
    time.sleep(0.05)
    runner.stop()
    assert not runner._thread.is_alive()
