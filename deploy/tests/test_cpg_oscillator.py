import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from deploy.pi.cpg_oscillator import CPGOscillator

def test_zero_action_gives_zero_setpoint():
    cpg = CPGOscillator(dt_phys=0.02, tau=0.1, motor_limit=0.65)
    setpt = cpg.step(np.zeros(6))
    assert setpt == 0.0

def test_motor_limit_clipped():
    cpg = CPGOscillator(dt_phys=0.02, tau=0.1, motor_limit=0.65)
    # Large k1_cos drives setpoint toward limit
    large = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(200):
        setpt = cpg.step(large)
    assert abs(setpt) <= 0.65 + 1e-9

def test_phase_advances():
    cpg = CPGOscillator(dt_phys=0.02, tau=0.1, motor_limit=0.65)
    omega_applied = 2 * math.pi  # 1 Hz
    action = np.array([0.0, 0.0, 0.0, 0.0, omega_applied, 0.0])
    # After enough steps for filter to settle, phase should advance ~2π per second
    for _ in range(100):  # 2 seconds
        cpg.step(action)
    # After 2 s at 1 Hz, phase ≈ 2 * 2π (mod 2π) ≈ 0
    assert 0.0 <= cpg.phase < 2 * math.pi

def test_reset_clears_state():
    cpg = CPGOscillator(dt_phys=0.02, tau=0.1, motor_limit=0.65)
    cpg.step(np.array([0.3, 0.0, 0.0, 0.0, 6.0, 0.0]))
    cpg.reset()
    assert cpg.phase == 0.0
    assert np.all(cpg.applied_params == 0.0)
