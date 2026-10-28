import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
from unittest.mock import MagicMock, patch
from deploy.pi.policy_runner import PolicyRunner, scale_action

def test_scale_action_k1():
    # k1_cos channel: raw=1.0 → 0.30 rad
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scaled = scale_action(a)
    assert abs(scaled[0] - 0.30) < 1e-6

def test_scale_action_omega_half_rectified():
    import math
    # omega channel: raw=1.0 → 0.5*(1+1)*12π = 12π rad/s
    a = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    scaled = scale_action(a)
    assert abs(scaled[4] - 12 * math.pi) < 1e-5

def test_scale_action_omega_at_minus1():
    # raw=-1.0 → 0.5*(−1+1)*12π = 0
    a = np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    scaled = scale_action(a)
    assert abs(scaled[4]) < 1e-6

def test_scale_action_theta_center():
    # theta_center: raw=1.0 → 0.15 rad
    a = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    scaled = scale_action(a)
    assert abs(scaled[5] - 0.15) < 1e-6

def test_predict_returns_6d():
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.zeros(6, dtype=np.float32), None)
    runner = PolicyRunner.__new__(PolicyRunner)
    runner._model = mock_model
    obs = np.zeros(14, dtype=np.float32)
    action, scaled = runner.predict(obs)
    assert action.shape == (6,)
    assert scaled.shape == (6,)
