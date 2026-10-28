"""The 24-D observation of the deployed policy (ObsAssemblerCPG24D), checked against the training layout:
[0-1] cos/sin motor, [2] motor velocity, [3] yaw rate, [4-5] cos/sin CPG phase,
[6-17] three vision frames (dist, vec_x, vec_y, age), oldest first, [18-23] three past actions (omega, theta_c)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import time
import numpy as np
from deploy.pi.obs_assembler import ObsAssemblerCPG24D, VisionState


def _config():
    return {"vision_stale_warn_ms": 100, "vision_stale_stop_ms": 500, "tag0_to_com_offset": [0.0, 0.0]}


def _motor(pos=0.1, vel=0.5):
    return {"pos_rad": pos, "vel_rad_s": vel}


def _imu(wz=0.03):
    return {"wx": 0.01, "wy": 0.02, "wz": wz}


def _vision(fish_x=1.0, fish_y=0.5, fish_yaw=0.0, target_x=2.0, target_y=1.0, frame_seq=1, t=None):
    t = time.time() if t is None else t
    return VisionState(fish_x=fish_x, fish_y=fish_y, fish_yaw=fish_yaw, target_x=target_x, target_y=target_y,
                       flags=0x03, pi_arrival_time=t, laptop_timestamp=t, frame_seq=frame_seq)


def _build(asm, vision, cpg_phase=0.0, omega=0.0, theta=0.0, motor=None, imu=None):
    return asm.build(motor or _motor(), imu or _imu(), vision, cpg_phase, omega, theta)


def test_obs_is_24d_float32():
    obs = _build(ObsAssemblerCPG24D(_config()), _vision())
    assert obs.shape == (24,) and obs.dtype == np.float32


def test_motor_and_phase_are_circular():
    obs = _build(ObsAssemblerCPG24D(_config()), _vision(), cpg_phase=1.0, motor=_motor(pos=0.3))
    assert np.isclose(obs[0], np.cos(0.3)) and np.isclose(obs[1], np.sin(0.3))
    assert np.isclose(obs[2], 0.5)
    assert np.isclose(obs[4], np.cos(1.0)) and np.isclose(obs[5], np.sin(1.0))


def test_only_yaw_rate_enters():
    obs = _build(ObsAssemblerCPG24D(_config()), _vision(), imu=_imu(wz=0.42))
    assert np.isclose(obs[3], 0.42)


def test_latest_frame_is_body_frame_target():
    # fish at (1, 0.5) heading +90 deg, target at (1, 1.5): one metre straight ahead
    obs = _build(ObsAssemblerCPG24D(_config()), _vision(fish_yaw=np.pi / 2, target_x=1.0, target_y=1.5))
    dist, vec_x, vec_y, age = obs[14:18]
    assert np.isclose(dist, 1.0, atol=1e-5)
    assert np.isclose(vec_x, 1.0, atol=1e-5) and np.isclose(vec_y, 0.0, atol=1e-5)
    assert 0.0 <= age < 0.05


def test_frame_buffer_pads_with_sentinels_and_shifts_per_frame():
    asm = ObsAssemblerCPG24D(_config())
    obs = _build(asm, _vision(frame_seq=1))
    # two sentinel frames (zeros, age 0.2 s) in front of the real one
    assert np.allclose(obs[6:9], 0.0) and np.isclose(obs[9], 0.2, atol=1e-3)
    assert np.allclose(obs[10:13], 0.0) and np.isclose(obs[13], 0.2, atol=1e-3)
    assert obs[14] > 0.0
    # the same frame again does not shift the buffer; a new frame_seq does
    obs2 = _build(asm, _vision(frame_seq=1))
    assert np.allclose(obs2[10:13], 0.0)
    obs3 = _build(asm, _vision(frame_seq=2, target_x=3.0))
    assert obs3[10] > 0.0 and not np.isclose(obs3[10], obs3[14])


def test_action_history_is_oldest_first():
    asm = ObsAssemblerCPG24D(_config())
    _build(asm, _vision(frame_seq=1), omega=1.0, theta=0.1)
    _build(asm, _vision(frame_seq=2), omega=2.0, theta=0.2)
    obs = _build(asm, _vision(frame_seq=3), omega=3.0, theta=0.3)
    assert np.allclose(obs[18:24], [1.0, 0.1, 2.0, 0.2, 3.0, 0.3])


def test_com_offset_moves_the_fish_along_its_heading():
    cfg = _config(); cfg["tag0_to_com_offset"] = [0.1, 0.0]
    obs = _build(ObsAssemblerCPG24D(cfg), _vision(fish_yaw=0.0, fish_x=0.0, fish_y=0.0, target_x=1.0, target_y=0.0))
    assert np.isclose(obs[14], 0.9, atol=1e-5)          # the CoM sits 0.1 m ahead of the tag


def test_vision_status_thresholds():
    asm = ObsAssemblerCPG24D(_config())
    now = time.time()
    assert asm.vision_status(_vision(t=now)) == "nominal"
    assert asm.vision_status(_vision(t=now - 0.2)) == "stale"
    assert asm.vision_status(_vision(t=now - 0.6)) == "lost"


def test_clock_offset_enters_the_frame_age():
    asm = ObsAssemblerCPG24D(_config())
    asm.set_clock_offset(0.15)                            # laptop clock 150 ms ahead
    obs = _build(asm, _vision(t=time.time()))
    assert np.isclose(obs[17], 0.15, atol=0.02)
