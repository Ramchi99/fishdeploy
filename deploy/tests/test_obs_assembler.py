import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import time
from deploy.pi.obs_assembler import ObsAssemblerCPG14D, VisionState

def _default_config():
    return {
        "vision_stale_warn_ms": 100,
        "vision_stale_stop_ms": 500,
        "tag0_to_com_offset": [0.0, 0.0],
    }

def _make_motor():
    return {"pos_rad": 0.1, "vel_rad_s": 0.5}

def _make_imu():
    return {"wx": 0.01, "wy": 0.02, "wz": 0.03}

def _make_vision(fish_x=1.0, fish_y=0.5, fish_yaw=0.2, target_x=2.0, target_y=1.0,
                  flags=0x03, arrival_time=None):
    return VisionState(
        fish_x=fish_x, fish_y=fish_y, fish_yaw=fish_yaw,
        target_x=target_x, target_y=target_y,
        flags=flags,
        pi_arrival_time=arrival_time if arrival_time is not None else time.time(),
        laptop_timestamp=arrival_time if arrival_time is not None else time.time(),
        frame_seq=1,
    )

def test_obs_shape_is_14d():
    asm = ObsAssemblerCPG14D(_default_config())
    cpg_phase = 0.0
    prev_omega = 0.0
    obs = asm.build(_make_motor(), _make_imu(), _make_vision(), cpg_phase, prev_omega)
    assert obs.shape == (14,)

def test_motor_circular_encoding():
    import math
    asm = ObsAssemblerCPG14D(_default_config())
    motor = {"pos_rad": math.pi / 3, "vel_rad_s": 1.0}
    obs = asm.build(motor, _make_imu(), _make_vision(), 0.0, 0.0)
    assert abs(obs[0] - math.cos(math.pi / 3)) < 1e-6  # cos(motor_pos)
    assert abs(obs[1] - math.sin(math.pi / 3)) < 1e-6  # sin(motor_pos)
    assert abs(obs[2] - 1.0) < 1e-6                    # motor_vel

def test_imu_in_obs():
    asm = ObsAssemblerCPG14D(_default_config())
    imu = {"wx": 0.1, "wy": 0.2, "wz": 0.3}
    obs = asm.build(_make_motor(), imu, _make_vision(), 0.0, 0.0)
    assert abs(obs[3] - 0.1) < 1e-6
    assert abs(obs[4] - 0.2) < 1e-6
    assert abs(obs[5] - 0.3) < 1e-6

def test_dist_target_computed_correctly():
    import math
    asm = ObsAssemblerCPG14D(_default_config())
    # fish at (0,0), target at (3,4) → dist=5
    vision = _make_vision(fish_x=0.0, fish_y=0.0, fish_yaw=0.0, target_x=3.0, target_y=4.0)
    obs = asm.build(_make_motor(), _make_imu(), vision, 0.0, 0.0)
    assert abs(obs[6] - 5.0) < 1e-5

def test_vec_target_body_frame():
    import math
    asm = ObsAssemblerCPG14D(_default_config())
    # fish at origin facing east (yaw=0), target directly ahead at (1,0)
    vision = _make_vision(fish_x=0.0, fish_y=0.0, fish_yaw=0.0, target_x=1.0, target_y=0.0)
    obs = asm.build(_make_motor(), _make_imu(), vision, 0.0, 0.0)
    assert abs(obs[7] - 1.0) < 1e-5   # vec_target_x (forward) = 1.0
    assert abs(obs[8] - 0.0) < 1e-5   # vec_target_y (lateral) = 0.0

def test_cpg_phase_circular_encoding():
    import math
    asm = ObsAssemblerCPG14D(_default_config())
    phase = math.pi / 4
    obs = asm.build(_make_motor(), _make_imu(), _make_vision(), phase, 0.0)
    assert abs(obs[11] - math.cos(phase)) < 1e-6
    assert abs(obs[12] - math.sin(phase)) < 1e-6

def test_prev_omega_in_obs():
    asm = ObsAssemblerCPG14D(_default_config())
    obs = asm.build(_make_motor(), _make_imu(), _make_vision(), 0.0, prev_omega=0.7)
    assert abs(obs[13] - 0.7) < 1e-6

def test_stale_vision_status():
    asm = ObsAssemblerCPG14D(_default_config())
    age_s = 0.5 * (asm._stale_warn_ms + asm._stale_stop_ms) / 1000.0  # between warn and stop → STALE
    old_vision = _make_vision(arrival_time=time.time() - age_s)
    status = asm.vision_status(old_vision)
    assert status == "stale"

def test_lost_vision_status():
    asm = ObsAssemblerCPG14D(_default_config())
    lost_vision = _make_vision(arrival_time=time.time() - 0.6)  # 600ms ago → LOST
    status = asm.vision_status(lost_vision)
    assert status == "lost"

def test_vel_vec_zeroed_when_stale_200ms():
    asm = ObsAssemblerCPG14D(_default_config())
    stale_vision = _make_vision(arrival_time=time.time() - 0.25)  # 250ms > 200ms
    obs = asm.build(_make_motor(), _make_imu(), stale_vision, 0.0, 0.0)
    assert obs[9] == 0.0   # vel_vec_target_x zeroed
    assert obs[10] == 0.0  # vel_vec_target_y zeroed
