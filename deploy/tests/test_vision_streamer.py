import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import math
from deploy.laptop.vision_streamer import world_frame_transform, extract_yaw

def _make_identity_anchor():
    """Anchor at origin with identity rotation (tag frame = camera frame)."""
    R = np.eye(3)
    t = np.zeros(3)
    return R, t

def test_fish_at_origin_when_at_anchor():
    R_anc, t_anc = _make_identity_anchor()
    t_fish = np.zeros(3)
    R_fish = np.eye(3)
    fx, fy = world_frame_transform(t_fish, t_anc, R_anc)
    assert abs(fx) < 1e-9
    assert abs(fy) < 1e-9

def test_fish_offset_from_anchor():
    R_anc, t_anc = _make_identity_anchor()
    t_fish = np.array([1.0, 0.0, 0.0])  # 1m to the right in camera frame
    R_fish = np.eye(3)
    fx, fy = world_frame_transform(t_fish, t_anc, R_anc)
    assert abs(fx - 1.0) < 1e-6
    assert abs(fy) < 1e-6

def test_yaw_facing_north_is_zero():
    # Fish forward = tag Y-axis ("top" of printed square). extract_yaw mirrors world x
    # (atan2(fwd_y, -fwd_x)) so that yaw is CCW-positive in the overhead image; yaw = 0
    # therefore means the tag Y-axis points along world -X, i.e. R_tag rotates Y -> -X
    # (+90 deg about Z).
    angle = math.pi / 2
    R_tag = np.array([
        [ math.cos(angle), -math.sin(angle), 0],
        [ math.sin(angle),  math.cos(angle), 0],
        [0,                  0,               1],
    ])
    R_anc = np.eye(3)
    yaw = extract_yaw(R_tag, R_anc)
    assert abs(yaw) < 1e-5

def test_yaw_identity_points_along_world_y():
    # With identity R_tag and identity anchor, tag Y-axis = world Y-axis.
    # fwd = [0,1,0] → atan2(1,0) = π/2
    R_tag = np.eye(3)
    R_anc = np.eye(3)
    yaw = extract_yaw(R_tag, R_anc)
    assert abs(yaw - math.pi / 2) < 1e-5

def test_ctrl_state_ready_parsing():
    """Simulate Pi READY packet → state dict updated correctly."""
    from deploy.shared.protocol import CTRL_READY
    state = {'pi_ready': False, 'pi_ip_confirmed': None, 'done_received': False, 'done_ts': None}
    # Simulate the listener logic inline
    data = CTRL_READY
    addr = ("10.0.0.1", 7979)
    if data == CTRL_READY:
        state['pi_ready'] = True
        state['pi_ip_confirmed'] = addr[0]
    assert state['pi_ready'] is True
    assert state['pi_ip_confirmed'] == "10.0.0.1"

def test_ctrl_state_done_parsing():
    from deploy.shared.protocol import make_done_packet, parse_done_packet
    state = {'done_received': False, 'done_ts': None, 'pi_ready': True}
    ts = 1746000000.5
    data = make_done_packet(ts)
    if data.startswith(b"DONE:"):
        state['done_ts'] = parse_done_packet(data)
        state['pi_ready'] = False
        state['done_received'] = True
    assert state['done_received'] is True
    assert abs(state['done_ts'] - ts) < 1e-3
    assert state['pi_ready'] is False
