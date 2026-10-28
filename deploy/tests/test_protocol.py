import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from deploy.shared.protocol import pack_pose, unpack_pose, PACKET_SIZE, FLAGS_FISH, FLAGS_TARGET, FLAGS_ANCHOR

def test_packet_size():
    pkt = pack_pose(1.0, 0, 1.0, 2.0, 0.5, 3.0, 4.0)
    assert len(pkt) == 37

def test_pack_unpack_roundtrip():
    pkt = pack_pose(
        timestamp=1746000000.123456,
        frame_seq=42,
        fish_x=1.23, fish_y=-0.45, fish_yaw=0.78,
        target_x=2.0, target_y=0.5,
        reserved=0.0,
        flags=FLAGS_FISH | FLAGS_TARGET,
    )
    ts, seq, fx, fy, fyaw, tx, ty, res, fl = unpack_pose(pkt)
    assert seq == 42
    assert abs(ts - 1746000000.123456) < 1e-3   # float64 → high precision
    assert abs(fx - 1.23) < 1e-4
    assert abs(fy - (-0.45)) < 1e-4
    assert fl == FLAGS_FISH | FLAGS_TARGET

def test_flags_constants():
    assert FLAGS_FISH   == 0x01
    assert FLAGS_TARGET == 0x02
    assert FLAGS_ANCHOR == 0x04
