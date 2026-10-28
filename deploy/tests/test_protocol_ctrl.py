import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deploy.shared.protocol import (
    CTRL_PORT, CTRL_READY, CTRL_STOP,
    make_start_packet, parse_start_packet,
    make_done_packet, parse_done_packet,
)

def test_ctrl_port_value():
    assert CTRL_PORT == 7979

def test_ctrl_ready_constant():
    assert CTRL_READY == b"READY\n"

def test_ctrl_stop_constant():
    assert CTRL_STOP == b"STOP\n"

def test_start_packet_format():
    pkt = make_start_packet(1746000000.123456)
    assert pkt.startswith(b"START:")
    assert pkt.endswith(b"\n")

def test_start_packet_roundtrip():
    ts = 1746000000.123456
    recovered = parse_start_packet(make_start_packet(ts))
    assert abs(recovered - ts) < 1e-3

def test_done_packet_roundtrip():
    ts = 1746000099.987654
    recovered = parse_done_packet(make_done_packet(ts))
    assert abs(recovered - ts) < 1e-3
