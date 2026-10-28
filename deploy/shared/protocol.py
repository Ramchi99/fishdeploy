import struct

# "!dIffffffB": network byte-order
#   d = float64  timestamp        8 B
#   I = uint32   frame_seq        4 B
#   f = float32  fish_x           4 B
#   f = float32  fish_y           4 B
#   f = float32  fish_yaw         4 B
#   f = float32  target_x         4 B
#   f = float32  target_y         4 B
#   f = float32  reserved         4 B
#   B = uint8    flags            1 B
#                               ------
#                               37 B total
PACKET_FMT = "!dIffffffB"
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # must be 37

FLAGS_FISH   = 0x01
FLAGS_TARGET = 0x02
FLAGS_ANCHOR = 0x04

SYNC_PORT = 5006
POSE_PORT = 5005

CTRL_PORT = 7979

CTRL_READY = b"READY\n"  # legacy — Pi now sends READY:{name}\n
CTRL_STOP  = b"STOP\n"


def make_ready_packet(name: str) -> bytes:
    return f"READY:{name}\n".encode()


def parse_ready_name(data: bytes) -> str:
    """Parse b'READY:my_run\\n' → 'my_run'. Returns '' for legacy b'READY\\n'."""
    if data.startswith(b"READY:"):
        return data[6:].rstrip(b"\n").decode()
    return ""


def make_start_packet(ts: float) -> bytes:
    return f"START:{ts:.6f}\n".encode()


def parse_start_packet(data: bytes) -> float:
    """Parse b'START:1746000000.123456\\n' → float timestamp."""
    return float(data.split(b":")[1].strip())


def make_done_packet(ts: float) -> bytes:
    return f"DONE:{ts:.6f}\n".encode()


def parse_done_packet(data: bytes) -> float:
    return float(data.split(b":")[1].strip())


def pack_pose(
    timestamp: float,
    frame_seq: int,
    fish_x: float,
    fish_y: float,
    fish_yaw: float,
    target_x: float,
    target_y: float,
    reserved: float = 0.0,
    flags: int = 0,
) -> bytes:
    return struct.pack(
        PACKET_FMT,
        timestamp, frame_seq,
        fish_x, fish_y, fish_yaw,
        target_x, target_y, reserved,
        flags,
    )


def unpack_pose(data: bytes):
    """Returns (timestamp, frame_seq, fish_x, fish_y, fish_yaw, target_x, target_y, reserved, flags)."""
    return struct.unpack(PACKET_FMT, data)
