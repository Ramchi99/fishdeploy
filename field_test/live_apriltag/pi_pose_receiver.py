"""
Receives live AprilTag pose packets from the laptop over UDP.

Run this on the Pi BEFORE starting live_apriltag.py on the laptop.

Usage:
    python3 field_test/live_apriltag/pi_pose_receiver.py
    python3 field_test/live_apriltag/pi_pose_receiver.py --port 5005 --timeout 0.5
"""

import socket
import struct
import numpy as np
import time
import argparse

# Must match PACKET_FMT in live_apriltag.py exactly
PACKET_FMT = "!didddddddddddd"
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # 108 bytes

STALE_SECONDS = 0.5  # treat pose as lost if no packet received for this long


def unpack_pose(data):
    """Returns (timestamp, tag_id, t_vec, R_mat, pose_err) or raises struct.error."""
    values = struct.unpack(PACKET_FMT, data)
    ts      = values[0]
    tag_id  = values[1]
    t_vec   = np.array(values[2:5])          # (3,)  metres, camera frame
    R_mat   = np.array(values[5:14]).reshape(3, 3)  # (3,3) rotation
    pose_err = values[14]
    return ts, tag_id, t_vec, R_mat, pose_err


def main():
    parser = argparse.ArgumentParser(description="UDP pose receiver for Pi")
    parser.add_argument("--port", type=int, default=5005, help="UDP port to listen on (default 5005)")
    parser.add_argument("--timeout", type=float, default=STALE_SECONDS,
                        help="Seconds before a missing packet is flagged as 'tag lost' (default 0.5)")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(args.timeout)

    print(f"Listening on UDP port {args.port}  (Ctrl+C to stop)")
    print(f"Expecting {PACKET_SIZE}-byte packets\n")

    last_received = None

    try:
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                if last_received is not None:
                    gap = time.monotonic() - last_received
                    print(f"[WARNING] Tag lost — no packet for {gap:.2f}s")
                    last_received = None
                continue

            if len(data) != PACKET_SIZE:
                print(f"[WARNING] Unexpected packet size {len(data)}, skipping")
                continue

            ts, tag_id, t_vec, R_mat, pose_err = unpack_pose(data)
            last_received = time.monotonic()
            latency_ms = (last_received - ts) * 1000

            # ----------------------------------------------------------------
            # YOUR CONTROL CODE GOES HERE
            # t_vec = [x, y, z] position in metres (camera frame)
            # R_mat = 3x3 rotation matrix
            # Example: use t_vec[2] (depth) or t_vec[0] (lateral) for control
            # ----------------------------------------------------------------

            print(
                f"ID={tag_id:2d}  "
                f"pos=({t_vec[0]:+.3f}, {t_vec[1]:+.3f}, {t_vec[2]:+.3f}) m  "
                f"err={pose_err:.2e}  "
                f"latency={latency_ms:.1f}ms"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
