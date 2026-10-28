"""
Entry point for RL policy deployment on Raspberry Pi 5.

Usage (on Pi, from fishdeploy/ repo root):
    python3 deploy/pi/deploy_main.py \
        --name pool_test_1 \
        --env-config deploy/configs/cpg_24d.json \
        --model policies/sofi_cpg_rl_policy.zip \
        --duration 60 \
        [--record] [--finetune]
"""
import argparse
import hashlib
import json
import os
import queue
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DXL_DIR = os.path.join(_REPO_ROOT, 'hardware')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _DXL_DIR)

from deploy.pi.cpg_oscillator import CPGOscillator
from deploy.pi.data_logger import DataLogger
from deploy.pi.imu_reader import ImuReader
from deploy.pi.motor_runner import MotorRunner
from deploy.pi.obs_assembler import ObsAssemblerCPG12D, ObsAssemblerCPG14D, ObsAssemblerCPG24D, ObsAssemblerDA40D, VisionState
from deploy.pi.policy_runner import PolicyRunner
from deploy.shared.protocol import (
    PACKET_SIZE, POSE_PORT, SYNC_PORT, CTRL_PORT,
    FLAGS_FISH, FLAGS_TARGET,
    CTRL_STOP,
    make_ready_packet, make_start_packet, parse_start_packet, make_done_packet,
    unpack_pose,
)

CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
CPG_TAU = 0.1
MOTOR_LIMIT = 0.65
STARTUP_TIMEOUT_S = 120.0
VISION_LOSS_ABORT_S = 5.0


def compute_reward(obs: np.ndarray, action: np.ndarray) -> float:
    dist = float(obs[6])
    action_penalty = -0.1 * float(np.linalg.norm(action))
    success_bonus = 1000.0 if dist < 0.05 else 0.0
    return -dist + action_penalty + success_bonus


def make_run_dir(data_root: str, name: str, ts: float = None) -> str:
    if ts is None:
        ts = time.time()
    ts_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
    return os.path.join(data_root, f"{ts_str}_{name}")


def _model_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _clock_sync_handshake(laptop_ip: str, sync_port: int, n_pings: int = 20,
                          max_retries: int = 3) -> float:
    """Pi-initiated NTP-style sync — Pi sends pings, laptop server responds.

    The laptop vision_streamer runs a persistent responder, so this can be
    called any time after the Pi receives its first pose packet (which gives
    us the laptop IP). Selects minimum-RTT sample.

    If all pings time out (laptop server not yet ready), retries up to
    max_retries times with a 1s pause. Returns offset in ms (Pi − laptop).
    Observed precision: ±10–25 ms within a session (well under one 20 ms step).
    The offset drifts ~75 ms/hour (Pi clock vs NTP-synced laptop) but is
    re-measured fresh at the start of every run, so drift is irrelevant.
    """
    for attempt in range(max_retries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(2.0)
        samples = []
        try:
            for _ in range(n_pings):
                try:
                    t1 = time.time()
                    sock.sendto(struct.pack("!d", t1), (laptop_ip, sync_port))
                    data, _ = sock.recvfrom(16)
                    t3 = time.time()
                    t1_echo, t2 = struct.unpack("!dd", data)
                    rtt_ms = (t3 - t1) * 1000.0
                    offset_ms = (t2 - (t1 + t3) / 2.0) * 1000.0
                    samples.append((rtt_ms, offset_ms))
                    time.sleep(0.05)
                except socket.timeout:
                    break
        finally:
            sock.close()

        if samples:
            best_rtt, best_offset = min(samples, key=lambda s: s[0])
            avg_rtt = sum(s[0] for s in samples) / len(samples)
            print(
                f"[ClockSync] {len(samples)} pings — "
                f"offset: {best_offset:+.1f} ms | "
                f"best RTT: {best_rtt:.1f} ms | avg RTT: {avg_rtt:.1f} ms"
            )
            return best_offset

        print(f"[ClockSync] No response (attempt {attempt + 1}/{max_retries}) — "
              f"retrying in 1s...")
        time.sleep(1.0)

    print("[ClockSync] All retries failed — offset=0. Check laptop vision_streamer is running.")
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    with open(args.env_config) as f:
        config = json.load(f)

    from dxlSetup.portAndPackets import openPortObject, portHandler, packetHandler
    from dxlSetup.XL430 import XL430
    import board, busio, adafruit_bno055

    openPortObject()
    servo = XL430(id=1, zeroPos=0)
    if servo.checkHardwareError():
        print("[Motor] Hardware error detected — rebooting servo to clear...")
        servo.reboot()
        time.sleep(1.5)
        servo.setOpMode(4)
        if servo.checkHardwareError():
            print("[Motor] WARNING: hardware error persists after reboot.")
    servo.setOpMode(4)

    i2c = busio.I2C(board.SCL, board.SDA)
    imu_sensor = adafruit_bno055.BNO055_I2C(i2c)
    imu_sensor.mode = adafruit_bno055.IMUPLUS_MODE

    motor_runner = MotorRunner(servo, packetHandler, portHandler)
    imu_reader = ImuReader(imu_sensor)

    policy = PolicyRunner(args.model)
    cpg = CPGOscillator(dt_phys=DT, tau=CPG_TAU, motor_limit=MOTOR_LIMIT)
    _assembler_map = {
        "cpg_14d": ObsAssemblerCPG14D,
        "cpg_12d": ObsAssemblerCPG12D,
        "cpg_24d": ObsAssemblerCPG24D,
        "da_rl_policy_40d": ObsAssemblerDA40D,
    }
    env_type = config.get("env_type", "cpg_14d")
    assembler_cls = _assembler_map.get(env_type)
    if assembler_cls is None:
        raise ValueError(f"Unknown env_type '{env_type}'. Valid: {list(_assembler_map)}")
    assembler = assembler_cls(config)
    # Direct-action (DA) vs CPG: selects the control-loop path below.
    _is_da = config.get("action_type") == "direct"
    _motor_limit_rad = float(config.get("motor_limit_rad", 1.0))
    _success_radius = float(config.get("success_radius", 0.1))

    # _stop     : hard stop — Ctrl+C / SIGTERM. Exits the entire script.
    # _stop_run : soft stop — CTRL_STOP from laptop (R pressed). Ends current
    #             run only; Pi loops back to READY for the next run.
    _stop = threading.Event()
    _stop_run = threading.Event()

    def _shutdown(sig, frame):
        print(f"\n[Deploy] Signal {sig} — stopping.")
        _stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    motor_runner.start()
    imu_reader.start()

    # IMU calibration — once at startup
    print("[IMU] Waiting for gyro calibration (keep sensor still)...", flush=True)
    cal_start = time.time()
    while not _stop.is_set():
        try:
            cal = imu_sensor.calibration_status
            gyro_cal = cal[1] if cal else 0
            print(f"[IMU] Calibration: {cal}\r", end="", flush=True)
            if gyro_cal >= 3:
                print(f"\n[IMU] Gyro calibrated in {time.time() - cal_start:.1f}s.")
                break
        except Exception:
            pass
        if time.time() - cal_start > 30.0:
            print("\n[IMU] WARNING: gyro calibration timeout — continuing.")
            break
        time.sleep(0.2)

    # Vision socket — persistent across runs
    pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    pose_sock.settimeout(0.0)
    pose_sock.bind(("", POSE_PORT))

    # CTRL socket — persistent across runs
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_sock.settimeout(1.0)
    ctrl_sock.bind(("", CTRL_PORT))

    latest_vision: VisionState = None
    laptop_ip: str = None
    frame_seq_prev = -1

    # Wait for first valid pose packet — captures laptop IP
    print(f"[Deploy] Waiting for first valid UDP pose packet (timeout {STARTUP_TIMEOUT_S}s)...")
    startup_start = time.time()
    while latest_vision is None and not _stop.is_set():
        try:
            data, pkt_addr = pose_sock.recvfrom(PACKET_SIZE)
            ts, seq, fx, fy, fyaw, tx, ty, res, flags = unpack_pose(data)
            if laptop_ip is None:
                laptop_ip = pkt_addr[0]
            if (flags & FLAGS_FISH) and (flags & FLAGS_TARGET):
                latest_vision = VisionState(
                    fish_x=fx, fish_y=fy, fish_yaw=fyaw,
                    target_x=tx, target_y=ty,
                    flags=flags, pi_arrival_time=time.time(), frame_seq=seq,
                    laptop_timestamp=ts,
                )
        except BlockingIOError:
            pass
        if time.time() - startup_start > STARTUP_TIMEOUT_S:
            print("[Deploy] TIMEOUT waiting for vision. Aborting.")
            motor_runner.stop(); imu_reader.stop(); pose_sock.close(); ctrl_sock.close()
            sys.exit(1)
        time.sleep(0.01)

    # Multi-run loop
    data_root = os.path.join(_REPO_ROOT, "deploy", "data")
    model_hash = _model_hash(args.model)
    git_hash = _git_hash()
    pi_ip_str = socket.gethostbyname(socket.gethostname())

    while not _stop.is_set():
        # Clock sync each round
        print(f"[Deploy] Clock sync with {laptop_ip}...")
        clock_offset_ms = _clock_sync_handshake(laptop_ip, SYNC_PORT) if laptop_ip else 0.0
        assembler.set_clock_offset(clock_offset_ms / 1000.0)

        # Send READY
        ctrl_sock.settimeout(1.0)
        ctrl_sock.sendto(make_ready_packet(args.name), (laptop_ip, CTRL_PORT))
        print(f"[Deploy] Sent READY:{args.name} — waiting for START (120s timeout)...")

        # Wait for START
        start_ts = None
        deadline = time.time() + 120.0
        while time.time() < deadline and not _stop.is_set():
            try:
                pkt, _ = ctrl_sock.recvfrom(64)
                if pkt.startswith(b"START:"):
                    start_ts = parse_start_packet(pkt)
                    break
            except socket.timeout:
                pass

        if start_ts is None or _stop.is_set():
            print("[Deploy] START timeout or stop — exiting multi-run loop.")
            break

        # Drain any stale ctrl packets (old STOPs, duplicate STARTs)
        _orig_to = ctrl_sock.gettimeout()
        ctrl_sock.settimeout(0.0)
        try:
            while True:
                ctrl_sock.recvfrom(128)
        except (BlockingIOError, OSError):
            pass
        ctrl_sock.settimeout(_orig_to)

        # Return motor to zero and re-zero reference before each run
        motor_runner.set_goal(0.0)
        time.sleep(0.35)
        motor_runner.rezero()

        # Drain stale pose packets and wait for a fresh one before starting
        print("[Deploy] Warmup — waiting for fresh vision...", flush=True)
        pose_sock.settimeout(0.0)
        _warmup_start = time.time()
        _warmup_got_fresh = False
        while time.time() - _warmup_start < 3.0:
            try:
                data, _ = pose_sock.recvfrom(256)
                if len(data) >= PACKET_SIZE:
                    ts_pkt, seq, fx, fy, fyaw, tx, ty, res, flags = unpack_pose(data)
                    if (flags & FLAGS_FISH) and (flags & FLAGS_TARGET):
                        latest_vision = VisionState(
                            fish_x=fx, fish_y=fy, fish_yaw=fyaw,
                            target_x=tx, target_y=ty, flags=flags,
                            pi_arrival_time=time.time(), frame_seq=seq,
                            laptop_timestamp=ts_pkt,
                        )
                        _warmup_got_fresh = True
            except (BlockingIOError, OSError):
                pass
            time.sleep(0.01)
        if _warmup_got_fresh:
            print("[Deploy] Vision confirmed fresh — starting run.", flush=True)
        else:
            print("[Deploy] WARNING: no fresh vision during warmup — starting anyway.", flush=True)

        run_dir = make_run_dir(data_root, args.name, ts=start_ts)
        os.makedirs(run_dir, exist_ok=True)
        print(f"[Deploy] Run dir: {run_dir}")

        prev_omega_raw       = 0.0
        prev_theta_center_raw = 0.0
        _stop_run.clear()
        cpg = CPGOscillator(dt_phys=DT, tau=CPG_TAU, motor_limit=MOTOR_LIMIT)
        logger = DataLogger(
            run_dir=run_dir,
            config=config,
            record=args.record,
            finetune=args.finetune,
            model_hash=model_hash,
            git_hash=git_hash,
            pi_ip=pi_ip_str,
            clock_offset_ms=clock_offset_ms,
        )

        frame_seq_prev = -1
        ctrl_sock.settimeout(0.0)  # non-blocking STOP checks during run
        run_start = time.time()
        _vision_lost_since: float = None

        try:
            while not _stop.is_set() and not _stop_run.is_set() and (time.time() - run_start) < args.duration:
                t0 = time.time()

                motor_state = motor_runner.get_state()
                imu_state = imu_reader.get_imu()

                dropped = 0
                try:
                    while True:
                        data, _ = pose_sock.recvfrom(PACKET_SIZE)
                        ts_pkt, seq, fx, fy, fyaw, tx, ty, res, flags = unpack_pose(data)
                        if seq != frame_seq_prev + 1 and frame_seq_prev >= 0:
                            dropped += seq - frame_seq_prev - 1
                        frame_seq_prev = seq
                        if (flags & FLAGS_FISH) and (flags & FLAGS_TARGET):
                            latest_vision = VisionState(
                                fish_x=fx, fish_y=fy, fish_yaw=fyaw,
                                target_x=tx, target_y=ty, flags=flags,
                                pi_arrival_time=time.time(), frame_seq=seq,
                                laptop_timestamp=ts_pkt,
                            )
                except BlockingIOError:
                    pass

                if dropped >= 4:
                    print(f"[WARN] Camera: {dropped} consecutive dropped frames")

                vstatus = assembler.vision_status(latest_vision)
                if vstatus == "lost":
                    if _vision_lost_since is None:
                        _vision_lost_since = time.time()
                        print("[WARN] Vision lost — will abort in 5s if not recovered.")
                    elif time.time() - _vision_lost_since >= VISION_LOSS_ABORT_S:
                        print("[SAFETY] Vision LOST >5s. Emergency stop.")
                        _stop.set()
                        break
                else:
                    _vision_lost_since = None

                if _is_da:
                    # Direct action: 1 motor-position setpoint in [-1,1] × ctrlrange.
                    obs = assembler.build(motor_state, imu_state, latest_vision)
                    action_raw = policy.predict_raw(obs)
                    setpoint_norm = float(np.clip(action_raw[0], -1.0, 1.0))
                    motor_setpoint = setpoint_norm * _motor_limit_rad
                    motor_runner.set_goal(motor_setpoint)
                    assembler.set_pending_action(setpoint_norm)   # setpoint↔next actual pairing
                    _dist = assembler.last_dist
                    reward = -_dist + (1000.0 if _dist < _success_radius else 0.0)
                    done = bool(_dist < _success_radius)
                    _cpg_phase_log = 0.0
                else:
                    obs = assembler.build(motor_state, imu_state, latest_vision, cpg.phase,
                                          prev_omega_raw, prev_theta_center_raw)
                    action_raw, action_scaled = policy.predict(obs)
                    prev_omega_raw        = float(action_raw[4])
                    prev_theta_center_raw = float(action_raw[5])
                    motor_setpoint = cpg.step(action_scaled)
                    motor_runner.set_goal(motor_setpoint)
                    reward = compute_reward(obs, action_raw)
                    done = bool(obs[6] < 0.05)
                    _cpg_phase_log = cpg.phase
                vision_age_ms = (time.time() - latest_vision.laptop_timestamp + clock_offset_ms / 1000.0) * 1000.0
                logger.log_step(
                    pi_time=t0,
                    laptop_time=latest_vision.laptop_timestamp,
                    obs=obs, action=action_raw, motor_setpoint=motor_setpoint,
                    reward=reward, done=done, vision_age_ms=vision_age_ms,
                    vision_flags=latest_vision.flags, frame_seq=latest_vision.frame_seq,
                    cpg_phase=_cpg_phase_log,
                    motor_load=motor_state.get("load_raw", 0),
                )

                # Non-blocking STOP check — R pressed on laptop ends this run only
                try:
                    ctrl_data, _ = ctrl_sock.recvfrom(64)
                    if ctrl_data == CTRL_STOP:
                        print("[Deploy] STOP received from laptop — ending run.")
                        _stop_run.set()
                except BlockingIOError:
                    pass

                elapsed = time.time() - t0
                time.sleep(max(0.0, DT - elapsed))

        finally:
            ctrl_sock.settimeout(1.0)
            _done_pkt = make_done_packet(time.time())
            for _ in range(3):
                ctrl_sock.sendto(_done_pkt, (laptop_ip, CTRL_PORT))
                time.sleep(0.05)
            logger.close()
            print(f"[Deploy] Run done: {run_dir}")

        if _stop.is_set():
            break
        # else: loop back to clock sync + READY for next run

    motor_runner.stop()
    imu_reader.stop()
    pose_sock.close()
    ctrl_sock.close()
    print("[Deploy] Exited cleanly.")


if __name__ == "__main__":
    main()
