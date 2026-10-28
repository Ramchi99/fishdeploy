"""
Open-loop and shadow-mode deployment for SOFI.

Modes
-----
open loop   — predefined trajectory commands the motor; full obs + sensor data logged.
shadow      — same as open loop, but the SAC policy also runs in parallel. Its actions
              are logged (policy_act_raw, policy_act_scaled, policy_motor_setpoint) so
              you can post-hoc compare what RL would have commanded vs the open-loop
              trajectory. The fish only moves according to the trajectory.

Usage (on Pi, from fishdeploy/ repo root)
-------------------------------------------

  # Noise floor — motor silent, record sensor noise
  python3 deploy/pi/open_loop_main.py \\
      --name noise_floor_1 \\
      --env-config deploy/configs/cpg_24d.json \\
      --trajectory zero --duration 15 --record

  # Sine glide without target tag in frame (fish tracking only)
  python3 deploy/pi/open_loop_main.py \\
      --name glide_notarget_1 \\
      --env-config deploy/configs/cpg_24d.json \\
      --trajectory sine_glide --freq 1.5 --amp 0.3 --sine_duration 10 \\
      --duration 20 --record --no-target

  # Continuous sine (no duration limit — runs until R is pressed again)
  python3 deploy/pi/open_loop_main.py \\
      --name sine_1 \\
      --env-config deploy/configs/cpg_24d.json \\
      --trajectory sine --freq 1.5 --amp 0.3 \\
      --record

  # Sine then glide (motor cuts to zero after --sine_duration)
  python3 deploy/pi/open_loop_main.py \\
      --name glide_1 \\
      --env-config deploy/configs/cpg_24d.json \\
      --trajectory sine_glide --freq 1.5 --amp 0.3 --sine_duration 10 \\
      --duration 20 --record

  # Shadow mode — open-loop trajectory, policy runs alongside (policy does NOT move motor)
  python3 deploy/pi/open_loop_main.py \\
      --name shadow_1 \\
      --env-config deploy/configs/cpg_24d.json \\
      --trajectory sine --freq 1.5 --amp 0.3 \\
      --shadow --model policies/sofi_cpg_rl_policy.zip \\
      --duration 30 --record

Start/stop
----------
  Press R in the laptop vision_streamer to START a run.
  Press R again (or wait for --duration) to STOP the current run.
  The Pi immediately loops back to READY for another run — no restart needed.
  Press Ctrl+C on the Pi to exit entirely.
"""
import argparse
import json
import os
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

from deploy.pi.imu_reader import ImuReader
from deploy.pi.motor_runner import MotorRunner, MOTOR_LIMIT_RAD
from deploy.pi.obs_assembler import ObsAssemblerCPG12D, ObsAssemblerCPG14D, ObsAssemblerCPG24D, VisionState
from deploy.pi.open_loop_logger import OpenLoopLogger
from deploy.pi.trajectory import make_trajectory
from deploy.shared.protocol import (
    PACKET_SIZE, POSE_PORT, SYNC_PORT, CTRL_PORT,
    FLAGS_FISH, FLAGS_TARGET,
    CTRL_STOP,
    make_ready_packet, make_start_packet, parse_start_packet, make_done_packet,
    unpack_pose,
)
from deploy.pi.deploy_main import make_run_dir, _clock_sync_handshake, _model_hash, _git_hash

CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ
STARTUP_TIMEOUT_S = 60.0


def _drain_ctrl_sock(sock) -> None:
    """Discard any stale packets in ctrl_sock before starting a run.

    Packets that arrived during the READY/START wait (e.g., duplicate STARTs,
    stale STOP from a previous run) would otherwise be processed immediately
    at the start of the new run and could prematurely terminate it.
    """
    orig_timeout = sock.gettimeout()
    sock.settimeout(0.0)
    try:
        while True:
            sock.recvfrom(128)
    except (BlockingIOError, OSError):
        pass
    finally:
        sock.settimeout(orig_timeout)


def main():
    parser = argparse.ArgumentParser(description="Open-loop / shadow deployment for SOFI")
    parser.add_argument("--name", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--trajectory", required=True, choices=["zero", "sine", "sine_glide", "sine_hold"])
    parser.add_argument("--freq", type=float, default=1.5)
    parser.add_argument("--amp", type=float, default=0.3)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--sine_duration", type=float, default=10.0)
    parser.add_argument("--hold_offset", type=float, default=0.0,
                        help="Tail position [rad] to hold after sine phase (sine_hold only)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Run duration in seconds. Omit to run until R is pressed again.")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--no-target", action="store_true", dest="no_target",
                        help="Accept fish-only UDP packets (no target tag required in frame)")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    if args.shadow and not args.model:
        parser.error("--shadow requires --model")

    with open(args.env_config) as f:
        config = json.load(f)

    traj_params = {"freq": args.freq, "amp": args.amp, "offset": args.offset}
    if args.trajectory in ("sine_glide", "sine_hold"):
        traj_params["sine_duration"] = args.sine_duration
    if args.trajectory == "sine_hold":
        traj_params["hold_offset"] = args.hold_offset
    trajectory = make_trajectory(args.trajectory, **traj_params)

    # Hardware init — once
    from dxlSetup.portAndPackets import openPortObject, portHandler, packetHandler
    from dxlSetup.XL430 import XL430
    import board, busio, adafruit_bno055

    openPortObject()
    servo = XL430(id=1, zeroPos=0)
    servo.setOpMode(4)

    i2c = busio.I2C(board.SCL, board.SDA)
    imu_sensor = adafruit_bno055.BNO055_I2C(i2c)
    imu_sensor.mode = adafruit_bno055.IMUPLUS_MODE

    motor_runner = MotorRunner(servo, packetHandler, portHandler)
    imu_reader = ImuReader(imu_sensor)

    shadow_policy = None
    model_hash_str = ""
    if args.shadow:
        from deploy.pi.policy_runner import PolicyRunner
        shadow_policy = PolicyRunner(args.model)
        model_hash_str = _model_hash(args.model)

    # _stop  : hard stop — Ctrl+C / SIGTERM. Exits the entire script.
    # _stop_run: soft stop — CTRL_STOP from laptop (R pressed). Ends the
    #            current run only; Pi loops back to READY for the next run.
    _stop = threading.Event()
    _stop_run = threading.Event()

    def _shutdown(sig, frame):
        print(f"\n[OpenLoop] Signal {sig} — stopping.")
        _stop.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    motor_runner.start()
    imu_reader.start()

    # IMU cal — once
    print("[IMU] Waiting for gyro calibration...", flush=True)
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

    pose_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pose_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    pose_sock.settimeout(0.0)
    pose_sock.bind(("", POSE_PORT))

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ctrl_sock.settimeout(1.0)
    ctrl_sock.bind(("", CTRL_PORT))

    latest_vision: VisionState = None
    laptop_ip: str = None
    frame_seq_prev = -1

    _assembler_map = {
        "cpg_14d": ObsAssemblerCPG14D,
        "cpg_12d": ObsAssemblerCPG12D,
        "cpg_24d": ObsAssemblerCPG24D,
    }
    env_type = config.get("env_type", "cpg_14d")
    assembler_cls = _assembler_map.get(env_type)
    if assembler_cls is None:
        raise ValueError(f"Unknown env_type '{env_type}'. Valid: {list(_assembler_map)}")
    assembler = assembler_cls(config)

    # Wait for first pose packet — once
    print(f"[OpenLoop] Waiting for first valid UDP pose packet (timeout {STARTUP_TIMEOUT_S}s)...")
    startup_start = time.time()
    while latest_vision is None and not _stop.is_set():
        try:
            data, pkt_addr = pose_sock.recvfrom(PACKET_SIZE)
            ts_pkt, seq, fx, fy, fyaw, tx, ty, res, flags = unpack_pose(data)
            if laptop_ip is None:
                laptop_ip = pkt_addr[0]
            fish_ok = bool(flags & FLAGS_FISH)
            target_ok = bool(flags & FLAGS_TARGET) or args.no_target
            if fish_ok and target_ok:
                latest_vision = VisionState(
                    fish_x=fx, fish_y=fy, fish_yaw=fyaw,
                    target_x=tx, target_y=ty,
                    flags=flags, pi_arrival_time=time.time(), frame_seq=seq,
                    laptop_timestamp=ts_pkt,
                )
        except BlockingIOError:
            pass
        if time.time() - startup_start > STARTUP_TIMEOUT_S:
            print("[OpenLoop] TIMEOUT waiting for vision. Aborting.")
            motor_runner.stop(); imu_reader.stop()
            pose_sock.close(); ctrl_sock.close()
            sys.exit(1)
        time.sleep(0.01)

    data_root = os.path.join(_REPO_ROOT, "deploy", "data")
    git_hash = _git_hash()
    pi_ip_str = socket.gethostbyname(socket.gethostname())

    duration_str = f"{args.duration}s" if args.duration is not None else "unlimited (R to stop)"
    print(f"[OpenLoop] Ready. Trajectory={args.trajectory}  duration={duration_str}")

    # Multi-run loop
    while not _stop.is_set():
        print(f"[OpenLoop] Clock sync with {laptop_ip}...")
        clock_offset_ms = _clock_sync_handshake(laptop_ip, SYNC_PORT) if laptop_ip else 0.0

        ctrl_sock.settimeout(1.0)
        ctrl_sock.sendto(make_ready_packet(args.name), (laptop_ip, CTRL_PORT))
        print(f"[OpenLoop] Sent READY:{args.name} — waiting for START (R on laptop, 120s timeout)...")

        start_ts = None
        deadline = time.time() + 120.0
        while time.time() < deadline and not _stop.is_set():
            try:
                pkt, _ = ctrl_sock.recvfrom(64)
                if pkt.startswith(b"START:"):
                    start_ts = parse_start_packet(pkt)
                    break
                # CTRL_STOP during wait: just ignore (stale or misfire)
            except socket.timeout:
                pass

        if start_ts is None or _stop.is_set():
            print("[OpenLoop] START timeout or stop — exiting.")
            break

        # Drain any stale control packets (duplicate STARTs, old STOPs)
        _drain_ctrl_sock(ctrl_sock)

        # Return motor to zero before each run so trajectory always starts centred.
        # Motor has torque enabled — set_goal(0.0) commands the neutral position,
        # then rezero() snaps the tick reference to the actual encoder reading.
        motor_runner.set_goal(0.0)
        time.sleep(0.35)
        motor_runner.rezero()

        run_dir = make_run_dir(data_root, args.name, ts=start_ts)
        os.makedirs(run_dir, exist_ok=True)

        prev_shadow_omega_raw = 0.0
        shadow_cpg = None
        if args.shadow:
            from deploy.pi.cpg_oscillator import CPGOscillator
            shadow_cpg = CPGOscillator(dt_phys=DT, tau=0.1, motor_limit=MOTOR_LIMIT_RAD)

        logger = OpenLoopLogger(
            run_dir=run_dir,
            config=config,
            trajectory_name=args.trajectory,
            trajectory_params=traj_params,
            shadow=args.shadow,
            record=args.record,
            git_hash=git_hash,
            pi_ip=pi_ip_str,
            clock_offset_ms=clock_offset_ms,
            model_hash=model_hash_str,
        )

        mode_label = "shadow" if args.shadow else "open_loop"
        print(f"[OpenLoop] Starting {mode_label} ({CONTROL_HZ} Hz, {duration_str}).")
        frame_seq_prev = -1
        _stop_run.clear()
        ctrl_sock.settimeout(0.0)
        run_start = time.time()
        _vision_lost_since: float = None

        try:
            while not _stop.is_set() and not _stop_run.is_set():
                t0 = time.time()
                t_run = t0 - run_start

                if args.duration is not None and t_run >= args.duration:
                    break

                motor_state = motor_runner.get_state()
                imu_state = imu_reader.get_imu()

                dropped = 0
                latest_ts_pkt = 0.0
                try:
                    while True:
                        data, _ = pose_sock.recvfrom(PACKET_SIZE)
                        ts_pkt, seq, fx, fy, fyaw, tx, ty, res, flags = unpack_pose(data)
                        latest_ts_pkt = ts_pkt
                        if seq != frame_seq_prev + 1 and frame_seq_prev >= 0:
                            dropped += seq - frame_seq_prev - 1
                        frame_seq_prev = seq
                        if (flags & FLAGS_FISH) and ((flags & FLAGS_TARGET) or args.no_target):
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

                # Vision loss: in open-loop the trajectory does not depend on vision,
                # so we never abort — just print a warning so the operator knows.
                vstatus = assembler.vision_status(latest_vision)
                if vstatus == "lost":
                    if _vision_lost_since is None:
                        _vision_lost_since = time.time()
                        print("[WARN] Vision lost (fish tag not detected) — continuing trajectory.")
                    elif time.time() - _vision_lost_since >= 5.0:
                        _vision_lost_since = time.time()  # re-arm to avoid spam
                        print("[WARN] Vision still lost — data will have stale vision obs.")
                else:
                    if _vision_lost_since is not None:
                        print(f"[INFO] Vision recovered after {time.time()-_vision_lost_since:.1f}s.")
                    _vision_lost_since = None

                motor_setpoint = float(np.clip(trajectory.step(t_run), -MOTOR_LIMIT_RAD, MOTOR_LIMIT_RAD))

                if args.shadow:
                    obs = assembler.build(motor_state, imu_state, latest_vision, shadow_cpg.phase, prev_shadow_omega_raw)
                    policy_act_raw, policy_act_scaled = shadow_policy.predict(obs)
                    prev_shadow_omega_raw = float(policy_act_raw[4])
                    policy_motor_sp = shadow_cpg.step(policy_act_scaled)
                else:
                    traj_phase = trajectory.cpg_phase(t_run)
                    obs = assembler.build(motor_state, imu_state, latest_vision, traj_phase, 0.0)
                    policy_act_raw = policy_act_scaled = policy_motor_sp = None

                motor_runner.set_goal(motor_setpoint)

                vision_age_ms = (time.time() - latest_vision.pi_arrival_time) * 1000.0
                logger.log_step(
                    pi_time=t0,
                    laptop_time=latest_vision.laptop_timestamp,  # from latest valid vision frame
                    obs=obs,
                    trajectory_cmd=motor_setpoint,
                    vision_age_ms=vision_age_ms,
                    vision_flags=latest_vision.flags,
                    frame_seq=latest_vision.frame_seq,
                    policy_act_raw=policy_act_raw,
                    policy_act_scaled=policy_act_scaled,
                    policy_motor_setpoint=policy_motor_sp,
                )

                # Non-blocking STOP check — R pressed on laptop
                try:
                    ctrl_data, _ = ctrl_sock.recvfrom(64)
                    if ctrl_data == CTRL_STOP:
                        print("[OpenLoop] STOP received — ending run.")
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
            stop_reason = "STOP from laptop" if _stop_run.is_set() else (
                f"duration {args.duration}s elapsed" if args.duration and t_run >= args.duration else "signal"
            )
            print(f"[OpenLoop] Run done ({stop_reason}): {run_dir}")

        if _stop.is_set():
            break
        # else: loop back for clock sync + READY for next run

    motor_runner.stop()
    imu_reader.stop()
    pose_sock.close()
    ctrl_sock.close()
    print("[OpenLoop] Exited cleanly.")


if __name__ == "__main__":
    main()
