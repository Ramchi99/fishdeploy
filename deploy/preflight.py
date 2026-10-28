#!/usr/bin/env python3
"""
Pre-flight check for SOFI RL deployment.

Run from repo root before each session:
    python deploy/preflight.py

Logs results to deploy/preflight_logs/YYYYMMDD_HHMMSS_preflight.md
"""
import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
ICON = {PASS: "✅", WARN: "⚠️", FAIL: "❌"}


class Result:
    def __init__(self, name, status, lines):
        self.name = name
        self.status = status
        self.lines = lines


# ── T1: network ping ──────────────────────────────────────────────────────────

def t1_ping(pi_ip):
    try:
        out = subprocess.run(
            ["ping", "-c", "3", "-W", "2", pi_ip],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return Result("Network Ping", FAIL, [f"No response from {pi_ip}"])
        for line in out.stdout.splitlines():
            if "avg" in line or "rtt" in line.lower():
                return Result("Network Ping", PASS, [line.strip()])
        return Result("Network Ping", PASS, ["Ping OK"])
    except Exception as e:
        return Result("Network Ping", FAIL, [str(e)])


# ── T2: UDP clock sync ────────────────────────────────────────────────────────

def t2_clock_sync(pi_ip, sync_port=5006, n_pings=20):
    # Pi listener: receives n_pings and echoes back Pi timestamp for each
    pi_script = (
        f"import socket,struct,time\n"
        f"s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
        f"s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
        f"s.bind(('',{sync_port}))\n"
        f"s.settimeout(5.0)\n"
        f"for _ in range({n_pings}):\n"
        f"  try:\n"
        f"    d,a=s.recvfrom(16)\n"
        f"    s.sendto(struct.pack('!d',time.time()),a)\n"
        f"  except socket.timeout: break\n"
        f"s.close()\n"
    )
    listener_cmd = f"python3 -c \"{pi_script}\""
    try:
        proc = subprocess.Popen(
            ["ssh", f"sofi@{pi_ip}", listener_cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.8)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(5.0)
        samples = []
        try:
            for _ in range(n_pings):
                try:
                    t1 = time.time()
                    sock.sendto(struct.pack("!d", t1), (pi_ip, sync_port))
                    data, _ = sock.recvfrom(16)
                    t4 = time.time()
                    t2_pi = struct.unpack("!d", data)[0]
                    samples.append(((t4 - t1) * 1000, (t2_pi - (t1 + t4) / 2) * 1000))
                    time.sleep(0.02)
                except socket.timeout:
                    break
        finally:
            sock.close()
        proc.wait(timeout=5)

        if not samples:
            return Result("Clock Sync (UDP)", FAIL, ["No response from Pi"])

        best_rtt, best_offset = min(samples, key=lambda s: s[0])
        avg_rtt = sum(s[0] for s in samples) / len(samples)
        lines = [
            f"Pi offset: {best_offset:+.1f} ms ({len(samples)} pings)",
            f"Best RTT: {best_rtt:.1f} ms | Avg RTT: {avg_rtt:.1f} ms",
        ]
        if abs(best_offset) > 200:
            lines.append("Large offset — run on Pi: sudo timedatectl set-ntp true")
            return Result("Clock Sync (UDP)", WARN, lines)
        if abs(best_offset) > 50:
            lines.append("Offset >50 ms — consider running: sudo chronyc makestep")
            return Result("Clock Sync (UDP)", WARN, lines)
        return Result("Clock Sync (UDP)", PASS, lines)
    except Exception as e:
        return Result("Clock Sync (UDP)", FAIL, [str(e)])


# ── T3: config file ───────────────────────────────────────────────────────────

def t3_config(config_path):
    lines = []
    status = PASS
    try:
        with open(config_path) as f:
            cfg = json.load(f)

        offset = cfg.get("tag0_to_com_offset", [0, 0])
        if offset == [0.0, 0.0] or offset == [0, 0]:
            lines.append("tag0_to_com_offset = [0, 0] — not measured yet")
            status = WARN
        else:
            lines.append(f"tag0_to_com_offset: {offset}")

        policy = cfg.get("model_path", "")
        if not policy:
            lines.append("model_path not set in config")
            status = WARN
        elif (_REPO / policy).exists():
            lines.append(f"Policy file: {policy}")
        else:
            lines.append(f"Policy file not found: {policy}")
            status = WARN

        return Result("Config", status, lines)
    except FileNotFoundError:
        return Result("Config", FAIL, [f"Config not found: {config_path}"])
    except Exception as e:
        return Result("Config", FAIL, [str(e)])


# ── T4: camera ────────────────────────────────────────────────────────────────

_CAMERA_FIX = (
    "Fix: in PowerShell (Admin) run: .\\scripts\\attach_camera_wsl.ps1"
    "  then verify with: ls /dev/video* in WSL"
)

def t4_camera(device=0):
    try:
        import cv2
        import os
        os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            return Result("Camera", FAIL, [
                f"Could not open /dev/video{device}",
                _CAMERA_FIX,
            ])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return Result("Camera", PASS, [f"Device {device} opened ({w}x{h})"])
    except Exception as e:
        return Result("Camera", FAIL, [str(e), _CAMERA_FIX])


# ── T5: SB3 on Pi ─────────────────────────────────────────────────────────────

def t5_sb3(pi_ip):
    try:
        out = subprocess.run(
            ["ssh", f"sofi@{pi_ip}", "~/fishdeploy/venv/bin/python -c \"from stable_baselines3 import SAC; print('ok')\""],
            capture_output=True, text=True, timeout=30,
        )
        if "ok" in out.stdout:
            return Result("SB3 on Pi", PASS, ["stable_baselines3 SAC import OK"])
        err = (out.stderr or out.stdout).strip().splitlines()
        return Result("SB3 on Pi", FAIL, err[:3] or ["Import failed"])
    except Exception as e:
        return Result("SB3 on Pi", FAIL, [str(e)])


# ── T6: motor port on Pi ──────────────────────────────────────────────────────

def t6_motor_port(pi_ip):
    try:
        out = subprocess.run(
            ["ssh", f"sofi@{pi_ip}", "ls /dev/ttyUSB0"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return Result("Motor Port", PASS, ["/dev/ttyUSB0 present"])
        return Result("Motor Port", FAIL, ["/dev/ttyUSB0 not found — USB disconnected?"])
    except Exception as e:
        return Result("Motor Port", FAIL, [str(e)])


# ── report writer ─────────────────────────────────────────────────────────────

def write_report(results, pi_ip, config_path):
    now = datetime.now()
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=_REPO
        ).strip()
    except Exception:
        git_hash = "unknown"

    lines = [
        "# SOFI Pre-flight Report",
        f"**Date:** {now.strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Git:** {git_hash}  ",
        f"**Pi:** sofi@{pi_ip}  ",
        f"**Config:** {config_path}  ",
        "",
        "---",
        "",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"## T{i} — {r.name}")
        lines.append(f"**Status:** {ICON[r.status]} {r.status}  ")
        for l in r.lines:
            lines.append(f"- {l}")
        lines.append("")

    n = {s: [r.status for r in results].count(s) for s in (PASS, WARN, FAIL)}
    lines += [
        "---",
        f"**Summary:** {n[PASS]} passed · {n[WARN]} warnings · {n[FAIL]} failed",
    ]

    log_dir = _REPO / "deploy" / "preflight_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    out_path = log_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_preflight.md"
    out_path.write_text("\n".join(lines))
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SOFI RL pre-flight check")
    parser.add_argument("--pi_ip", default="raspberrypi.local")
    parser.add_argument("--config", default="deploy/configs/cpg_24d.json")
    parser.add_argument("--camera_device", type=int, default=0)
    parser.add_argument("--skip_camera", action="store_true", help="Skip camera open test")
    args = parser.parse_args()

    config_path = _REPO / args.config

    print("=== SOFI Pre-flight ===\n")
    results = []

    tests = [
        ("T1  Network Ping  ", lambda: t1_ping(args.pi_ip)),
        ("T2  Clock Sync    ", lambda: t2_clock_sync(args.pi_ip)),
        ("T3  Config        ", lambda: t3_config(config_path)),
        ("T4  Camera        ", lambda: t4_camera(args.camera_device)
                               if not args.skip_camera
                               else Result("Camera", WARN, ["Skipped (--skip_camera)"])),
        ("T5  SB3 on Pi     ", lambda: t5_sb3(args.pi_ip)),
        ("T6  Motor Port    ", lambda: t6_motor_port(args.pi_ip)),
    ]

    for label, fn in tests:
        print(f"  {label}...", end=" ", flush=True)
        r = fn()
        results.append(r)
        print(ICON[r.status], r.status)
        for l in r.lines:
            print(f"       {l}")

    out_path = write_report(results, args.pi_ip, args.config)

    n_fail = sum(1 for r in results if r.status == FAIL)
    n_warn = sum(1 for r in results if r.status == WARN)
    print(f"\nReport → {out_path.relative_to(_REPO)}")
    if n_fail:
        print(f"  {n_fail} test(s) FAILED — fix before running policy")
    elif n_warn:
        print(f"  {n_warn} warning(s) — review report before proceeding")
    else:
        print("  All clear.")


if __name__ == "__main__":
    main()
