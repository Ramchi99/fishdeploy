#!/usr/bin/env python3
"""
SOFI RL Observation Verifier
=============================
Walks through 10 physical tests and verifies that every observation
dimension matches the MuJoCo training convention.  Run on the laptop;
optionally connect to the Pi sensor server for live IMU + motor data.

Usage
-----
    # Vision-only tests (no Pi needed):
    python deploy/obs_verify/obs_verifier.py --device 1

    # Full tests (Pi sensor server must be running on the Pi first):
    python deploy/obs_verify/obs_verifier.py --device 1 --pi-ip raspberrypi.local

Controls (during each test)
---------------------------
    ENTER  →  mark PASS and advance to next test
    f      →  mark FAIL and advance
    s      →  SKIP this test
    r      →  reset frame buffer (fresh ObsAssembler instance)
    q      →  quit

Test sequence (indices of the deployed 24-D observation, ObsAssemblerCPG24D)
-------------
 1. Vision Alive       — all 3 tags visible, fps > 20
 2. Distance           — obs[14] (dist of the most recent frame) matches the tape measure
 3. Fish → Target      — obs[15] (vec_x) < 0,  obs[16] (vec_y) ≈ 0
 4. Target to LEFT     — obs[16] (vec_y) < 0
 5. Target to RIGHT    — obs[16] (vec_y) > 0
 6. Gyro CCW (yaw)     — obs[3] (wz) > 0   [needs Pi sensor]
 7. Gyro CW  (yaw)     — obs[3] (wz) < 0   [needs Pi sensor]
 8. Roll / pitch       — not in the observation; read wx, wy on the Pi sensor server
 9. Gyro at rest       — obs[3] (wz) ≈ 0   [needs Pi sensor]
10. Motor centre       — obs[0] ≈ 1, obs[1] ≈ 0  [needs Pi sensor]

MuJoCo body-frame convention
----------------------------
  obs[15] vec_x NEGATIVE when target is directly ahead
                (fish forward = body −X in MuJoCo)
  obs[16] vec_y NEGATIVE when target is to the LEFT
                POSITIVE when target is to the RIGHT
  obs[3]  wz    POSITIVE for CCW rotation from above (nose-left)
                NEGATIVE for CW  rotation from above (nose-right)
  obs[6..13] are the two older vision frames, obs[18..23] the three past actions.

Human-readable bearing (shown in live stats, NOT sent to policy)
  0°   = target directly ahead
  +90° = target to the LEFT
  -90° = target to the RIGHT
"""
import argparse
import json
import math
import os
import queue
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

try:
    from dt_apriltags import Detector
except ModuleNotFoundError:
    from pupil_apriltags import Detector

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from deploy.pi.obs_assembler import ObsAssemblerCPG24D, VisionState
from deploy.shared.tag_tracker import (
    TagTracker, _roi_detect, _fullframe_detect, ANCHOR_INTERVAL, TARGET_INTERVAL,
)

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ── Pi sensor protocol (must match pi_sensor_server.py) ───────────────────────
SENSOR_PORT = 5007
# wx, wy, wz, motor_pos_rad, motor_vel_rad_s, _reserved, timestamp
SENSOR_FMT  = "!fffffff"
SENSOR_SIZE = struct.calcsize(SENSOR_FMT)

# ── ANSI colours ──────────────────────────────────────────────────────────────
_G   = "\033[92m"
_R   = "\033[91m"
_Y   = "\033[93m"
_C   = "\033[96m"
_DIM = "\033[90m"
_B   = "\033[1m"
_RST = "\033[0m"

def _clr():
    sys.stdout.write("\033[2J\033[H")

# ── Camera helpers (same transforms as vision_streamer.py) ────────────────────
_SETTINGS  = os.path.join(_REPO, "deploy", "configs", "camera_settings.json")
_TUNE_FILE = os.path.join(_REPO, "deploy", "configs", "image_tune.json")


def _load_tune():
    """Load alpha/beta/blur/quad_decimate from image_tune.json (same file as vision_streamer)."""
    try:
        with open(_TUNE_FILE) as f:
            d = json.load(f)
        alpha    = float(d.get("alpha",         1.0))
        beta     = float(d.get("beta",          0.0))
        blur     = int(d.get("blur",            0))
        decimate = float(d.get("quad_decimate", 4.0))
        print(f"[Tune] Loaded: alpha={alpha}  beta={beta}  blur={blur}  "
              f"quad_decimate={decimate:.0f}")
        return alpha, beta, blur, decimate
    except (FileNotFoundError, json.JSONDecodeError):
        return 1.0, 0.0, 0, 4.0


def _load_camera(camera_name: str):
    try:
        with open(_SETTINGS) as f:
            data = json.load(f)
        cam = data.get("cameras", {}).get(camera_name, {})
        fx, fy, cx, cy = cam.get("camera_params", [1310.0, 1310.0, 960.0, 540.0])
        dist = np.array(cam.get("dist_coeffs", [0.0] * 5), dtype=np.float64)
    except FileNotFoundError:
        fx, fy, cx, cy = 1310.0, 1310.0, 960.0, 540.0
        dist = np.zeros(5)
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return mtx, dist, [fx, fy, cx, cy]


def _world_xy(t_tag, t_anchor, R_anchor):
    """World-frame (x, y) of a tag.  Y negated for right-handed frame."""
    p = R_anchor.T @ (t_tag - t_anchor)
    return float(p[0]), -float(p[1])


def _extract_yaw(R_tag, R_anchor):
    """Fish heading in world frame — tail direction, matching MuJoCo convention.

    Tag Y-axis points toward the nose; adding π (via negating fwd[0]) gives
    the tail direction, which is what the obs_assembler rotation formula expects
    (vec_x < 0 when target is directly ahead, matching env_sofi training).
    The -fwd[1] accounts for camera Y-down → world Y-up sign flip.
    """
    R_fw = R_anchor.T @ R_tag
    fwd  = R_fw[:, 1]
    return float(math.atan2(fwd[1], -fwd[0]))


def _bearing_label(deg: float) -> str:
    """Human-readable direction: 0°=AHEAD, +90°=LEFT, -90°=RIGHT, ±180°=BEHIND."""
    if abs(deg) < 45:   return "AHEAD"
    if abs(deg) > 135:  return "BEHIND"
    return "LEFT" if deg > 0 else "RIGHT"


# ── Pi sensor listener ────────────────────────────────────────────────────────
class _PiSensor:
    """Receives IMU+motor packets from pi_sensor_server.py."""

    _STALE_S = 2.0

    def __init__(self, port: int):
        self._data = {"wx": 0., "wy": 0., "wz": 0., "pos_rad": 0., "vel_rad_s": 0.}
        self._last_rx = 0.0
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.settimeout(0.5)
            self._sock.bind(("", port))
            ok = True
        except OSError as e:
            print(f"[PiSensor] Cannot bind port {port}: {e}")
            ok = False
        if ok:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw, _ = self._sock.recvfrom(256)
                if len(raw) >= SENSOR_SIZE:
                    v = struct.unpack(SENSOR_FMT, raw[:SENSOR_SIZE])
                    with self._lock:
                        self._data = {
                            "wx": v[0], "wy": v[1], "wz": v[2],
                            "pos_rad": v[3], "vel_rad_s": v[4],
                        }
                        self._last_rx = time.time()
            except socket.timeout:
                pass
            except Exception:
                pass

    def get(self):
        """Returns (data_dict, connected_bool)."""
        with self._lock:
            ok = (time.time() - self._last_rx) < self._STALE_S and self._last_rx > 0
            return dict(self._data), ok

    def stop(self):
        self._stop.set()


# ── Test definitions ──────────────────────────────────────────────────────────
# check_fn(obs, extra) → True/False
# extra keys: fps, tags_visible, bearing_deg, measured_dist

def _make_tests(dist_holder: list):
    """Build and return the test list.

    dist_holder is a mutable list [float | None] shared with test 2 so the
    user-entered distance is visible inside the lambda.
    """
    return [
        # ─────────────────────────────────────────────────────────── test 1 ──
        {
            "name": "Vision Alive",
            "instruction": [
                "Place all 3 AprilTags so the camera can see them:",
                "  Tag 0 = fish body     Tag 1 = anchor (fixed)     Tag 2 = target",
                "Check that FPS is stable above 20.",
            ],
            "need_imu":   False,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      None,
                    "name":         "tags 0+1+2",
                    "expected_str": "{0, 1, 2} all seen",
                    "check_fn":     lambda obs, e: e["tags_visible"] >= {0, 1, 2},
                },
                {
                    "obs_idx":      None,
                    "name":         "fps",
                    "expected_str": "> 20",
                    "check_fn":     lambda obs, e: e["fps"] > 20,
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 2 ──
        {
            "name": "Distance — obs[14]",
            "instruction": [
                "Place the fish at a KNOWN distance from the target.",
                "Any orientation is fine — only position matters.",
                "You will be prompted to enter the measured distance (m).",
            ],
            "need_imu":   False,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      14,
                    "name":         "dist",
                    "expected_str": "≈ measured ± 0.10 m",
                    "check_fn":     lambda obs, e: (
                        dist_holder[0] is not None
                        and abs(obs[14] - dist_holder[0]) < 0.10
                    ),
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 3 ──
        {
            "name": "Fish nose → Target  (obs[15] < 0, obs[16] ≈ 0)",
            "instruction": [
                "Point the fish nose DIRECTLY at the target tag.",
                "Live bearing below should read ≈ 0°.",
                "",
                "Why obs[15] < 0: MuJoCo fish forward = body −X.",
                "Target directly ahead → vec_x is in the −X direction → negative.",
            ],
            "need_imu":   False,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      15,
                    "name":         "vec_x",
                    "expected_str": "< −0.10  (≈ −dist)",
                    "check_fn":     lambda obs, e: obs[15] < -0.10,
                },
                {
                    "obs_idx":      16,
                    "name":         "vec_y",
                    "expected_str": "≈ 0  (|vy| < 0.15 m)",
                    "check_fn":     lambda obs, e: abs(obs[16]) < 0.15,
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 4 ──
        {
            "name": "Target to fish's LEFT  (obs[16] < 0)",
            "instruction": [
                "Rotate the fish so the target is 90° to its LEFT.",
                "  → Fish nose points away from target, target is on the left side.",
                "Live bearing below should read ≈ +90°.",
                "",
                "Why obs[16] < 0: facing body −X, left = body −Y → vec_y negative.",
            ],
            "need_imu":   False,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      16,
                    "name":         "vec_y",
                    "expected_str": "< −0.05  (target LEFT)",
                    "check_fn":     lambda obs, e: obs[16] < -0.05,
                },
                {
                    "obs_idx":      15,
                    "name":         "vec_x",
                    "expected_str": "≈ 0  (not pointing at it)",
                    "check_fn":     lambda obs, e: abs(obs[15]) < max(0.30, 0.35 * obs[14]),
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 5 ──
        {
            "name": "Target to fish's RIGHT  (obs[16] > 0)",
            "instruction": [
                "Rotate the fish so the target is 90° to its RIGHT.",
                "Live bearing below should read ≈ −90°.",
                "",
                "Why obs[16] > 0: facing body −X, right = body +Y → vec_y positive.",
            ],
            "need_imu":   False,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      16,
                    "name":         "vec_y",
                    "expected_str": "> +0.05  (target RIGHT)",
                    "check_fn":     lambda obs, e: obs[16] > 0.05,
                },
                {
                    "obs_idx":      15,
                    "name":         "vec_x",
                    "expected_str": "≈ 0  (not pointing at it)",
                    "check_fn":     lambda obs, e: abs(obs[15]) < max(0.30, 0.35 * obs[14]),
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 6 ──
        {
            "name": "Gyro — Yaw CCW  (obs[3] wz > 0)",
            "instruction": [
                "Slowly spin the fish COUNTERCLOCKWISE from above",
                "(nose moves to the LEFT while body stays level).",
                "Hold the rotation and watch obs[3] (wz).",
                "",
                "Expected: wz > +0.2 rad/s while rotating CCW.",
            ],
            "need_imu":   True,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      3,
                    "name":         "wz (yaw CCW)",
                    "expected_str": "> +0.2 rad/s",
                    "check_fn":     lambda obs, e: obs[3] > 0.2,
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 7 ──
        {
            "name": "Gyro — Yaw CW  (obs[3] wz < 0)",
            "instruction": [
                "Slowly spin the fish CLOCKWISE from above",
                "(nose moves to the RIGHT).",
                "Hold the rotation and watch obs[3] (wz).",
                "",
                "Expected: wz < −0.2 rad/s while rotating CW.",
            ],
            "need_imu":   True,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      3,
                    "name":         "wz (yaw CW)",
                    "expected_str": "< −0.2 rad/s",
                    "check_fn":     lambda obs, e: obs[3] < -0.2,
                },
            ],
        },

        # ─────────────────────────────────────────────────────────── test 8 ──
        {
            "name": "Gyro — Roll/Pitch  (Pi sensor only — not in 24D obs)",
            "instruction": [
                "CPG24D policy does not use roll (wx) or pitch (wy) —",
                "only yaw rate (wz) is in the obs vector at index [3].",
                "",
                "Check the Pi sensor server terminal directly:",
                "  Tilt right-side down → wx should spike",
                "  Tilt nose up        → wy should spike",
                "",
                "Press ENTER to skip or s to skip.",
            ],
            "need_imu":   True,
            "need_motor": False,
            "checks": [],
        },

        # ─────────────────────────────────────────────────────────── test 9 ──
        {
            "name": "Gyro — at-rest noise check  (obs[3] wz ≈ 0)",
            "instruction": [
                "Place the fish completely still on a flat surface.",
                "Watch obs[3] (wz) — should read near zero.",
                "",
                "Expected: |wz| < 0.05 rad/s at rest.",
            ],
            "need_imu":   True,
            "need_motor": False,
            "checks": [
                {
                    "obs_idx":      3,
                    "name":         "wz (at rest)",
                    "expected_str": "|wz| < 0.05 rad/s",
                    "check_fn":     lambda obs, e: abs(obs[3]) < 0.05,
                },
            ],
        },

        # ────────────────────────────────────────────────────────── test 10 ──
        {
            "name": "Motor — Centre Position  (obs[0] ≈ 1, obs[1] ≈ 0)",
            "instruction": [
                "Motor should be homed at zero position (tail centred).",
                "This is set automatically by deploy_main.py at startup.",
                "",
                "Expected: obs[0] cos(θ) > 0.9,  obs[1] sin(θ) near 0.",
            ],
            "need_imu":   False,
            "need_motor": True,
            "checks": [
                {
                    "obs_idx":      0,
                    "name":         "cos(θ_motor)",
                    "expected_str": "> 0.90  (θ ≈ 0)",
                    "check_fn":     lambda obs, e: obs[0] > 0.90,
                },
                {
                    "obs_idx":      1,
                    "name":         "sin(θ_motor)",
                    "expected_str": "< 0.15  (θ ≈ 0)",
                    "check_fn":     lambda obs, e: abs(obs[1]) < 0.15,
                },
            ],
        },
    ]


# ── Obs dimension labels ──────────────────────────────────────────────────────
_OBS_LABEL = [
    "cos_θm", "sin_θm", "ω_m",
    "wz",     "cos_φ",  "sin_φ",
    "f0.d",   "f0.vx",  "f0.vy",  "f0.age",
    "f1.d",   "f1.vx",  "f1.vy",  "f1.age",
    "f2.d",   "f2.vx",  "f2.vy",  "f2.age",
    "a0.ω",   "a0.θc",
    "a1.ω",   "a1.θc",
    "a2.ω",   "a2.θc",
]


# ── Display ───────────────────────────────────────────────────────────────────
def _st(result):
    if result is True:  return f"{_G}✓{_RST}"
    if result is False: return f"{_R}✗{_RST}"
    return f"{_DIM}—{_RST}"


def _render(test_idx, n_tests, test, obs, extra, check_results, sensor_ok, tune=None):
    W = 72
    out = []

    def ln(s=""): out.append(s + "\n")

    ln(f"{_C}{'─'*W}{_RST}")
    ln(f"{_B}{_C}  SOFI OBS VERIFIER   Test {test_idx+1}/{n_tests}: {test['name']}{_RST}")
    ln(f"{_C}{'─'*W}{_RST}")

    for line in test["instruction"]:
        ln(f"  {line}")
    ln()

    # Pi sensor status (only shown if test needs it)
    needs_pi = test.get("need_imu") or test.get("need_motor")
    if needs_pi:
        if sensor_ok:
            ln(f"  Pi sensor: {_G}● connected{_RST}")
        else:
            ln(f"  Pi sensor: {_R}✗ not connected{_RST}  — run pi_sensor_server.py on Pi")
        ln()

    # Key checks table
    ln(f"{_B}  {'Obs':<8} {'Name':<16} {'Expected':<22} {'Actual':>10}  St{_RST}")
    ln(f"  {'─'*66}")
    for chk, res in zip(test["checks"], check_results):
        idx  = chk["obs_idx"]
        istr = f"[{idx}]" if idx is not None else "[--]"

        if obs is not None and idx is not None:
            val_str = f"{obs[idx]:+.4f}"
        elif idx is None and extra is not None:
            name = chk["name"]
            if "fps" in name:
                val_str = f"{extra.get('fps', 0):.1f}"
            elif "tag" in name:
                val_str = str(sorted(extra.get("tags_visible", set())))
            else:
                val_str = "?"
        else:
            val_str = "---"

        # If needs Pi but not connected, suppress check status
        if needs_pi and not sensor_ok and idx in (0, 1, 2, 3, 4, 5):
            st = f"{_DIM}n/a{_RST}"
        else:
            st = _st(res)

        ln(f"  {istr:<8} {chk['name']:<16} {chk['expected_str']:<22} {val_str:>10}  {st}")
    ln()

    # Compact 24-dim obs grid (4 per row)
    ln(f"{_B}  All 24 observations:{_RST}")
    if obs is not None:
        for r in range(0, 24, 4):
            row = "  "
            for i in range(r, min(r + 4, 24)):
                row += f"[{i:2d}]{_OBS_LABEL[i]:<7}={obs[i]:+7.3f}  "
            ln(row.rstrip())
    else:
        ln(f"  {_Y}(waiting for fish + target tags…){_RST}")
    ln()

    # Live stats
    if extra is not None:
        b    = extra.get("bearing_deg", float("nan"))
        fps  = extra.get("fps", 0.0)
        tags = sorted(extra.get("tags_visible", set()))
        d    = obs[14] if obs is not None else float("nan")
        tune_str = ""
        if tune:
            tune_str = f"  α={tune[0]:.1f} β={tune[1]:.0f} blur={tune[2]}"
        if not extra.get("roi_mode"):
            mode_str = "FULL"
        elif extra.get("sparse_target"):
            mode_str = "ROI-s"
        else:
            mode_str = "ROI"
        ln(f"  fps={fps:.1f} [{mode_str}]  tags={tags}  bearing={b:+.0f}° [{_bearing_label(b)}]  dist={d:.3f} m{tune_str}")
    ln()

    # Pass/fail summary for this test
    n_pass  = sum(1 for r in check_results if r is True)
    n_check = len(check_results)
    if all(r is True for r in check_results) and n_check > 0:
        status = f"{_G}ALL PASSING{_RST}"
    elif any(r is False for r in check_results):
        status = f"{_R}FAILING{_RST}"
    else:
        status = f"{_Y}checking…{_RST}"
    ln(f"  Status: {status}  ({n_pass}/{n_check} checks pass)")

    ln(f"{_C}{'─'*W}{_RST}")
    ln(f"  {_B}ENTER{_RST}=Pass  {_B}f{_RST}=Fail  {_B}s{_RST}=Skip  {_B}r{_RST}=Reset buffer  {_B}q{_RST}=Quit")

    _clr()
    sys.stdout.write("".join(out))
    sys.stdout.flush()


# ── Keyboard input thread ─────────────────────────────────────────────────────
_cmd_q: "queue.Queue[str]" = queue.Queue()

def _start_input_thread():
    def _reader():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                _cmd_q.put(line.strip().lower() or "pass")
            except Exception:
                break
    threading.Thread(target=_reader, daemon=True).start()


# ── Single-test runner ────────────────────────────────────────────────────────
def _run_test(test_idx, n_tests, test, cap, detector, detect_params,
              detect_w, detect_h, img_alpha, img_beta, img_blur,
              tag_size, fish_id, anchor_id, target_id,
              config, sensor: _PiSensor, dist_holder: list,
              no_roi: bool = False, full_roi: bool = False):

    assembler  = ObsAssemblerCPG24D(config)
    cpg_phase  = 0.0
    prev_omega = 0.0

    fish_trk   = TagTracker()
    anchor_trk = TagTracker()
    target_trk = TagTracker()
    total_frames = 0
    _roi_mode    = False

    fps_count  = 0
    fps_time   = time.time()
    fps_val    = 0.0
    cpg_time   = time.time()
    last_display_t = 0.0

    obs_last   = None
    extra_last = {"fps": 0.0, "tags_visible": set(), "bearing_deg": 0.0, "roi_mode": False}
    frame_seq  = 0

    result = "skip"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        _do_display = (now - last_display_t >= 0.5)

        # FPS
        fps_count += 1
        if now - fps_time >= 1.0:
            fps_val   = fps_count / (now - fps_time)
            fps_count = 0
            fps_time  = now

        # CPG phase advance (1 Hz for display purposes only)
        cpg_phase += 2.0 * math.pi * 1.0 * (now - cpg_time)
        cpg_time   = now

        # AprilTag detection — ROI tracking, same logic as vision_streamer
        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        total_frames += 1
        target_redetect = True  # default for full-frame path

        if no_roi or fish_trk.use_fullframe or target_trk.use_fullframe or anchor_trk.use_fullframe:
            # Full-frame path: preprocess entire image once
            gray = gray_raw
            if img_alpha != 1.0 or img_beta != 0.0:
                gray = cv2.convertScaleAbs(gray, alpha=img_alpha, beta=img_beta)
            if img_blur >= 3:
                gray = cv2.GaussianBlur(gray, (img_blur, img_blur), 0)
            tag_map  = _fullframe_detect(
                detector, gray, detect_params, tag_size,
                {fish_id: fish_trk, anchor_id: anchor_trk, target_id: target_trk},
            )
            _roi_mode = False
        else:
            # ROI path: pass raw gray — each _roi_detect preprocesses only its crop
            anchor_redetect  = (total_frames % ANCHOR_INTERVAL == 0)
            target_redetect  = full_roi or (total_frames % TARGET_INTERVAL == 0)
            tag_map = {
                fish_id: _roi_detect(detector, gray_raw, fish_id, fish_trk,
                                     detect_params, tag_size, detect_w, detect_h,
                                     alpha=img_alpha, beta=img_beta, blur=img_blur),
            }
            if target_redetect:
                tag_map[target_id] = _roi_detect(detector, gray_raw, target_id, target_trk,
                                                 detect_params, tag_size, detect_w, detect_h,
                                                 alpha=img_alpha, beta=img_beta, blur=img_blur)
            if anchor_redetect and anchor_trk.initialised:
                tag_map[anchor_id] = _roi_detect(detector, gray_raw, anchor_id, anchor_trk,
                                                 detect_params, tag_size, detect_w, detect_h,
                                                 alpha=img_alpha, beta=img_beta, blur=img_blur)
            _roi_mode = True

        # Preview window — rate-limited to 2 fps; waitKey runs every frame for responsiveness
        if _do_display:
            preview = cv2.resize(frame, (640, 360))
            sx = 640 / detect_w
            sy = 360 / detect_h
            for tid, trk in [(fish_id, fish_trk), (anchor_id, anchor_trk), (target_id, target_trk)]:
                if not trk.initialised:
                    continue
                detected_now = tag_map.get(tid) is not None
                # Anchor and target (in sparse mode) are only re-checked on their
                # respective intervals.  Show green whenever the tracker has a good
                # last-known pose, not just on the frames they happen to be in tag_map.
                is_interval_tag = (tid == anchor_id) or (tid == target_id and not target_redetect)
                healthy = detected_now or (is_interval_tag and trk.miss_count == 0)
                colour = (0, 220, 0) if healthy else (0, 100, 255)
                pt = (int(trk.cx_full * sx), int(trk.cy_full * sy))
                cv2.circle(preview, pt, 18, colour, 2)
                cv2.putText(preview, f"id={tid}", (pt[0] + 12, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.imshow("OBS VERIFIER — camera", preview)
        # Handle keys from the OpenCV window (works when that window is in focus)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            _cmd_q.put("quit")
        elif key == 13 or key == ord('p'):   # ENTER or p
            _cmd_q.put("pass")
        elif key == ord('f'):
            _cmd_q.put("fail")
        elif key == ord('s'):
            _cmd_q.put("skip")
        elif key == ord('r'):
            _cmd_q.put("reset")

        fish_x = fish_y = fish_yaw = 0.0
        target_x = target_y = 0.0
        have_fish   = False
        have_target = False

        if tag_map.get(fish_id) is not None and anchor_trk.initialised:
            fish_x, fish_y = _world_xy(fish_trk.pose_t, anchor_trk.pose_t, anchor_trk.pose_R)
            fish_yaw = _extract_yaw(fish_trk.pose_R, anchor_trk.pose_R)
            have_fish = True

        if tag_map.get(target_id) is not None and anchor_trk.initialised:
            target_x, target_y = _world_xy(target_trk.pose_t, anchor_trk.pose_t, anchor_trk.pose_R)
            have_target = True
        elif not target_redetect and target_trk.initialised and anchor_trk.initialised:
            # Sparse skip — no detection this frame; carry last known target position
            target_x, target_y = _world_xy(target_trk.pose_t, anchor_trk.pose_t, anchor_trk.pose_R)
            have_target = True

        # Human-readable bearing (NOT sent to policy; uses non-negated formula)
        dx_  = target_x - fish_x
        dy_  = target_y - fish_y
        vx_  =  math.cos(fish_yaw) * dx_ + math.sin(fish_yaw) * dy_
        vy_  = -math.sin(fish_yaw) * dx_ + math.cos(fish_yaw) * dy_
        bearing_deg = math.degrees(math.atan2(-vy_, -vx_))

        # Anchor is only in tag_map every ANCHOR_INTERVAL frames; include it in
        # tags_visible whenever the tracker is healthy (miss_count == 0), so the
        # display doesn't show it blinking absent 29/30 frames in ROI mode.
        tags_vis = set(t for t, v in tag_map.items() if v is not None)
        if anchor_trk.initialised and anchor_trk.miss_count == 0:
            tags_vis.add(anchor_id)
        if not target_redetect and target_trk.initialised and target_trk.miss_count == 0:
            tags_vis.add(target_id)
        extra_last = {
            "fps":           fps_val,
            "tags_visible":  tags_vis,
            "bearing_deg":   bearing_deg,
            "roi_mode":      _roi_mode,
            "sparse_target": not full_roi and _roi_mode,
        }

        # Sensor data from Pi (zeros if not connected)
        sensor_data, sensor_ok = sensor.get()
        imu   = {"wx": sensor_data["wx"],     "wy": sensor_data["wy"],
                  "wz": sensor_data["wz"]}
        motor = {"pos_rad": sensor_data["pos_rad"], "vel_rad_s": sensor_data["vel_rad_s"]}

        # Build obs when we have enough vision data
        if have_fish and have_target:
            frame_seq += 1
            vs = VisionState(
                fish_x=fish_x, fish_y=fish_y, fish_yaw=fish_yaw,
                target_x=target_x, target_y=target_y,
                flags=0, pi_arrival_time=now,
                frame_seq=frame_seq, laptop_timestamp=now,
            )
            obs_last = assembler.build(motor, imu, vs, cpg_phase, prev_omega)

        # Evaluate checks
        needs_pi = test.get("need_imu") or test.get("need_motor")
        check_results = []
        for chk in test["checks"]:
            if obs_last is None:
                check_results.append(None)
            elif needs_pi and not sensor_ok and chk["obs_idx"] in (0, 1, 2, 3, 4, 5):
                check_results.append(None)
            else:
                try:
                    check_results.append(bool(chk["check_fn"](obs_last, extra_last)))
                except Exception:
                    check_results.append(None)

        # Render — rate-limited to 2 fps (same interval as preview)
        if _do_display:
            last_display_t = now
            tune = (img_alpha, img_beta, img_blur)
            _render(test_idx, n_tests, test, obs_last, extra_last, check_results, sensor_ok, tune)

        # Poll for keyboard command
        try:
            cmd = _cmd_q.get_nowait()
        except queue.Empty:
            cmd = None

        if cmd is not None:
            # Test 2: accept a numeric string as the measured distance
            if test_idx == 1 and dist_holder[0] is None:
                try:
                    dist_holder[0] = float(cmd)
                    continue  # stay in loop to show the updated check
                except ValueError:
                    pass  # not a float — fall through to normal command handling

            if cmd in ("", "pass", "p"):
                result = "pass"
                break
            elif cmd in ("f", "fail"):
                result = "fail"
                break
            elif cmd in ("s", "skip"):
                result = "skip"
                break
            elif cmd in ("r", "reset"):
                assembler = ObsAssemblerCPG24D(config)
                cpg_phase = 0.0
            elif cmd in ("q", "quit"):
                result = "quit"
                break

    cv2.destroyAllWindows()
    return result


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SOFI RL Observation Verifier")
    ap.add_argument("--config",      default="deploy/configs/cpg_24d.json",
                    help="Path to env config JSON (relative to repo root)")
    ap.add_argument("--device",      type=int, default=0,
                    help="Camera device index (0, 1, …)")
    ap.add_argument("--tag-size",    type=float, default=0.080,
                    help="AprilTag physical size in metres")
    ap.add_argument("--camera",      default="simonPhone",
                    help="Camera profile name in camera_settings.json")
    ap.add_argument("--sensor-port", type=int, default=SENSOR_PORT,
                    help="UDP port to receive Pi sensor packets on")
    ap.add_argument("--fps",    type=int, default=90,
                    help="Requested camera fps (default 90; capped to 30 on Linux/V4L2)")
    ap.add_argument("--no-roi",   action="store_true", dest="no_roi",
                    help="Disable ROI tracking — detect all tags full-frame every frame")
    ap.add_argument("--full-roi", action="store_true", dest="full_roi",
                    help="Detect target every frame in ROI mode (default: every 10th frame for fixed targets)")
    args = ap.parse_args()

    # Load config
    cfg_path = os.path.join(_REPO, args.config)
    with open(cfg_path) as f:
        config = json.load(f)
    tag_ids   = config.get("tag_ids", {"fish": 0, "anchor": 1, "target": 2})
    fish_id   = tag_ids.get("fish", 0)
    anchor_id = tag_ids.get("anchor", 1)
    target_id = tag_ids.get("target", 2)

    # Camera setup — match vision_streamer.py exactly (backend, resolution, codec)
    _, _, params = _load_camera(args.camera)
    fx, fy, cx, cy = params
    if sys.platform == "linux":
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS, min(args.fps, 30))  # V4L2/WSL can't sustain >30fps
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(1080 * 16 / 9))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit(f"[Error] Cannot open camera device {args.device}")
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dw = w
    dh = h
    detect_params = [fx, fy, cx, cy]

    img_alpha, img_beta, img_blur, quad_decimate = _load_tune()

    detector = Detector(
        families="tag36h11",
        nthreads=6,
        quad_decimate=quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    sensor   = _PiSensor(args.sensor_port)

    dist_holder = [None]
    tests   = _make_tests(dist_holder)
    n_tests = len(tests)

    # Welcome screen
    sys.stdout.write("\033[2J\033[H")
    print(f"\n  {_B}{_C}SOFI RL Observation Verifier{_RST}\n")
    print(f"  Config : {args.config}")
    print(f"  Camera : device={args.device}  {w}×{h}  detect={dw}×{dh}")
    print(f"  Tune   : α={img_alpha}  β={img_beta}  blur={img_blur}  (from image_tune.json)")
    print(f"  Pi     : listening on UDP port {args.sensor_port}")
    print(f"           (run pi_sensor_server.py on Pi for IMU+motor tests)")
    print()
    print(f"  {n_tests} tests will step through each observation dimension.")
    print(f"  Controls: ENTER=Pass  f=Fail  s=Skip  r=Reset buffer  q=Quit")
    print()
    print(f"  Press ENTER to begin…", end="", flush=True)
    sys.stdin.readline()
    # Start input thread AFTER the initial ENTER so it doesn't steal the keypress
    _start_input_thread()

    results_log = []

    for i, test in enumerate(tests):
        outcome = _run_test(
            test_idx=i, n_tests=n_tests, test=test,
            cap=cap, detector=detector,
            detect_params=detect_params,
            detect_w=dw, detect_h=dh,
            img_alpha=img_alpha, img_beta=img_beta, img_blur=img_blur,
            tag_size=args.tag_size,
            fish_id=fish_id, anchor_id=anchor_id, target_id=target_id,
            config=config, sensor=sensor, dist_holder=dist_holder,
            no_roi=args.no_roi,
            full_roi=args.full_roi,
        )
        results_log.append((test["name"], outcome))

        if outcome == "quit":
            break

        # Brief inter-test flash
        sys.stdout.write("\033[2J\033[H")
        col = _G if outcome == "pass" else (_R if outcome == "fail" else _Y)
        print(f"\n  Test {i+1}/{n_tests}: {col}{outcome.upper()}{_RST} — {test['name']}")
        time.sleep(1.0)

    cap.release()
    sensor.stop()

    # Summary
    sys.stdout.write("\033[2J\033[H")
    print(f"\n  {_B}{_C}{'─'*60}{_RST}")
    print(f"  {_B}Test Summary{_RST}")
    print(f"  {'─'*60}")
    for name, res in results_log:
        col = _G if res == "pass" else (_R if res == "fail" else _Y)
        print(f"  {col}{res.upper():<6}{_RST}  {name}")
    n_pass = sum(1 for _, r in results_log if r == "pass")
    n_fail = sum(1 for _, r in results_log if r == "fail")
    n_skip = len(results_log) - n_pass - n_fail
    print(f"\n  {_G}{n_pass} passed{_RST}   {_R}{n_fail} failed{_RST}   {_Y}{n_skip} skipped{_RST}")
    print()


if __name__ == "__main__":
    main()
