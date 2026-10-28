"""
Laptop-side vision streamer for SOFI RL deployment.

Detects 3 AprilTags (fish ID=0, anchor ID=1, target ID=2),
converts to world frame using the anchor, and streams 37-byte
UDP pose packets to the Pi at camera fps.

Usage:
    python deploy/laptop/vision_streamer.py \
        --pi_ip raspberrypi.local \
        --tag_size 0.080 \
        --env-config deploy/configs/cpg_24d.json

Also handles the Pi clock sync handshake at startup.
"""
import argparse
import json
import math
import os
import queue
import socket
import struct
import sys
import subprocess
import threading
import time
from datetime import datetime

import cv2
import numpy as np

try:
    from dt_apriltags import Detector
except ModuleNotFoundError:
    from pupil_apriltags import Detector

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _REPO_ROOT)

from deploy.shared.protocol import (
    PACKET_SIZE, POSE_PORT, SYNC_PORT, CTRL_PORT,
    FLAGS_FISH, FLAGS_TARGET, FLAGS_ANCHOR,
    CTRL_READY, CTRL_STOP,
    make_start_packet, make_done_packet, parse_done_packet, parse_ready_name,
    pack_pose,
)
from deploy.shared.tag_tracker import (
    TagTracker, _roi_detect, _fullframe_detect, ANCHOR_INTERVAL, TARGET_INTERVAL,
)

SETTINGS_FILE = os.path.join(_REPO_ROOT, "deploy", "configs", "camera_settings.json")
_TUNE_FILE    = os.path.join(_REPO_ROOT, "deploy", "configs", "image_tune.json")


def _load_tune(fallback_alpha=1.0, fallback_beta=0.0, fallback_blur=0, fallback_decimate=4.0):
    try:
        with open(_TUNE_FILE) as f:
            d = json.load(f)
        alpha    = float(d.get("alpha",         fallback_alpha))
        beta     = float(d.get("beta",          fallback_beta))
        blur     = int(d.get("blur",            fallback_blur))
        decimate = float(d.get("quad_decimate", fallback_decimate))
        print(f"[Tune] Loaded: alpha={alpha:.1f}  beta={beta:.0f}  blur={blur}  "
              f"quad_decimate={decimate:.0f}")
        return alpha, beta, blur, decimate
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback_alpha, fallback_beta, fallback_blur, fallback_decimate


def _save_tune(alpha: float, beta: float, blur: int) -> None:
    # Preserve quad_decimate (set by tune_vision.py) — live key tuning only touches
    # alpha/beta/blur so we must not overwrite the decimation on each keystroke.
    try:
        existing = {}
        try:
            with open(_TUNE_FILE) as f:
                existing = json.load(f)
        except Exception:
            pass
        existing.update({"alpha": alpha, "beta": beta, "blur": blur})
        with open(_TUNE_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception:
        pass


def _fetch_run(run_folder: str) -> None:
    """Launch fetch_data.sh rl <run_folder> — works on both Linux and Windows (WSL)."""
    if sys.platform == "win32":
        # Convert Windows path to WSL path so bash can find the script
        result = subprocess.run(
            ["wsl.exe", "wslpath", _REPO_ROOT.replace("\\", "/")],
            capture_output=True, text=True,
        )
        wsl_root = result.stdout.strip()
        subprocess.Popen(
            ["wsl.exe", "bash", "-c",
             f"cd '{wsl_root}' && ./scripts/fetch_data.sh rl '{run_folder}'"],
        )
    else:
        subprocess.Popen(
            ["bash", os.path.join(_REPO_ROOT, "scripts", "fetch_data.sh"),
             "rl", run_folder],
            cwd=_REPO_ROOT,
        )


class AsyncVideoRecorder:
    """Writes video frames in a background thread so the capture loop never blocks.

    Uses MJPEG codec — each frame is an independent JPEG, ~10x faster to encode
    than mp4v (software MPEG-4) with no quality penalty for post-processing.
    Saves as .avi (MJPEG is natively an AVI codec on Windows).

    Timestamp ownership: the recorder also owns frame_timestamps.csv and writes
    exactly one row PER FRAME ACTUALLY WRITTEN, in the worker thread. A frame
    that is dropped because the encode queue is full produces NO timestamp row,
    so the CSV row count always equals the AVI frame count and frame_index is
    the true AVI position. This makes VFR replay exact even when frames drop —
    each surviving frame is shown for the real gap until the next one.

    (Previously the timestamp was written in the capture loop for every captured
    frame regardless of drops, so frame_index drifted from the AVI position and
    replays sped up wherever the encoder had fallen behind.)
    """

    def __init__(self, path: str, fps: float, width: int, height: int, buf: int = 1024):
        avi_path = os.path.splitext(path)[0] + ".avi"
        ts_path  = os.path.join(os.path.dirname(avi_path), "frame_timestamps.csv")
        self._writer = cv2.VideoWriter(
            avi_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
        )
        self._path = avi_path
        self._ts_file = open(ts_path, "w")
        self._ts_file.write("frame_index,laptop_time\n")
        self._written = 0
        self._q: queue.Queue = queue.Queue(maxsize=buf)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._dropped = 0

    def push(self, frame, capture_time: float) -> None:
        try:
            self._q.put_nowait((frame, capture_time))
        except queue.Full:
            self._dropped += 1

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            frame, capture_time = item
            self._writer.write(frame)
            self._ts_file.write(f"{self._written},{capture_time:.6f}\n")
            self._written += 1
        self._writer.release()
        self._ts_file.flush()
        self._ts_file.close()

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=60)
        if self._dropped:
            print(f"[Recorder] {self._dropped} frames dropped (queue full) — "
                  f"{self._written} written (timestamps stay matched, replay exact)")
        print(f"[Recorder] Saved → {self._path}")


def world_frame_transform(t_tag: np.ndarray, t_anchor: np.ndarray, R_anchor: np.ndarray):
    """Project tag camera-frame position into world frame (x,y) using anchor.

    World frame: origin at anchor tag.  Y is negated relative to the raw
    anchor-frame Y so that the world is right-handed (+Y = left when facing
    +X), matching the MuJoCo training convention.  The camera's Y-down image
    convention would otherwise make the raw frame left-handed (+Y = right),
    flipping the sign of all lateral target vectors.

    Returns (world_x, world_y) in metres.
    """
    p_world = R_anchor.T @ (t_tag - t_anchor)
    return float(p_world[0]), -float(p_world[1])


def extract_yaw(R_tag: np.ndarray, R_anchor: np.ndarray) -> float:
    """Extract fish heading (yaw) in world frame from tag rotation matrices.

    Uses the tag Y-axis ("top" of the printed square) as the fish's swim direction —
    confirmed by the existing pipeline (02_process_swimming.py line 168:
    fwd = R_fish_to_world[:, 1]).

    Returns yaw in radians (CCW positive from world x-axis).
    """
    R_fish_world = R_anchor.T @ R_tag
    fwd = R_fish_world[:, 1]
    return float(math.atan2(fwd[1], -fwd[0]))


def _bearing_label(deg: float) -> str:
    """Human-readable direction: 0°=AHEAD, +90°=LEFT, -90°=RIGHT, ±180°=BEHIND."""
    if abs(deg) < 45:   return "AHEAD"
    if abs(deg) > 135:  return "BEHIND"
    return "LEFT" if deg > 0 else "RIGHT"


def load_camera_params(settings_file, camera_name=None):
    with open(settings_file) as f:
        data = json.load(f)
    name = camera_name or "overhead"
    cam = data.get("cameras", {}).get(name, {})
    fx, fy, cx, cy = cam.get("camera_params", [1310.0, 1310.0, 960.0, 540.0])
    dist = np.array(cam.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    print(f"Camera: {name}  fx={fx:.1f}  fy={fy:.1f}")
    return mtx, dist, [fx, fy, cx, cy]


def _run_sync_server(sync_port: int, stop_event: threading.Event) -> None:
    """Persistent UDP clock-sync responder.

    Binds to sync_port and responds to every ping from the Pi for the lifetime
    of the vision streamer. The Pi initiates sync after it receives its first
    pose packet (learning the laptop IP from the UDP source address).

    Protocol (Pi → laptop → Pi):
      Pi sends   8 B: pack("!d",   t1_pi)
      Laptop replies  16 B: pack("!dd", t1_pi, t2_laptop)
      Pi computes offset = t2_laptop − (t1_pi + t3_pi) / 2
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    try:
        sock.bind(("", sync_port))
    except OSError as e:
        print(f"[ClockSync] Cannot bind to port {sync_port}: {e}")
        return
    n_responded = 0
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(8)
            t2 = time.time()
            t1 = struct.unpack("!d", data)[0]
            sock.sendto(struct.pack("!dd", t1, t2), addr)
            n_responded += 1
            if n_responded == 1:
                print(f"[ClockSync] Sync request from Pi ({addr[0]}) — responding.")
        except socket.timeout:
            continue
        except Exception:
            continue
    sock.close()


def _run_ctrl_listener(ctrl_port: int, stop_event: threading.Event, state: dict) -> None:
    """Receives READY and DONE packets from Pi on CTRL_PORT, updates shared state dict."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    try:
        sock.bind(("", ctrl_port))
    except OSError as e:
        print(f"[Ctrl] Cannot bind port {ctrl_port}: {e}")
        return
    while not stop_event.is_set():
        try:
            data, addr = sock.recvfrom(128)
            if data == CTRL_READY or data.startswith(b"READY:"):
                run_name = parse_ready_name(data)
                state['pi_ready'] = True
                state['pi_name'] = run_name
                state['pi_ip_confirmed'] = addr[0]
                label = run_name if run_name else "(no name)"
                print(f"\n[Ctrl] Pi READY: {label}  ({addr[0]}) — press R to start")
            elif data.startswith(b"DONE:"):
                state['done_ts'] = parse_done_packet(data)
                state['pi_ready'] = False
                state['done_received'] = True
                print(f"\n[Ctrl] Pi DONE")
        except socket.timeout:
            continue
        except Exception:
            continue
    sock.close()


def _start_recording(name: str, fps: float, width: int, height: int):
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts_str}_{name}" if name else ts_str
    session_dir = os.path.join(_REPO_ROOT, "deploy", "recordings", folder_name)
    os.makedirs(session_dir, exist_ok=True)
    recorder = AsyncVideoRecorder(
        os.path.join(session_dir, "video.avi"),
        fps=fps, width=width, height=height,
    )
    print(f"[Recorder] Started → {session_dir}/  (async MJPEG)")
    return recorder


def main():
    parser = argparse.ArgumentParser(description="SOFI vision streamer")
    parser.add_argument("--pi_ip", default=None, help="Pi IP address. Omit for camera-only preview mode.")
    parser.add_argument("--env-config", default=None, help="Path to env_config.json (optional — only needed for non-default tag IDs)")
    parser.add_argument("--tag_size", type=float, default=0.080, help="Tag size in metres")
    parser.add_argument("--camera", default=None, help="Camera profile from vision_settings.json")
    parser.add_argument("--device", type=int, default=0, help="USB camera device index")
    parser.add_argument("--no_preview", action="store_true", help="Disable preview window entirely (max fps, headless)")
    parser.add_argument("--live_plot", action="store_true", help="Live diagnostic plots (vector, distance, speed)")
    parser.add_argument("--record_old", action="store_true", help="Record video + frame timestamps (starts immediately)")
    parser.add_argument("--record", action="store_true", help="Toggleable recording — press R in preview window to start/stop")
    parser.add_argument("--name", default=None, help="Fallback session name if Pi does not send one (normally not needed)")
    parser.add_argument("--fps", type=int, default=90, help="Requested camera fps")
    parser.add_argument("--autostart", action="store_true",
                        help="Send START immediately when Pi READY (headless sessions with --no_preview)")
    parser.add_argument("--no-target", action="store_true", dest="no_target",
                        help="Stream fish-only packets even when target tag is not visible (open-loop without target)")
    parser.add_argument("--no-roi", action="store_true", dest="no_roi",
                        help="Disable ROI tracking — detect all tags full-frame every frame (slower, pre-ROI behaviour)")
    parser.add_argument("--full-roi", action="store_true", dest="full_roi",
                        help="Detect target tag every frame in ROI mode (default: every 10th frame for fixed targets)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Contrast multiplier before tag detection (1.0=off, try 0.4–1.8)")
    parser.add_argument("--beta", type=float, default=0.0, help="Brightness offset before tag detection (0=off, try -50 to +50)")
    parser.add_argument("--blur", type=int, default=0, help="Gaussian blur kernel size before tag detection (0=off, must be odd e.g. 3,5,7)")
    args = parser.parse_args()

    # Live-tunable image preprocessing (A/Z=alpha, S/X=beta, D/C=blur)
    # Saved values from image_tune.json take priority; args are the factory defaults.
    img_alpha, img_beta, img_blur, quad_decimate = _load_tune(
        fallback_alpha=args.alpha,
        fallback_beta=args.beta,
        fallback_blur=args.blur if args.blur % 2 == 1 else 0,
    )

    tag_ids = {"fish": 0, "anchor": 1, "target": 2}
    tag_heading_flip = False
    if args.env_config:
        with open(args.env_config) as f:
            _cfg = json.load(f)
        tag_ids.update(_cfg.get("tag_ids", {}))
        tag_heading_flip = bool(_cfg.get("tag_heading_flip", False))
    if tag_heading_flip:
        print("[VisionStreamer] tag_heading_flip=True — fish yaw will be rotated 180°")
    fish_id = tag_ids["fish"]
    anchor_id = tag_ids["anchor"]
    target_id = tag_ids["target"]

    plotter = None
    if args.live_plot:
        from live_plotter import LivePlotter
        plotter = LivePlotter()
        plotter.start()
        print("[VisionStreamer] Live plots started.")

    mtx, dist, cam_params = load_camera_params(SETTINGS_FILE, args.camera)

    udp_sock = None
    pi_addr = None
    _sync_stop = threading.Event()
    if args.pi_ip:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        pi_addr = (args.pi_ip, POSE_PORT)
        threading.Thread(
            target=_run_sync_server, args=(SYNC_PORT, _sync_stop), daemon=True
        ).start()
        print("[VisionStreamer] Clock sync server ready (Pi will connect after first pose).")
    else:
        print("[VisionStreamer] No --pi_ip given — preview-only mode, not streaming to Pi.")

    # CTRL channel — receive READY/DONE from Pi, send START/STOP
    _ctrl_state = {'pi_ready': False, 'pi_ip_confirmed': None, 'pi_name': '', 'done_received': False, 'done_ts': None}
    _ctrl_stop = threading.Event()
    _ctrl_send_sock = None  # unbound socket for sending START/STOP
    if args.pi_ip:
        threading.Thread(
            target=_run_ctrl_listener,
            args=(CTRL_PORT, _ctrl_stop, _ctrl_state),
            daemon=True,
        ).start()
        _ctrl_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("[VisionStreamer] CTRL listener ready (waiting for Pi READY).")

    _current_run_folder: str = None  # tracks folder name for DONE → fetch

    detector = Detector(
        families="tag36h11",
        nthreads=6,
        quad_decimate=quad_decimate,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    if sys.platform == "linux":
        # WSL/Linux: V4L2, no MJPEG forcing (usbipd virtual USB can't sustain it)
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, min(args.fps, 30))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        # Windows native: MSMF with MJPEG for high-fps 1080p
        cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(1080 * 16 / 9))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    detect_params = cam_params

    recorder = None
    if args.record_old:
        recorder = _start_recording(args.name, actual_fps, actual_w, actual_h)

    fish_trk   = TagTracker()
    anchor_trk = TagTracker()
    target_trk = TagTracker()

    fish_x, fish_y, fish_yaw = 0.0, 0.0, 0.0
    target_x, target_y = 1.0, 0.0
    frame_seq    = 0
    total_frames = 0      # never resets — used for anchor re-detect interval
    _roi_mode    = False

    frame_count = 0      # resets every second for fps display
    fps_timer = time.time()
    last_pose_t = time.time()   # tracks last received pose packet — used to detect Pi dropout
    _PI_DROPOUT_S = 8.0         # stop recording if no pose packets for this long
    last_pkt_sent_t = time.time()   # tracks last time a pose packet was sent to Pi
    _PKT_WARN_S = 1.0               # warn if no packet sent for this long while recording
    _pkt_warn_printed = False

    if args.no_preview:
        print("[VisionStreamer] Running headless — press Ctrl+C to stop.")

    PREVIEW_W = 320          # render overlays at this resolution (cheap)
    PREVIEW_H = 180
    DISPLAY_W = 960          # upscale for display — INTER_NEAREST is essentially free
    DISPLAY_H = 540
    PREVIEW_INTERVAL = 0.5   # redraw at 2 fps
    last_preview_t = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[VisionStreamer] Camera read failed.")
                break

            capture_time = time.time()

            gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            total_frames += 1
            target_redetect = True  # default for full-frame path

            if args.no_roi or fish_trk.use_fullframe or target_trk.use_fullframe or anchor_trk.use_fullframe:
                # Full-frame path: preprocess entire image once
                gray = gray_raw
                if img_alpha != 1.0 or img_beta != 0.0:
                    gray = cv2.convertScaleAbs(gray, alpha=img_alpha, beta=img_beta)
                if img_blur >= 3:
                    gray = cv2.GaussianBlur(gray, (img_blur, img_blur), 0)
                tag_map  = _fullframe_detect(
                    detector, gray, detect_params, args.tag_size,
                    {fish_id: fish_trk, anchor_id: anchor_trk, target_id: target_trk},
                )
                _roi_mode = False
            else:
                # ROI path: pass raw gray — each _roi_detect preprocesses only its crop
                anchor_redetect = (total_frames % ANCHOR_INTERVAL == 0)
                target_redetect = args.full_roi or (total_frames % TARGET_INTERVAL == 0)
                tag_map = {
                    fish_id: _roi_detect(detector, gray_raw, fish_id, fish_trk,
                                         detect_params, args.tag_size, actual_w, actual_h,
                                         alpha=img_alpha, beta=img_beta, blur=img_blur),
                }
                if target_redetect:
                    tag_map[target_id] = _roi_detect(detector, gray_raw, target_id, target_trk,
                                                     detect_params, args.tag_size, actual_w, actual_h,
                                                     alpha=img_alpha, beta=img_beta, blur=img_blur)
                if anchor_redetect and anchor_trk.initialised:
                    tag_map[anchor_id] = _roi_detect(detector, gray_raw, anchor_id, anchor_trk,
                                                     detect_params, args.tag_size, actual_w, actual_h,
                                                     alpha=img_alpha, beta=img_beta, blur=img_blur)
                _roi_mode = True

            flags = 0

            if anchor_trk.initialised:
                flags |= FLAGS_ANCHOR

            if tag_map.get(fish_id) is not None and anchor_trk.initialised:
                fish_x, fish_y = world_frame_transform(
                    fish_trk.pose_t, anchor_trk.pose_t, anchor_trk.pose_R)
                fish_yaw = extract_yaw(fish_trk.pose_R, anchor_trk.pose_R)
                if tag_heading_flip:
                    fish_yaw = fish_yaw + math.pi if fish_yaw < 0 else fish_yaw - math.pi
                flags |= FLAGS_FISH

            if tag_map.get(target_id) is not None and anchor_trk.initialised:
                target_x, target_y = world_frame_transform(
                    target_trk.pose_t, anchor_trk.pose_t, anchor_trk.pose_R)
                flags |= FLAGS_TARGET
            elif not target_redetect and target_trk.initialised and anchor_trk.initialised:
                # Sparse skip — no detection this frame; carry last known position through
                flags |= FLAGS_TARGET

            send_pkt = (flags & FLAGS_FISH) and (
                (flags & FLAGS_TARGET) or args.no_target
            )
            if send_pkt:
                frame_seq += 1
                if plotter is not None:
                    plotter.push(capture_time, fish_x, fish_y, target_x, target_y)
                if udp_sock is not None:
                    pkt = pack_pose(
                        timestamp=capture_time,
                        frame_seq=frame_seq,
                        fish_x=fish_x, fish_y=fish_y, fish_yaw=fish_yaw,
                        target_x=target_x, target_y=target_y,
                        flags=flags,
                    )
                    udp_sock.sendto(pkt, pi_addr)
                    udp_sock.sendto(pkt, pi_addr)  # duplicate for WiFi loss resilience
                    last_pose_t = time.time()
                    last_pkt_sent_t = time.time()
                    _pkt_warn_printed = False

            # Warn if fish tag lost while recording (Pi will stop receiving pose packets)
            if recorder is not None and not _pkt_warn_printed:
                if (time.time() - last_pkt_sent_t) > _PKT_WARN_S:
                    print("[WARN] No pose packet sent to Pi for >1s — fish tag not visible!")
                    _pkt_warn_printed = True

            # Stop recording if Pi has gone silent (crashed/disconnected)
            if recorder is not None and (time.time() - last_pose_t) > _PI_DROPOUT_S:
                print(f"[Ctrl] Pi silent for >{_PI_DROPOUT_S:.0f}s — stopping recording.")
                recorder.stop()
                recorder = None

            # Handle Pi DONE
            if _ctrl_state.get('done_received'):
                _ctrl_state['done_received'] = False
                if recorder is not None:
                    print("[Ctrl] Pi DONE — stopping recording...")
                    recorder.stop()
                    recorder = None
                if _current_run_folder:
                    print(f"[Ctrl] Fetching run data: {_current_run_folder}")
                    _fetch_run(_current_run_folder)
                    _current_run_folder = None

            # Autostart: send START immediately when Pi signals READY
            if args.autostart and _ctrl_state.get('pi_ready') and recorder is None and args.pi_ip:
                _ctrl_state['pi_ready'] = False  # consume the READY event
                ts = time.time()
                ts_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
                _run_name = _ctrl_state.get('pi_name') or args.name or ''
                run_folder = f"{ts_str}_{_run_name}" if _run_name else ts_str
                _current_run_folder = run_folder
                session_dir = os.path.join(_REPO_ROOT, "deploy", "recordings", run_folder)
                os.makedirs(session_dir, exist_ok=True)
                recorder = AsyncVideoRecorder(
                    os.path.join(session_dir, "video.avi"),
                    fps=actual_fps, width=actual_w, height=actual_h,
                )
                if _ctrl_send_sock is not None:
                    _ctrl_send_sock.sendto(make_start_packet(ts), (args.pi_ip, CTRL_PORT))
                print(f"[Ctrl] Autostart: sent START → {args.pi_ip}  recording → {session_dir}/")

            if recorder is not None:
                recorder.push(frame, capture_time)

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                fps = frame_count / (now - fps_timer)
                dx_ = target_x - fish_x
                dy_ = target_y - fish_y
                vx_ =  math.cos(fish_yaw) * dx_ + math.sin(fish_yaw) * dy_
                vy_ = -math.sin(fish_yaw) * dx_ + math.cos(fish_yaw) * dy_
                bearing_deg = math.degrees(math.atan2(-vy_, -vx_))
                if not _roi_mode:
                    mode_str = "FULL"
                elif args.full_roi:
                    mode_str = "ROI"
                else:
                    mode_str = "ROI-s"
                # Show anchor/target as visible whenever tracker is healthy (miss_count==0)
                vis_tags = [k for k, v in tag_map.items() if v is not None]
                if anchor_trk.initialised and anchor_trk.miss_count == 0 and anchor_id not in vis_tags:
                    vis_tags.append(anchor_id)
                if not target_redetect and target_trk.initialised and target_trk.miss_count == 0 and target_id not in vis_tags:
                    vis_tags.append(target_id)
                print(f"fps={fps:.1f} [{mode_str}]  tags={sorted(vis_tags)}  "
                      f"fish=({fish_x:.3f},{fish_y:.3f})  yaw={math.degrees(fish_yaw):.0f}°  "
                      f"target=({target_x:.3f},{target_y:.3f})  "
                      f"bearing={bearing_deg:.0f}° [{_bearing_label(bearing_deg)}]")
                frame_count = 0
                fps_timer = now

            if not args.no_preview:
                # redraw at ~2 fps — waitKey every frame keeps window responsive
                if capture_time - last_preview_t >= PREVIEW_INTERVAL:
                    last_preview_t = capture_time
                    preview = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
                    display = cv2.resize(preview, (DISPLAY_W, DISPLAY_H), interpolation=cv2.INTER_NEAREST)
                    if args.record:
                        if recorder is not None:
                            cv2.circle(display, (20, 20), 8, (0, 0, 255), -1)
                            cv2.putText(display, "REC  (R=stop)", (33, 26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                        else:
                            cv2.putText(display, "R = start recording", (8, 26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
                    # Tune overlay — bottom-left
                    tune_str = f"a={img_alpha:.1f}  b={img_beta:.0f}  blur={img_blur}"
                    cv2.putText(display, tune_str, (8, DISPLAY_H - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
                    cv2.imshow("VisionStreamer (q=quit, r=rec, click window first)", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('r') and args.record and args.pi_ip:
                    if recorder is None and not _ctrl_state.get('pi_ready'):
                        print("[Ctrl] Pi not ready yet — wait for 'Pi READY' before pressing R.")
                    elif recorder is None:
                        # START a new run
                        _ctrl_state['pi_ready'] = False  # consume the READY event
                        ts = time.time()
                        ts_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
                        _run_name = _ctrl_state.get('pi_name') or args.name or ''
                        run_folder = f"{ts_str}_{_run_name}" if _run_name else ts_str
                        _current_run_folder = run_folder
                        session_dir = os.path.join(_REPO_ROOT, "deploy", "recordings", run_folder)
                        os.makedirs(session_dir, exist_ok=True)
                        recorder = AsyncVideoRecorder(
                            os.path.join(session_dir, "video.avi"),
                            fps=actual_fps, width=actual_w, height=actual_h,
                        )
                        if _ctrl_send_sock is not None:
                            _ctrl_send_sock.sendto(make_start_packet(ts), (args.pi_ip, CTRL_PORT))
                        print(f"[Ctrl] Sent START → {args.pi_ip}:{CTRL_PORT}  recording → {session_dir}/")
                    else:
                        # STOP current run
                        print("[Ctrl] Sending STOP to Pi...")
                        if _ctrl_send_sock is not None:
                            _ctrl_send_sock.sendto(CTRL_STOP, (args.pi_ip, CTRL_PORT))
                elif key == ord('r') and args.record and not args.pi_ip:
                    # Legacy: local-only recording toggle (no Pi connected)
                    if recorder is None:
                        recorder = _start_recording(args.name or '', actual_fps, actual_w, actual_h)
                    else:
                        print("[Recorder] Stopping...")
                        recorder.stop()
                        recorder = None

                # Live image tuning: A/Z=alpha  S/X=beta  D/C=blur
                elif key == ord('a'):
                    img_alpha = round(min(img_alpha + 0.1, 4.0), 1)
                    print(f"[Tune] alpha={img_alpha:.1f}  beta={img_beta:.0f}  blur={img_blur}")
                    _save_tune(img_alpha, img_beta, img_blur)
                elif key == ord('z'):
                    img_alpha = round(max(img_alpha - 0.1, 0.1), 1)
                    print(f"[Tune] alpha={img_alpha:.1f}  beta={img_beta:.0f}  blur={img_blur}")
                    _save_tune(img_alpha, img_beta, img_blur)
                elif key == ord('s'):
                    img_beta = min(img_beta + 5, 255)
                    print(f"[Tune] alpha={img_alpha:.1f}  beta={img_beta:.0f}  blur={img_blur}")
                    _save_tune(img_alpha, img_beta, img_blur)
                elif key == ord('x'):
                    img_beta = max(img_beta - 5, -255)
                    print(f"[Tune] alpha={img_alpha:.1f}  beta={img_beta:.0f}  blur={img_blur}")
                    _save_tune(img_alpha, img_beta, img_blur)
                elif key == ord('d'):
                    img_blur = img_blur + 2 if img_blur >= 3 else 3
                    print(f"[Tune] alpha={img_alpha:.1f}  beta={img_beta:.0f}  blur={img_blur}")
                    _save_tune(img_alpha, img_beta, img_blur)
                elif key == ord('c'):
                    img_blur = max(img_blur - 2, 0)
                    print(f"[Tune] alpha={img_alpha:.1f}  beta={img_beta:.0f}  blur={img_blur}")
                    _save_tune(img_alpha, img_beta, img_blur)

    except KeyboardInterrupt:
        print("\n[VisionStreamer] Stopped.")
    finally:
        if plotter is not None:
            plotter.stop()
        cap.release()
        if recorder is not None:
            print("[VisionStreamer] Flushing video buffer (may take a moment)...")
            recorder.stop()
        cv2.destroyAllWindows()
        _ctrl_stop.set()
        if _ctrl_send_sock is not None:
            _ctrl_send_sock.close()
        _sync_stop.set()
        if udp_sock is not None:
            udp_sock.close()


if __name__ == "__main__":
    main()
