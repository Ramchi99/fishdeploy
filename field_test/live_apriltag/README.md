# Live AprilTag Vision System

Real-time AprilTag detection on the **Windows laptop** via USB camera, with optional UDP pose streaming to the **Pi** for closed-loop fish control.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `live_apriltag.py` | Laptop (Windows) | Captures camera, detects tags, shows preview, sends UDP |
| `pi_pose_receiver.py` | Pi | Receives pose packets, exposes data to control code |

---

## Quick Start

**1. Start the receiver on the Pi first:**
```bash
python3 field_test/live_apriltag/pi_pose_receiver.py
```

**2. Start the detector on the laptop (from repo root):**
```bash
python field_test/live_apriltag/live_apriltag.py --pi_ip raspberrypi.local
```

Both scripts default to **port 5005**. The Pi must be running the receiver before the laptop starts sending or packets will be silently dropped (normal UDP behaviour — no error either side).

---

## live_apriltag.py — All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `0` | USB camera index. Try `1`, `2` etc. if the wrong camera opens |
| `--camera` | *(from settings)* | Calibration profile name in `vision_settings.json` (e.g. `simonPhone`, `overhead`) |
| `--tag_size` | `0.100` | Physical tag side length **in metres** (100 mm = 0.100) |
| `--preview_width` | `640` | Width of the live preview window in pixels |
| `--detect_width` | `0` | Width to resize to before detection. `0` = half of capture width. Lower = faster, less accurate |
| `--decimate` | `2.0` | AprilTag internal decimation factor. Higher = faster but misses smaller/distant tags |
| `--capture_height` | `1080` | Capture resolution height. Try `720` or `480` if fps is still low |
| `--fps` | `90` | Requested camera fps. Camera must support MJPEG at this rate |
| `--no_preview` | *(disabled)* | Disable the preview window — console output only, slightly faster |
| `--pi_ip` | *(disabled)* | IP address of the Pi. Omit to run detection-only with no UDP |
| `--port` | `5005` | UDP port. Must match `--port` on the Pi receiver |

**Examples:**
```bash
# Detection only, no streaming
python field_test/live_apriltag/live_apriltag.py --device 1

# Full pipeline to Pi
python field_test/live_apriltag/live_apriltag.py --device 1 --pi_ip raspberrypi.local

# Detection only, no preview window (fastest)
python field_test/live_apriltag/live_apriltag.py --device 1 --no_preview

# Different camera index, bigger preview
python field_test/live_apriltag/live_apriltag.py --device 1 --preview_width 960

# Faster detection (bigger tag, close range)
python field_test/live_apriltag/live_apriltag.py --device 1 --decimate 4.0 --pi_ip raspberrypi.local

# Maximum accuracy (slower — full resolution detection)
python field_test/live_apriltag/live_apriltag.py --device 1 --detect_width 1920 --decimate 1.0
```

**Windows notes:**
- Uses **MSMF** (Media Foundation) with MJPEG — required to reach high fps at 1080p over USB. `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` is set automatically at startup to prevent a known OpenCV hang on `VideoCapture` init.
- The startup line prints `Format: MJPG` when MJPEG is active. Anything else means the camera is in uncompressed mode and USB bandwidth will cap fps to ~5fps at 1080p.
- Install `pupil-apriltags` instead of `dt-apriltags` on Windows (`dt-apriltags` has no Windows wheels). The script tries both automatically.

**Hotkeys in the preview window:**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save undistorted snapshot as `snapshot_<timestamp>.jpg` |

---

## pi_pose_receiver.py — All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | `5005` | UDP port to listen on. Must match laptop sender |
| `--timeout` | `0.5` | Seconds without a packet before printing a "tag lost" warning |

---

## UDP Packet Format

Each detected tag produces one 108-byte packet. Format string: `!didddddddddddd` (network/big-endian byte order).

| Field | Type | Size | Description |
|-------|------|------|-------------|
| `timestamp` | float64 | 8 B | `time.monotonic()` on the laptop at detection time |
| `tag_id` | int32 | 4 B | AprilTag ID |
| `tx` | float64 | 8 B | X position, metres, camera frame |
| `ty` | float64 | 8 B | Y position, metres, camera frame |
| `tz` | float64 | 8 B | Z position (depth), metres, camera frame |
| `R[0..8]` | float64 × 9 | 72 B | 3×3 rotation matrix, row-major |
| `pose_err` | float64 | 8 B | Reprojection error (quality indicator — lower is better) |

**Total: 108 bytes per packet.**

If multiple tags are visible, one packet is sent per tag per frame.
No packet is sent for frames where no tag is detected — the Pi detects this via the `--timeout` gap.

---

## Camera Calibration

Intrinsics are read from `field_test/vision_settings.json` under the `"cameras"` key.
The active profile is selected with `--camera`. If omitted, the profile named by the `"camera"` key in the `"default"` section is used (currently `"simonPhone"`).

The calibration was produced by `field_test/pipeline/calibrate_camera.py` using a ChArUco board.
If you switch cameras or the lens is changed, re-run calibration and add the new profile to `vision_settings.json`.

---

## Adding Control Logic on the Pi

Edit `pi_pose_receiver.py` at the marked section:

```python
# YOUR CONTROL CODE GOES HERE
# t_vec = [x, y, z] position in metres (camera frame)
# R_mat = 3x3 rotation matrix
```

Typical use:
- `t_vec[2]` — depth (distance from camera), useful for speed control
- `t_vec[0]` — lateral offset, useful for heading correction
- `R_mat` — full 3D orientation if you need yaw/pitch/roll

Check packet freshness before acting:
```python
age = time.monotonic() - ts
if age > 0.2:
    # stale — tag may be occluded or detection dropped a frame
    pass
```

---

## Prerequisites

**Laptop (Windows):**
```
pip install opencv-python pupil-apriltags numpy
```

**Pi:**
```
# No extra packages needed — socket and struct are Python stdlib
```
