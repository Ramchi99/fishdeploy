# RL Deployment — Command Reference

**Laptop (vision + control):** `deploy/laptop/`  
**Pi (motor + sensors):** `deploy/pi/`  
**Run data:** `deploy/runs/{YYYYMMDD_HHMMSS}_{name}/`

---

## Critical: Always Use Stochastic Inference

**Never use `deterministic=True` with an SAC policy trained for ±180° navigation.**

SAC learns a Gaussian distribution over actions. For targets behind the fish (θ ≈ 180°), the optimal strategy is "turn left" OR "turn right" — both are equally valid. SAC represents this as a wide distribution centred near zero. The deterministic mean ≈ 0 means "no turn" → the fish swims forward and times out on every lateral or behind target.

Stochastic inference (`deterministic=False`) randomly picks a direction and commits to it. This is the correct deployment mode — the fish reaches ~70% of all targets including behind ones. `PolicyRunner.predict` already uses `deterministic=False`.

If cold eval numbers look low (~25–40%), re-run with `deterministic=False`. A deterministic evaluation underestimates the policy by 2–3×.

---

## Quick-start Workflows

### Workflow A — Open-loop (motor follows trajectory, no policy)

**Laptop:**
```bash
python3 deploy/laptop/vision_streamer.py \
    --pi_ip raspberrypi.local \
    --env-config deploy/configs/cpg_24d.json \
    --name sine_1 --record
```

**Pi:**
```bash
python3 deploy/pi/open_loop_main.py \
    --name sine_1 \
    --env-config deploy/configs/cpg_24d.json \
    --trajectory sine --freq 1.5 --amp 0.3 \
    --record
```

Press **R** in the laptop preview window to start. Press **R** again to stop.

> **No target tag?** Add `--no-target` to the Pi command and `--no-target` to the laptop command. The motor follows its trajectory regardless of vision; you only need the fish tag visible for pose logging.

> **Duration:** Omit `--duration` to run until you press R. Add `--duration 30` to auto-stop after 30 s.

---

### Workflow B — Shadow (open-loop trajectory, policy runs in parallel for logging)

**Laptop:**
```bash
python3 deploy/laptop/vision_streamer.py \
    --pi_ip raspberrypi.local \
    --env-config deploy/configs/cpg_24d.json \
    --name shadow_zero_1 --record
```

**Pi:**
```bash
python3 deploy/pi/open_loop_main.py \
    --name shadow_zero_1 \
    --env-config deploy/configs/cpg_24d.json \
    --trajectory zero --duration 30 \
    --shadow --model policies/sofi_cpg_rl_policy.zip \
    --record
```

Press **R** to start. Press **R** to stop.

> **No target tag in frame:** If you are running shadow mode without a target AprilTag, add `--no-target` to both the Pi and laptop commands.

---

### Workflow C — Noise floor (motor silent, record sensor noise)

Same as Workflow B with `--trajectory zero` and no `--shadow`. Produces `processed/noise_config.json` for RL training.

**Pi:**
```bash
python3 deploy/pi/open_loop_main.py \
    --name noise_floor_1 \
    --env-config deploy/configs/cpg_24d.json \
    --trajectory zero --duration 30 --record
```

---

### Workflow D — RL deployment (policy drives motor)

#### The deployed policy (24-dim observation, sensor noise + physics domain randomisation, targets to ±120°)

**Laptop:**
```bash
python3 deploy/laptop/vision_streamer.py \
    --pi_ip raspberrypi.local \
    --env-config deploy/configs/cpg_24d.json \
    --name phase2_test_1 --record
```

**Pi:**
```bash
python3 deploy/pi/deploy_main.py \
    --name phase2_test_1 \
    --env-config deploy/configs/cpg_24d.json \
    --model policies/sofi_cpg_rl_policy.zip \
    --duration 60 --record
```

> This is the policy behind the 43 runs of 2026-05-20 in `deploy/runs/` (static, disturbance, moving-target and figure-eight targets). Its md5 `7bbe08b8` is recorded in each run's `run_config.json`.


---

## Session Startup Checklist

1. **Preflight:** `python3 deploy/preflight.py` (from repo root on laptop)
2. **Deploy code:** `./scripts/deploy_to_pi.sh`
3. **Start laptop script** — wait for "Camera: WxH @ fps"
4. **Start Pi script** — wait for "Gyro calibrated" + "Waiting for first UDP pose packet"
5. **Vision online** — Pi prints "Sent READY" and laptop shows "Pi READY — press R"
6. **Press R** — run begins. Both sides log to matching `{YYYYMMDD_HHMMSS}_{name}` folder.
7. **Press R again** (or run expires naturally) — Pi sends DONE, **loops back to READY** for the next run (same config). Motor is automatically returned to zero before the next run. Laptop auto-fetches data and runs `post_run.py`.
8. **To exit entirely:** Ctrl+C on the Pi. To change the run name, trajectory, or policy, you must restart the Pi script with new arguments.

---

## Pi Setup

### First-time setup (once per Pi, survives re-flash)

**1. Install Python dependencies**

SSH into the Pi and install the packages the deploy scripts need:

```bash
python3 -m venv ~/fishdeploy/venv && ~/fishdeploy/venv/bin/pip install -r ~/fishdeploy/requirements-pi.txt
```

`dynamixel-sdk` comes from pip (listed in `requirements-pi.txt`); the thin wrappers around it live in `hardware/dxlSetup/`.

**2. Fix the USB-serial latency timer (critical for motor reading quality)**

By default the USB-serial adapter (`/dev/ttyUSB0`) buffers incoming bytes for up to **16 ms** before delivering them to Python. Every Dynamixel read is a round-trip, so without this fix the motor-runner thread runs at only ~16 Hz — far below the 50 Hz control loop — causing ~67% of motor encoder readings to be stale fallbacks. This was confirmed across all run types.

Set the timer to 1 ms permanently via a udev rule:

```bash
# On the Pi
echo 'SUBSYSTEM=="usb-serial", ATTR{latency_timer}="1"' \
    | sudo tee /etc/udev/rules.d/99-dynamixel-latency.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify it took effect (should print `1`):

```bash
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

If the rule does not survive reboot on your Pi OS version, add the one-liner to `/etc/rc.local` as a fallback:

```bash
echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

**3. Check motor port permissions**

The Pi user (`sofi`) must be able to open `/dev/ttyUSB0` without sudo:

```bash
ls -l /dev/ttyUSB0           # should show group=dialout
sudo usermod -aG dialout sofi
# log out and back in for the group to take effect
```

**4. Verify hardware wiring**

- Motor USB adapter plugged into Pi USB-A port → `/dev/ttyUSB0`
- IMU (BNO055) on I²C SDA/SCL pins (board.SDA / board.SCL)
- Pi reachable at `sofi@raspberrypi.local` from laptop

**5. Verify IMU axis alignment (run once after any physical re-mounting of the IMU)**

The BNO055 gyro axes must match the training convention. `imu_reader.py` applies the **BulkheadFixed software remap** (`conf=0x12, sign=0x06`) in software so that:

| Output | Formula | Observation index | Meaning |
|--------|---------|-------------------|---------|
| `wx` | `+physical_Z` | `obs[3]` | roll rate |
| `wy` | `−physical_X` | `obs[4]` | pitch rate |
| `wz` | `+physical_Y` | `obs[5]` | **yaw rate** (most important) |

`wz` (obs[5]) must be **positive when the fish spins counter-clockwise** (nose left) viewed from above — this is how the training environment (MuJoCo `headRollZ` joint) and the field-to-sim EKF (`Meas_GyrZ`) both define it.

To verify, run the live diagnostic on the Pi **with the motor unpowered**:

```bash
python3 sysid/imu_axis_check.py
```

Slowly rotate the fish in each direction and confirm:
- Nose left (CCW from above) → `wz` goes **positive**
- Nose right (CW from above) → `wz` goes **negative**
- Nose up → `wy` spikes
- Roll onto right side → `wx` spikes

The right-hand column in the output (hardware BNO055 remap) must be **identical** to the left-hand column (software remap in `imu_reader.py`). If they differ, the IMU has been re-mounted in a different orientation and `imu_reader.py` must be updated accordingly.

---

### Per-session Pi steps

These steps apply every time you run an experiment (open-loop, shadow, or RL).

**Step 1 — Deploy latest code from laptop**

```bash
# On laptop
./scripts/deploy_to_pi.sh
```

**Step 2 — SSH into the Pi**

```bash
ssh sofi@raspberrypi.local
cd ~/fishdeploy
```

**Step 3 — Start the appropriate Pi script**

Choose one depending on the mode (see Quick-start Workflows above):

```bash
# Open-loop (runs until R pressed — add --duration 30 to auto-stop)
python3 deploy/pi/open_loop_main.py --name <name> --env-config deploy/configs/cpg_24d.json \
    --trajectory sine --freq 1.5 --amp 0.3 --record

# Open-loop without target tag in frame
python3 deploy/pi/open_loop_main.py --name <name> --env-config deploy/configs/cpg_24d.json \
    --trajectory sine --freq 1.5 --amp 0.3 --no-target --record

# Shadow (open-loop + policy logged in parallel)
python3 deploy/pi/open_loop_main.py --name <name> --env-config deploy/configs/cpg_24d.json \
    --trajectory sine --freq 1.5 --amp 0.3 \
    --shadow --model policies/sofi_cpg_rl_policy.zip --record

# RL (policy drives the motor)
python3 deploy/pi/deploy_main.py --name <name> --env-config deploy/configs/cpg_24d.json \
    --model policies/sofi_cpg_rl_policy.zip --duration 60 --record
```

**What you will see on the Pi terminal:**

```
Dynamixel has been successfully connected
[IMU] Waiting for gyro calibration...
[IMU] Calibration: (0, 3, 0, 0)       ← gyro column = 3 means ready
[IMU] Gyro calibrated in 4.2s.
[OpenLoop] Waiting for first valid UDP pose packet (timeout 60s)...
[OpenLoop] Sent READY:sine_1 — waiting for START (120s timeout)...
```

> The IMU calibration step requires the Pi (and fish) to be **still** for a few seconds. Do not move the fish during this phase or gyro cal will take much longer.

**Step 4 — Start laptop vision_streamer, then press R**

The laptop streamer must already be running and seeing both the fish and target AprilTags before the Pi reaches "Waiting for first UDP pose packet". Once both sides are ready the laptop shows:

```
Pi READY — press R to start
```

Press **R** in the preview window. The run begins simultaneously on both sides.

**Step 5 — After the run**

The Pi prints `Run done: deploy/data/...`, then automatically:
1. Returns the motor to zero and re-zeroes the encoder reference
2. Re-measures the laptop↔Pi clock offset
3. Sends `READY` and loops back to "Waiting for START" for the next run (same config)

On the laptop, `fetch_data.sh` runs automatically (triggered by DONE), rsync's the run folder, and then calls `post_run.py` which writes `processed/summary.png` and `processed/diagnostics.png`.

To fetch data manually (e.g. after a hard Ctrl+C):

```bash
./scripts/fetch_data.sh rl <run_folder_name>
```

**To change the run name, trajectory, frequency, or policy:** Ctrl+C on the Pi (hard exit), then restart with new arguments. The timestamp in the folder name (`{YYYYMMDD_HHMMSS}_{name}`) uniquely identifies each individual run even within the same session.

---

## Flag Reference

### `vision_streamer.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--pi_ip` | None | Pi IP. Omit for camera-only preview. |
| `--env-config` | required | Path to env_config JSON |
| `--tag_size` | 0.100 | AprilTag size [m] |
| `--camera` | None | Camera profile from `configs/camera_settings.json` |
| `--device` | 0 | USB camera device index |
| `--no_preview` | False | Headless mode (max fps, no window) |
| `--record` | False | Enable R-key controlled recording |
| `--autostart` | False | Send START immediately on Pi READY (use with `--no_preview`) |
| `--name` | None | Session name (must match Pi `--name`) |
| `--fps` | 90 | Requested camera fps |
| `--live_plot` | False | Live distance/velocity diagnostic plots |
| `--no-target` | False | Accept fish-only packets — do not require target tag in frame. Use with open-loop/shadow when no target AprilTag is deployed. |
| `--no-roi` | False | Disable ROI tracking — detect all tags full-frame every frame (pre-ROI behaviour, ~52 fps) |
| `--full-roi` | False | Detect target every frame in ROI mode. **Omit for sparse target mode (every 10th frame)** — use this when the target tag does not move. |
| `--alpha` | 1.0 | Contrast multiplier before tag detection (try 0.4–1.8) |
| `--beta` | 0.0 | Brightness offset before tag detection (try -50 to +50) |
| `--blur` | 0 | Gaussian blur kernel size (0=off; must be odd, e.g. 3, 5, 7) |

### `deploy_main.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | required | Run name (match laptop `--name`) |
| `--env-config` | required | Path to env_config JSON |
| `--model` | required | SAC policy .zip path |
| `--duration` | 60.0 | Run duration [s] per START. Fish tag must remain visible; run aborts if vision is lost >1 s. |
| `--record` | False | Enable run.csv logging |
| `--finetune` | False | Enable replay_buffer.csv logging |

### `open_loop_main.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--name` | required | Run name |
| `--env-config` | required | Path to env_config JSON |
| `--trajectory` | required | `zero`, `sine`, `sine_glide`, or `sine_hold` |
| `--freq` | 1.5 | Oscillation frequency [Hz] |
| `--amp` | 0.3 | Amplitude [rad] |
| `--offset` | 0.0 | DC offset [rad] |
| `--sine_duration` | 10.0 | Active sine phase duration for `sine_glide` / `sine_hold` [s] |
| `--hold_offset` | 0.0 | Tail position to hold after sine phase (`sine_hold` only) [rad] |
| `--duration` | None | Total run duration [s]. **Omit to run until R is pressed.** |
| `--record` | False | Log run.csv |
| `--no-target` | False | Accept fish-only packets — no target tag required in frame. Safe for open-loop: vision loss is warned but never aborts. |
| `--shadow` | False | Run policy in parallel (logs only, motor = trajectory) |
| `--model` | None | SAC .zip path (required with `--shadow`) |

### `post_run.py`

| Flag | Default | Description |
|------|---------|-------------|
| `run_dir` | required | Path to `deploy/runs/{name}` |
| `--render` | False | Generate `processed/replay.mp4` (~2 min, needs `raw/video.avi`) |
| `--no-noise` | False | Skip `noise_config.json` / `noise_report.md` |
| `--all` | False | All outputs including render |

---

## Vision System Performance

### Detection architecture

The laptop vision pipeline (`vision_streamer.py`, `obs_verifier.py`) uses:

```
Camera frame (1920×1080 Windows / 1280×720 Linux)
  │
  ├─ convertScaleAbs(alpha, beta)   ← contrast/brightness tuned by tune_vision.py
  ├─ GaussianBlur(kernel)           ← optional pre-detection smoothing
  │
  └─ AprilTag detector (dt-apriltags / pupil-apriltags)
       quad_decimate = 4.0          ← initial quad search at 1/4 resolution (fast)
       refine_edges  = 1            ← corner positions re-fit at FULL resolution (accurate)
       nthreads      = 6            ← parallel quad search across image strips
```

**What `quad_decimate=4.0` actually does:**
The tag search grid (finding quad corners) runs on a 480×270 or 320×180 image — cheap.
Once a candidate tag is found, `refine_edges=1` re-localises its corners by gradient
analysis on the original full-resolution frame. Pose estimation then uses those
full-resolution corners, so 3D accuracy is not degraded by the decimation.
The only real cost of high decimation is missing tags that are very small in the frame;
a 100 mm tag at ≤1.5 m occupies enough pixels at 1/4 res that this is not a concern.

### ROI tracking

After the first full-frame detection, each tag is tracked via a padded ROI crop
(1.5× the tag bounding box in each direction). The detector only sees that small
crop each frame instead of the full image:

```
ROI-s mode (default — sparse target):
  fish tag   → detected every frame via ROI crop
  target tag → detected every TARGET_INTERVAL=10 frames via ROI crop
             → last known world position passed through on skipped frames (FLAGS_TARGET stays set)
             → falls back to full-frame if 3 consecutive detection intervals are missed
  anchor tag → detected every ANCHOR_INTERVAL=30 frames via ROI crop
             → stored pose used in between (anchor is fixed — pose doesn't change)
             → falls back to full-frame if miss_count ≥ MISS_THRESH=3

ROI mode (--full-roi — detect target every frame):
  fish tag   → detected every frame via ROI crop
  target tag → detected every frame via ROI crop
  anchor tag → detected every ANCHOR_INTERVAL=30 frames via ROI crop

Full-frame fallback (triggered automatically when any tag is lost):
  all 3 tags → scanned across the full frame simultaneously
```

The mode string in the status line reflects the current detection mode:
`[ROI-s]` = sparse target (default), `[ROI]` = full target (--full-roi), `[FULL]` = full-frame fallback.

### Measured frame rates (Windows, MSMF + MJPEG, 1920×1080)

| Mode | fps observed | Notes |
|------|-------------|-------|
| Full-frame (`--no-roi`) | ~52 fps | All 3 tags full-frame every frame |
| ROI-s (default, sparse target) | ~70–75 fps | Fish ROI every frame; target ROI every 10 frames; anchor ROI every 30 frames |
| ROI (`--full-roi`) | ~65–71 fps | Fish + target ROI every frame; anchor ROI every 30 frames |
| Camera ceiling (no detection) | **90 fps** | Hard hardware limit — requesting 120 fps or dropping resolution both return the same 90 fps; this camera is capped at 90 fps regardless |

On **Linux / WSL2** the V4L2 backend caps at 30 fps regardless of mode
(USB forwarding cannot sustain MJPEG at high data rates).

### Tuning for higher fps

| Change | Expected gain | Trade-off |
|--------|---------------|-----------|
| Increase `quad_decimate` from 4.0 → 8.0 | +10–20% | Tags must be larger in frame; small/distant tags may be missed |
| Set `refine_edges=0` | +5–10% | Pose accuracy degrades slightly |
| Reduce input resolution (e.g. 1280×720 on Windows) | +15–25% | Fewer pixels; pose accuracy and tag detection range both decrease |
| Increase `nthreads` beyond 6 | marginal | Already above CPU core count on most laptops |
| Async detection thread | could reach camera ceiling | Complex; not implemented |

### Hardware vision latency — preliminary results (2026-05-20, needs thorough analysis)

Measured across ~50 runs with `quad_decimate=2`, Windows native camera, Pi over WiFi near water.
End-of-run vision-loss tail trimmed (only steps before last fresh pose counted).

**Actual camera fps on hardware (from `frame_seq` in run.csv):** 32–42 fps — NOT the 70 fps
the streamer reports. Likely limited by older PC + Windows processing overhead + decimate=2.
Training used `camera_fps=30` — hardware matches well.

| Metric | Training DR | Hardware (trimmed) |
|--------|-------------|-------------------|
| Camera fps | 30 fps | 32–42 fps ✓ |
| Vision delay (median) | 10–50 ms | 32 ms ✓ |
| Control steps per frame | ~1.7 | 1.2–1.6 ✓ |
| Steps > 50 ms (stale) | ~0% | 27% ⚠ |
| Steps > 500 ms (lost) | 0% | 3% ⚠ |

**Why the tail is elevated:** near-surface water absorbs 2.4 GHz RF → intermittent WiFi
dropouts cause 200–500 ms blackouts. The frame buffer age signal (`obs[9–17]`) partially
compensates — policy knows data is stale — but it was never trained on gaps this large.

**TODO (thorough analysis):**
- Separate WiFi drops (frame_seq gaps > expected) from tag-out-of-frame gaps
- Per-run breakdown to identify which runs had hardware vs WiFi vs setup issues
- Check if latency tail correlates with failed approaches (especially from behind)
- Consider extending training DR to latency_max=300ms to cover the p95

### Camera stress test

To measure the raw frame-delivery rate with zero processing overhead:

```bash
python deploy/laptop/camera_stress_test.py --device 1
# Windows default: 1920×1080 @ 90fps MJPEG
# Linux default:   1280×720  @ 30fps V4L2
```

This opens the camera, reads frames as fast as possible for 10 s, and prints per-second
and overall fps. Use this to confirm what the camera hardware actually delivers before
adding any detection overhead.

---

## Trajectory Reference

### `zero` — motor stationary
```bash
--trajectory zero --duration 30
```
Use for: noise floor characterization, shadow policy evaluation at rest.

### `sine` — sinusoidal oscillation
```bash
--trajectory sine --freq 1.5 --amp 0.3 --duration 30
# freq range: 0.5–3.0 Hz | amp range: 0.1–0.65 rad
--trajectory sine --freq 2.0 --amp 0.5 --offset 0.1 --duration 60
```

### `sine_glide` — sine then motor cut to zero
```bash
--trajectory sine_glide --freq 1.5 --amp 0.3 --sine_duration 10 --duration 20
# Active sine for 10s, then motor=0 for remaining 10s
```

### `sine_hold` — sine startup then hold at offset
```bash
--trajectory sine_hold --freq 1.5 --amp 0.3 --sine_duration 5 --hold_offset 0.4 --duration 20
# Sine for 5s (startup), then motor holds at +0.4 rad for remaining 15s
# hold_offset=0.0 is equivalent to glide (motor holds at centre)
```
Use for: testing steady-state turning response; system ID at a fixed tail deflection.

---

## Multi-run Sessions

The Pi script runs a **multi-run loop**: after each run ends (R pressed or duration elapsed), the Pi automatically:
1. Sends DONE to the laptop
2. Returns the motor to zero and re-zeroes the encoder reference
3. Re-measures the clock offset
4. Sends READY and waits for the next START

This means you can do several runs back-to-back without restarting anything. Each run gets its own timestamped folder (`{YYYYMMDD_HHMMSS}_{name}/`).

**The config is baked in at startup.** The `--name`, `--trajectory`, `--freq`, `--amp`, `--model`, and all other args are fixed for the entire session. To change any of them: **Ctrl+C on the Pi** → restart with new arguments.

### Common scenarios

| Scenario | Action |
|----------|--------|
| Same trajectory, new run | Press R on laptop — Pi loops back automatically |
| Change frequency or amplitude | Ctrl+C Pi → restart with new `--freq` / `--amp` |
| Change policy model | Ctrl+C Pi → restart with new `--model` |
| Give this session a different name | Ctrl+C Pi → restart with new `--name` |
| Laptop script needs restart | Just restart it — Pi will detect the new laptop IP from the next pose packet |

### Clock offset reliability

The clock offset (laptop − Pi time) is measured at the start of each run using an NTP-style 20-ping handshake. Within a session the offset is stable to **±10–25 ms** — well under one 20 ms control step. Drift is ~75 ms/hour (irrelevant since each run re-measures). If all pings time out (laptop sync server not ready), the script retries automatically up to 3 times before falling back to 0.0 ms (logged in `run_config.json`).

---

## How to Add a New Trajectory

1. Add a class to `deploy/pi/trajectory.py`:

```python
class ChirpTrajectory:
    name = "chirp"

    def __init__(self, amp: float, f0: float, f1: float, duration: float):
        self.amp = amp
        self.f0 = f0
        self.f1 = f1
        self.duration = duration

    def step(self, t: float) -> float:
        f = self.f0 + (self.f1 - self.f0) * min(t, self.duration) / self.duration
        return self.amp * math.sin(2.0 * math.pi * f * t)

    def cpg_phase(self, t: float) -> float:
        f = self.f0 + (self.f1 - self.f0) * min(t, self.duration) / self.duration
        return (2.0 * math.pi * f * t) % (2.0 * math.pi)
```

2. Register in `make_trajectory()` in `trajectory.py`:

```python
if name == "chirp":
    return ChirpTrajectory(
        amp=float(kwargs["amp"]),
        f0=float(kwargs.get("f0", 0.5)),
        f1=float(kwargs.get("f1", 3.0)),
        duration=float(kwargs.get("duration", 30.0)),
    )
```

3. Add `--f0`, `--f1` arguments to `open_loop_main.py`'s argparse. Pass them via `traj_params`.

4. Use it:
```bash
python3 deploy/pi/open_loop_main.py --trajectory chirp --amp 0.3 --f0 0.5 --f1 3.0 ...
```

---

## Directory Layout

```
deploy/
  pi/                ← runs on Raspberry Pi 5
  laptop/            ← runs on laptop (WSL/Ubuntu)
  shared/            ← protocol.py (shared constants + packet helpers)
  configs/           ← env_config JSONs (cpg_24d.json, camera_settings.json, image_tune.json)
  policy/            ← SAC policy .zip files (gitignored)
  runs/              ← all run data (gitignored)
  preflight.py       ← hardware preflight check
  tests/             ← pytest suite
```

---

## Data Layout

Every run lives in `deploy/runs/{YYYYMMDD_HHMMSS}_{name}/`:

```
raw/
  run.csv                  ← Pi sensor data at 50 Hz (always)
  run_config.json          ← mode, trajectory, policy hash, clock offset (always)
  video.avi                ← MJPEG camera recording (if --record used)
  frame_timestamps.csv     ← per-frame laptop_time (if --record used)
  replay_buffer.csv        ← SAR tuples (if --finetune used)

processed/
  summary.png              ← 4-panel mode-specific overview (always generated)
  diagnostics.png          ← 6-panel system health (always generated)
  noise_config.json        ← obs noise std for RL training (zero-trajectory runs only)
  noise_report.md          ← human-readable noise table (zero-trajectory runs only)
  replay.mp4               ← annotated video (post_run.py --render only)
```

`raw/` is written only during the run. `processed/` is re-generable: delete and re-run `post_run.py`.

---

## Fine-tuning Modes

| Flag | Effect |
|------|--------|
| (none) | Motor moves, obs+actions+reward logged to `run.csv` |
| `--finetune` | Also logs SAR tuples to `replay_buffer.csv` for offline SAC update |
| `--shadow` | Policy runs but motor follows trajectory; policy actions logged for comparison |

---

## IMU Axis Reference

This section documents the full coordinate-frame chain from MuJoCo training through to the physical BNO055 on the fish. **All three systems must agree — any mismatch silently corrupts the yaw input to the policy.**

### The chain

```
MuJoCo sim (training)
  data.qvel[3:6]  →  headRollX, headRollY, headRollZ rates
  obs[3] = wx = roll   (around swim / MuJoCo-X axis)
  obs[4] = wy = pitch  (around lateral / MuJoCo-Y axis)
  obs[5] = wz = YAW    (around vertical / MuJoCo-Z axis, positive = CCW from above)

field-to-sim EKF  (run_field2sim.py, used for system identification)
  z_dict['imu'] = [0.0, 0.0, Meas_GyrZ]
  EKF slot [2] = Meas_GyrZ  →  drives EKF yaw-rate state  →  same as obs[5] in sim
  Meas_GyrZ = field_test GyrZ = +physical_Y  (confirmed on hardware)

Physical BNO055 (imu_reader.py, after software remap)
  raw gyro tuple from adafruit: (physical_X, physical_Y, physical_Z)
  wx =  raw_z  (+physical_Z)  →  obs[3]
  wy = −raw_x  (−physical_X)  →  obs[4]
  wz = +raw_y  (+physical_Y)  →  obs[5] = YAW  ✓  (CCW-positive confirmed on hardware)
```

### Why the remap is applied in software

`field_test/unified_test.py` writes the BulkheadFixed remap to BNO055 registers 0x41/0x42 (requires `CONFIG_MODE`, which resets the fusion state). The RL deploy keeps the sensor in `IMUPLUS_MODE` throughout and applies the identical transformation in Python instead — no register writes needed, no mode resets.

### Sign check

| Physical motion | `wz` after remap | Correct? |
|-----------------|-----------------|----------|
| Nose left (CCW from above) | **positive** | ✓ matches MuJoCo |
| Nose right (CW from above) | **negative** | ✓ matches MuJoCo |

**Confirmed on hardware** (`imu_axis_check.py`, 2026-05-15): rotating fish nose-left (CCW from above) → `wz = +raw_y > 0` ✓. The previous run-data analysis was incorrect — the sign is `+raw_y`, not `−raw_y`.

### What was wrong before this fix

Before the BulkheadFixed remap was applied in `imu_reader.py`, the raw BNO055 physical axes were passed directly to the policy:
- `obs[4]` = `physical_Y` = **yaw** (wrong index, and wrong sign for left turn)
- `obs[5]` = `physical_Z` = **not yaw** (policy read this as yaw)

Across all recorded runs this caused obs[5] to receive irrelevant data while obs[4] carried yaw with the wrong sign — the policy was effectively flying blind on its most important navigational input.

### If the IMU is ever re-mounted

Run `python3 sysid/imu_axis_check.py` on the Pi. Rotate the fish in pure yaw (CCW) and confirm `wz > 0`. If `wz < 0`, negate the `raw_y` term. If a different axis spikes, update the three lines in `imu_reader._run()` accordingly and re-run the check.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Pi prints "TIMEOUT waiting for vision" | Laptop streamer not running, or `--pi_ip` wrong | Start vision_streamer first; verify IP |
| Laptop shows no "Pi READY" | CTRL_PORT 7979 blocked | Check firewall; verify Pi reached clock sync |
| R key press does nothing | No `--record` flag, or no `--pi_ip` | Add `--record` to laptop command; ensure `--pi_ip` set |
| START timeout (120s) | Pi waiting but laptop R not pressed | Press R in laptop preview, or use `--autostart` |
| run.csv empty or 0 rows | Vision lost immediately (RL mode aborts on 1 s without fish tag) | Verify tags visible; check `--tag_size`; use open-loop with `--no-target` if no target tag deployed |
| Open-loop run prints "Vision lost" but continues | Normal — in open-loop mode the motor follows its trajectory regardless; vision loss is only a warning | Ignore, or ensure the fish tag is visible |
| post_run.py error "run_config.json not found" | Wrong run_dir or run not fetched yet | Run `./scripts/fetch_data.sh rl 20260508_143022_pool_test_1` |
| Clock offset returns 0.0 | All pings timed out — laptop sync server not yet running | Start `vision_streamer.py` first; sync auto-retries up to 3 times |
| Clock offset > 100ms between runs | Expected if NTP corrected the laptop or Pi clock | Normal — each run re-measures; the offset in `run_config.json` is authoritative |
| IMU gyro timeout | BNO055 not stable | Keep Pi stationary ≥5s after power-on |
| Motor encoder reads as 0 every 2–3 steps | FTDI latency timer is 16ms (default) — motor thread runs at ~16 Hz instead of 50 Hz | Follow the latency-timer fix in **Pi Setup → First-time setup** above |
| Motor encoder frozen (same value repeating) | USB-serial communication error — motor thread falling back to cached state | Check USB cable; verify `/dev/ttyUSB0` exists; reconnect motor adapter |
| `wz` spikes on wrong rotation (e.g. pitching instead of yawing) | IMU re-mounted in different orientation | Re-run `imu_axis_check.py`, update the three remap lines in `imu_reader._run()` |
| `wz` has opposite sign during left turn | Remap sign error | Negate the `raw_y` term in `imu_reader._run()` |
| Left/right columns differ in `imu_axis_check.py` | Software remap doesn't match hardware remap | `imu_reader.py` remap is wrong; cross-check with **IMU Axis Reference** section |
