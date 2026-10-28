# deploy: the policy on the fish

Two machines. The **laptop** runs the overhead camera, detects the AprilTags of the fish, the anchor and
the target, and streams their poses to the Pi over UDP at the camera rate (`laptop/vision_streamer.py`).
The **Raspberry Pi 5 inside the fish** reads the motor encoder and the IMU, assembles the 24-D observation,
runs the SAC policy at 50 Hz, drives the motor through the CPG and logs every step (`pi/deploy_main.py`).
The two clocks are synchronised at the start of every run; the laptop's START timestamp names the run
folder on both sides, `data/deploy_runs/<YYYYMMDD_HHMMSS>_<name>/raw/`, so the Pi's `run.csv` and the
laptop's `video.avi` end up in the same place once `scripts/fetch_data.sh` has pulled the Pi's half.

```
laptop                                        Pi (in the fish)
──────                                        ────────────────
camera → AprilTags → poses ──UDP 50-90 Hz──►  obs (24-D) → SAC policy → CPG → Dynamixel
video.avi + frame_timestamps.csv              run.csv + run_config.json + summary.png
          ◄──── READY / DONE, clock sync ────
fetch_data.sh ◄──────────────────────────────  (rsync after DONE) → post_run.py
```

## Always stochastic inference

Never evaluate or deploy the policy with `deterministic=True`. For a target behind the fish, turning left
and turning right are equally good, and SAC represents that as a wide distribution centred on "no turn".
The deterministic mean swims straight and times out; sampling picks a side and commits.
`PolicyRunner.predict` samples. A deterministic evaluation underestimates the policy by two to three times.

## The three workflows

Every command runs from the repository root, on the machine named. The laptop command is the same for
all three; what changes is the Pi script.

**Laptop, always first:**
```bash
python3 deploy/laptop/vision_streamer.py --pi_ip raspberrypi.local --env-config deploy/configs/cpg_24d.json --record
```
Wait for `Camera: WxH @ fps`. When the Pi is ready the window says `Pi READY — press R`; **R** starts the
run, **R** again stops it (or `--duration` on the Pi ends it). After DONE the laptop pulls the run and
post-processes it. Add `--no-target` on both sides if no target tag is in the pool.

**1. Deployment (the policy drives the fish):**
```bash
python3 deploy/pi/deploy_main.py --name <name> --env-config deploy/configs/cpg_24d.json \
    --model policies/sofi_cpg_rl_policy.zip --duration 60 --record
```
This is the policy behind the 43 runs of 2026-05-20 in `data/02_deploy_20260520/`; its md5 `7bbe08b8` is
in every run's `run_config.json`. The run aborts if the fish tag is lost for more than 1 s.

**2. Open-loop (the motor follows a trajectory, no policy):** motor and tracking checks, and the
reference stroke of a system-identification measurement.
```bash
python3 deploy/pi/open_loop_main.py --name <name> --env-config deploy/configs/cpg_24d.json \
    --trajectory sine --freq 1.5 --amp 0.3 --record            # omit --duration: runs until R
```
Add `--shadow --model policies/sofi_cpg_rl_policy.zip` to run the policy in parallel and log its actions
without letting it drive (a dry run of a new policy).

**3. Noise floor (motor silent, fish still):** the sensor-noise measurement behind the training noise
model. `post_run.py` writes `processed/noise_config.json` and `noise_report.md` for zero-trajectory runs.
```bash
python3 deploy/pi/open_loop_main.py --name noise_floor --env-config deploy/configs/cpg_24d.json \
    --trajectory zero --duration 30 --record
```
The measurement of 2026-05-07 that set `sensor_model.py` in fishsim is `data/03_sysid/sensor_noise_20260507/`.

### Session checklist

1. `python3 deploy/preflight.py` on the laptop (network, ports, camera, policy hash, Pi packages).
2. `./scripts/deploy_to_pi.sh` (rsync of the code, the policy and the configs; `PI_HOST=<ip>` overrides).
3. Start the laptop streamer, wait for the camera line.
4. `ssh sofi@raspberrypi.local`, `cd ~/fishdeploy`, start the Pi script, wait for `Gyro calibrated` (keep the
   fish still) and `Sent READY`.
5. Press **R**. Press **R** again or let the duration run out. The Pi returns the motor to zero, re-measures
   the clock offset and sends READY again: the next run needs only another **R**.
6. To change name, trajectory or policy: Ctrl+C on the Pi and restart it with the new arguments.
7. If the laptop did not fetch a run (hard Ctrl+C): `./scripts/fetch_data.sh rl <run_folder>`.

## Pi setup (once)

```bash
./scripts/deploy_to_pi.sh                                       # laptop
ssh sofi@raspberrypi.local
python3 -m venv ~/fishdeploy/venv && ~/fishdeploy/venv/bin/pip install -r ~/fishdeploy/requirements-pi.txt
sudo usermod -aG dialout sofi                                   # /dev/ttyUSB0 without sudo; log out and in
```

**USB-serial latency timer.** The adapter buffers bytes for 16 ms by default; every Dynamixel read is a
round trip, so the motor thread would run at 16 Hz instead of 50 Hz and two thirds of the encoder readings
would be stale. Set it to 1 ms permanently:
```bash
echo 'SUBSYSTEM=="usb-serial", ATTR{latency_timer}="1"' | sudo tee /etc/udev/rules.d/99-dynamixel-latency.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer           # must print 1
```

Wiring: motor USB adapter on a USB-A port (`/dev/ttyUSB0`), BNO055 on the I²C pins (`board.SDA`,
`board.SCL`), Pi reachable as `sofi@raspberrypi.local`. After any re-mounting of the IMU run the axis check
(see the IMU section below).

## Flags

`vision_streamer.py` (laptop): `--pi_ip` (omit for a camera-only preview), `--env-config`, `--tag_size`
(0.080 m), `--camera` (profile in `configs/camera_settings.json`), `--device`, `--fps` (90), `--record`
(R-key recording), `--autostart` (START on READY, with `--no_preview` for headless use), `--no-target`,
`--no-roi` (full-frame detection every frame), `--full-roi` (target every frame instead of every 10th),
`--alpha`, `--beta`, `--blur` (image pre-processing; `tune_vision.py` finds them and writes
`configs/image_tune.json`).

`deploy_main.py` (Pi): `--name`, `--env-config`, `--model`, `--duration` (60 s), `--record`.

`open_loop_main.py` (Pi): the same plus `--trajectory` (`zero`, `sine`, `sine_glide`, `sine_hold`),
`--freq` (1.5 Hz), `--amp` (0.3 rad), `--offset`, `--sine_duration` (the active sine phase of the glide
and hold trajectories), `--hold_offset`, `--no-target`, `--shadow` with `--model`. `sine_glide` cuts the
motor to zero after the sine phase, `sine_hold` holds the tail at `--hold_offset` (steady-state turning).

`post_run.py` (laptop): `<run_dir>`, `--render` (`processed/replay.mp4`, needs the video), `--no-noise`,
`--all`. `raw/` is written only during the run; `processed/` is regenerable.

## Vision

Each frame is contrast-adjusted (`alpha`, `beta`), optionally blurred, and handed to the AprilTag detector
with `quad_decimate` 4 (the quad search runs at quarter resolution, the corners are re-fitted at full
resolution, so the pose is not degraded) and `refine_edges` 1.

After the first full-frame detection each tag is tracked in a padded region of interest (1.5 times its
box): the fish every frame, the target every 10th frame (its last pose is passed through in between), the
anchor every 30th (it does not move). Three misses in a row fall back to a full-frame scan. The status line
shows the mode: `[ROI-s]` (default), `[ROI]` (`--full-roi`), `[FULL]` (fallback). `shared/tag_tracker.py`
holds the per-tag state.

Measured on the deployment laptop (Windows, MSMF + MJPEG, 1920×1080): full-frame about 52 fps, ROI 65 to
75 fps, camera ceiling 90 fps. Under WSL the V4L2 backend caps at 30 fps, so the streamer ran natively on
Windows. On the 2026-05-20 runs the delivered rate was 32 to 42 fps (older laptop, `quad_decimate` 2),
the median vision delay 32 ms, and 3 % of the control steps saw no frame for more than 500 ms: WiFi near
the water surface. The training noise model (vision latency U(10, 50) ms, 10 % drops, 30 Hz camera)
sits inside these numbers except for that tail. `laptop/camera_stress_test.py` measures the raw
frame-delivery rate of a camera without detection.

## Run folder

```
data/deploy_runs/<YYYYMMDD_HHMMSS>_<name>/
  raw/                      written during the run, never edited
    run.csv                 Pi, 50 Hz: pi_time, laptop_time, obs_00..23, act_00..05, motor_setpoint, reward,
                            done, vision_age_ms, vision_flags, frame_seq, cpg_phase, motor_load
    run_config.json         env_type, policy hash, clock offset, start time
    video.avi               laptop, MJPEG, with --record
    frame_timestamps.csv    laptop, one line per written frame (exact variable frame rate)
  processed/                post_run.py; delete and re-run at will
    summary.png, diagnostics.png, noise_config.json + noise_report.md (zero-trajectory runs), replay.mp4 (--render)
```

Commit a run only if a paper or report reads it; the 2026-05-20 session lives in `data/02_deploy_20260520/`
in exactly this layout.

## IMU axes

The gyro's yaw rate is the policy's most important navigation input and a wrong axis or sign corrupts it
silently. All three systems agree on: `wz` positive when the fish turns counter-clockwise seen from
above (MuJoCo `headRollZ`, the field-test EKF's `Meas_GyrZ`, and the BNO055 after the remap in
`pi/imu_reader.py`):

```
wx =  raw_z    roll rate
wy = -raw_x    pitch rate
wz = +raw_y    yaw rate   → obs[3]
```

The field-test script writes this remap into the BNO055 registers; the deploy stack keeps the sensor in
`IMUPLUS_MODE` and applies the same transformation in Python (no register writes, no fusion resets).
Confirmed on hardware on 2026-05-15 with `sysid/imu_axis_check.py`: nose left gives `wz > 0`. After any
re-mounting of the IMU, run that check with the motor unpowered, rotate the fish nose-left and confirm
`wz > 0`; if another axis spikes, update the three lines in `imu_reader._run()` and run the check again.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Pi: `TIMEOUT waiting for vision` | streamer not running or wrong `--pi_ip` | start the streamer first, check the IP |
| laptop never shows `Pi READY` | control port 7979 blocked | firewall; check the Pi reached the clock sync |
| R does nothing | no `--record` or no `--pi_ip` on the laptop | add them |
| START timeout (120 s) | R not pressed | press R, or `--autostart` |
| `run.csv` empty | vision lost at once (deployment aborts after 1 s without the fish tag) | tags visible? `--tag_size`? open-loop with `--no-target` if no target is deployed |
| open-loop prints `Vision lost` and continues | normal: open-loop never aborts on vision | ignore |
| clock offset 0.0 | sync server not up yet | start the streamer first; the sync retries three times |
| offset differs by > 100 ms between runs | NTP corrected a clock | normal; `run_config.json` holds the offset of each run |
| IMU gyro timeout | BNO055 not stable | keep the fish still for 5 s after power-on |
| encoder reads 0 every few steps | USB-serial latency timer at 16 ms | the udev rule above |
| encoder frozen | USB-serial error, cached state | check the cable and `/dev/ttyUSB0` |
| `wz` spikes on pitch, or has the wrong sign on a left turn | IMU re-mounted | the axis check above |
| video has fewer frames than the run has steps | the MJPEG recorder sustains about 20 fps at 1080p on an 8-core laptop and drops frames on runs longer than about 33 s; the timestamps stay exact, so timing analyses are unaffected | keep runs short, or record at 720p |

`obs_verify/` is the ten-step check that every observation dimension on the hardware matches the training
convention; run it after any change to the sensors, the tags or the assembler.
