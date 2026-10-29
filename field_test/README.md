# field_test: the open-loop swimming runs

The fish swims a fixed sine stroke on its own, a phone films from above, and the laptop pipeline turns the
two recordings into one CSV per run that the fluid identification in fishsim reads. Nothing is sent over the
network during a run: the Pi script is started over SSH and logs locally, the phone records video and its
own IMU, and the two clocks are aligned afterwards. The twelve runs of 2026-04-01 are in
`data/01_field_openloop_20260401/`, with the pipeline's outputs in each run's `processed/`.

## 1. On the Pi: run the stroke

```bash
python3 field_test/openloop_swim.py --name first_swim --freq 2.0 --amp 9.0 --duration 30.0
```

`--freq` (Hz), `--amp` (peak-to-peak tendon travel, mm), `--offset` (mm, a turn bias), `--duration` (s),
`--p` (Dynamixel P gain, 800), `--rate` (log rate, 100 Hz), `--radius` (pulley, 12.5 mm by default: the
twelve runs were commanded with this value, so a "9 mm" run is 12.6 mm of travel at the real 17.5 mm pulley;
the pipeline converts), `--imu-mode 6dof|9dof`, `--skip-calib`. The script waits for the IMU calibration,
runs the stroke and writes `data/field_logs/<timestamp>_<name>.csv` (motor position, velocity, load,
voltage, temperature; IMU Euler, quaternion, acceleration, gyro) plus a summary PNG. Stop at 60 °C motor
temperature or below 10 V battery.

Before a session, `python3 field_test/hardware_check.py` on the Pi scans the I²C bus, calibrates the IMU,
and moves the motor once while reading the battery.

## 2. On the laptop: one folder per run, then four steps

Put the Pi's CSV, the phone video and the phone's sensor folder (its IMU CSVs and the sync frames) into
`data/<session>/run_<NN>_<f>Hz_<amp>mm/raw/`, then:

```bash
python field_test/pipeline/01_sync_and_convert.py     # phone clock → Unix time; finds the sync frame in the video; writes processed/01_*
python field_test/pipeline/02_process_swimming.py     # AprilTag pose per frame, gravity-aligned; writes 02_clean_path.csv, 02_fish_log_trimmed.csv, a quality report
python field_test/pipeline/03_prep_field2sim.py       # resamples pose, motor and gyro onto one clock; writes 03_field2sim_ready.csv
python field_test/pipeline/04_plot_field_test.py      # trajectory, speed, motor tracking and yaw plots per run
```

Steps 01, 02 and 04 walk every `data/<session>/run_*/raw/`; 02 and 03 also take one run folder as an
argument. `03_field2sim_ready.csv` is the file fishsim fits on (`--radius 17.5`, `--yaw_source imu` are
the defaults the twelve runs used). Per-run detection settings and manual outlier corrections live in
`vision_settings.json`, found with `pipeline/vision_tuner.py --video <video>`.

**The phone clock.** The phone stamps frames and IMU samples in nanoseconds since boot. `BOOT_OFFSET` in
`01_sync_and_convert.py` is the Unix time of that boot (62 ADB snapshots over 24 h, std 7 ms, drift
1 ms per day, i.e. well inside one frame). It is a constant until the phone reboots or its clock is
changed; then re-measure it with

```powershell
.\adb.exe shell "date +%s.%N; cat /proc/uptime"      # offset = Unix time − uptime, averaged over a few snapshots
```

## 3. Camera calibration

```bash
python field_test/pipeline/generate_pattern.py --diag 27 --res_w 2560 --res_h 1440    # a ChArUco board sized for a given screen
python field_test/pipeline/calibrate_camera.py --video <board video> --checker_size <mm> --marker_size <mm>
python field_test/pipeline/verify_calibration.py --video <video> --camera <name>        # reprojection check against vision_settings.json
```

The pattern is shown on a screen and filmed from many angles; the calibration writes the camera entry into
`vision_settings.json`. The 276 frames behind the phone's calibration are in the archive
(`03_sysid/camera_calibration/`).
