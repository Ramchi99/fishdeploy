# fishdeploy

Hardware side of **Swim2Real / "All You Need Is Low Fidelity"** (ICRA 2026 submission): a tendon-driven soft
robotic fish (SOFI, Soft Robotics Lab, ETH Zurich) that swims to targets with a reinforcement-learning policy
trained entirely in a low-fidelity MuJoCo simulation. This repository has everything that touches the physical
fish: the motor and sensor drivers, the system-identification measurements, the open-loop field tests the fluid
model was fitted on, the closed-loop deployment stack, and the analysis of the deployment session. The
simulation, the model fits, the training code and the paper figures are in
[fishsim](https://github.com/srl-ethz/fishsim) (branch `sofi`), which reads its measured data from this repo.

## The fish

- Tail: silicone tail with five passive hinges, driven by one tendon over a pulley (radius 12.5 mm in the field
  software; the tendon travel in mm divided by 12.5 is the tail angle in rad).
- Actuator: one Dynamixel XW430-T200-R (waterproof), Protocol 2.0 at 1 Mbaud on `/dev/ttyUSB0`.
- Sensors: a BNO055 IMU in the head (gyro at 50 Hz in the observation), and an overhead camera on the laptop
  that tracks an AprilTag on the fish, one on the pool edge (world anchor) and one on the target.
- Compute: a Raspberry Pi 5 inside the fish runs the policy at 50 Hz and drives the motor; the laptop streams
  the tag poses to the Pi over UDP and records the video. Both log every step.
- Pool: a 4 by 2 m tank, camera about 2.5 m above the water.

## Layout

| Folder | What |
|---|---|
| `data/` | The measured data the paper reads, in the folder names of the public archive: the 12 open-loop field runs, the deployment runs of 2026-05-20 with the analysis caches and the curated closed-loop segments, the bench sweeps. `data/README.md` says for every file what produced it and what reads it |
| `hardware/` | Dynamixel wrappers (`dxlSetup/`), IMU reader, UPS/battery monitor, single-motor test scripts |
| `sysid/` | Measurement scripts: the tail benchmark (`tail/`: stiffness ramp, frequency sweeps in air and water), the IMU noise log (`imu/`), the IMU axis check; `sysid/README.md` is the protocol |
| `field_test/` | Open-loop swimming runs: `openloop_swim.py` on the Pi, the laptop pipeline `pipeline/01…04` (video sync, AprilTag pose, resampling to `03_field2sim_ready.csv`, plots), camera calibration, cross-run evaluation |
| `deploy/` | Closed-loop deployment: `pi/` (policy, CPG, motor, IMU, logger), `laptop/` (vision streamer, post-run analysis, renders), `shared/` (UDP protocol, tag tracker), `configs/`, `preflight.py`, `obs_verify/` |
| `analysis/` | The post-session analysis of the deployment runs: tag re-detection, trajectory caches, reachability tables, strobe composites, trace videos; `canonical/`: the conversion of curated segments into the 1 kHz CSVs the closed-loop fits read |
| `policies/` | The deployed SAC policy (md5 `7bbe08b8`) |
| `scripts/` | `deploy_to_pi.sh` (rsync code to the Pi), `fetch_data.sh` (pull a run back and post-process it) |
| `paths.py` | Where every script finds `data/`; change a folder in one place |

## Setup

Laptop (Linux or WSL, Python 3.11):

```bash
git clone https://github.com/Ramchi99/fishdeploy && cd fishdeploy
pip install -r requirements.txt
pytest                                  # 12 test files, no hardware
```

Raspberry Pi 5 (once; details and the USB-latency fix in `deploy/README.md`):

```bash
./scripts/deploy_to_pi.sh               # rsync this repo to sofi@raspberrypi.local:~/fishdeploy
ssh sofi@raspberrypi.local
python3 -m venv ~/fishdeploy/venv && ~/fishdeploy/venv/bin/pip install -r ~/fishdeploy/requirements-pi.txt
```

`PI_HOST=<ip>` and `PI_USER=<user>` override the defaults of both scripts.

## 1. System identification (measure on the fish)

```bash
# Pi, tail hanging in air: static stiffness ramp, then the frequency × amplitude sweep
python3 sysid/tail/calibrate_stiffness.py --max_current 600 --pulley_radius 17.5 --dataset r17.5
python3 sysid/tail/run_dynamic_benchmark.py --env air --pulley_radius 17.5 --dataset r17.5
# the same sweep with the tail in water
python3 sysid/tail/run_dynamic_benchmark.py --env water --pulley_radius 17.5 --dataset r17.5
# Pi, fish still: IMU noise log
python3 sysid/imu/log_imu.py --mode 9dof
```

The sweeps write `data/03_sysid/tail_benchmark/<env>/<date>_<set>_r<radius>_d<amplitude>_Benchmark_<env>_<f>Hz_<amp>Amp.csv`.
The 100 files `20260320_*_r17.5_*_Benchmark_air_*` are the input of the tail-stiffness fit in fishsim
(`sysid/tail_stiffness/step1_stiffness.py`, paper Fig. 4). `sysid/README.md` has the full protocol;
`sysid/tail/analyze_dynamic_benchmark.py` plots the Bode curves on the laptop.

## 2. Open-loop field test (collect the data the fluid model is fitted on)

```bash
# Pi, fish in the pool, phone filming from above
python3 field_test/openloop_swim.py --name first_swim --freq 2.0 --amp 9.0 --duration 30.0
# laptop, after copying the phone video and the Pi CSV into data/<session>/<run>/raw/
python3 field_test/pipeline/01_sync_and_convert.py     # phone clock → Unix time via the reference frame
python3 field_test/pipeline/02_process_swimming.py     # AprilTag → gravity-aligned 6-DOF pose
python3 field_test/pipeline/03_prep_field2sim.py       # resample → processed/03_field2sim_ready.csv
python3 field_test/pipeline/04_plot_field_test.py
```

The twelve runs of 2026-04-01 (1 to 4 Hz, 5 to 12 mm, two with a turn offset) are in
`data/01_field_openloop_20260401/`. Their `03_field2sim_ready.csv` files are what fishsim's
`sysid/fluid/opt_field2sim.py` fits the 15 fluid coefficients to (paper Fig. 5, Table I) and what paper Fig. 9
reads. Camera calibration, the phone clock offset and the per-run vision settings: `field_test/README.md`, `field_test/vision_settings.json`.

## 3. Deployment (the policy drives the fish)

```bash
python3 deploy/preflight.py                            # laptop: checks camera, Pi, policy, config
./scripts/deploy_to_pi.sh
python3 deploy/laptop/vision_streamer.py --pi_ip raspberrypi.local \
    --env-config deploy/configs/cpg_24d.json --name run_1 --record
```

```bash
# Pi
python3 deploy/pi/deploy_main.py --name run_1 --env-config deploy/configs/cpg_24d.json \
    --model policies/sofi_cpg_rl_policy.zip --duration 60 --record
```

The Pi calibrates the gyro (fish must be still), waits for the first pose packet and reports READY. Press R
in the laptop window to start and R again to stop. Both machines write into
`data/deploy_runs/<timestamp>_<name>/raw/` (the laptop's START timestamp names the run on both sides); the laptop
then pulls the Pi's half and runs `deploy/laptop/post_run.py`, which writes `processed/`.
`deploy/README.md` is the complete reference: the three workflows (deployment, open loop with an optional shadow
policy, noise floor), the Pi setup, every flag, the IMU axes, and why inference must be stochastic. `deploy/obs_verify/` walks through the ten physical checks
that each observation dimension matches the simulator before a water run.

The 43 runs of 2026-05-20 in `data/02_deploy_20260520/` are the paper's deployment set (static, disturbance, out-of-distribution,
moving-target and figure-eight targets), each with `raw/run.csv` (50 Hz: observation, action, CPG state, pose age),
`run_config.json` and `frame_timestamps.csv`. The videos are in the archive.

## 4. Analysis of a deployment session

```bash
python analysis/build_analysis.py      # re-detect the tags in each video, cache the tracks, render trace videos
python analysis/build_reach.py         # reach outcomes, target geometry, reachability plots
python analysis/build_catalog.py       # RUN_CATALOG.md
python analysis/gen_report_strobes.py  # the strobe composites of paper Fig. 1 and Fig. 7
```

Needs the videos from the archive next to the CSVs. The outputs that the paper figures read (`cache/`,
`cache_redetect/`, `cache_heading/`, `reachability/run_geometry.csv`, `report_strobes/`) are committed under
`data/02_deploy_20260520/analysis/`, so fishsim rebuilds every figure from a clone of this repo alone.

The closed-loop fits in fishsim (the direct-action fluid set and the identified motor of the paper's motor
table) read curated 1 kHz segments of these runs, `data/02_deploy_20260520/canonical/`. The segments were
chosen by hand (the manifest `layer2_parts.yml` records them); `analysis/canonical/convert_segments.py`
turns a run's video and `run.csv` into the canonical CSVs.

## Archive

Videos, raw phone data, calibration frames and the identification checkpoints (15 GB) are on OneDrive:
<https://1drv.ms/f/c/b311444f04e9154f/IgA5FN-kerWHS5HrusoArlQrAQ6lR6rIJepderhEuW8ZCB8?e=016YIx>.
Its `README.md` and `SHA256SUMS` describe and verify every file; `data/README.md` here maps archive to repo paths.

## Citation and credits

Simon Ramchandani, Liam Maloney and Mike Y. Michelis, Soft Robotics Lab, ETH Zurich, 2026. The Dynamixel
wrappers in `hardware/dxlSetup/` descend from Liam Maloney's `tendonSofi`. MIT license.
