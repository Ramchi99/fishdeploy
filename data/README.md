# data: the measured data the paper reads

Rule: a file is here if a figure, a table or a fit of the paper reads it. The folder names are those of
the public archive on OneDrive (<https://1drv.ms/f/c/b311444f04e9154f/IgA5FN-kerWHS5HrusoArlQrAQ6lR6rIJepderhEuW8ZCB8?e=016YIx>),
which holds the same tree plus everything too large for git: the videos, the raw phone data, the
calibration frames, the identification checkpoints. Copying an archive folder over its twin here completes
it. `SHA256SUMS` in the archive verifies a download.

Every script finds these folders through `paths.py` at the repository root. New experiments go to
`data/<YYYYMMDD>_<topic>/` (the field-test pipeline) or `data/deploy_runs/<run>/` (`scripts/fetch_data.sh`);
commit a new run only if a paper or report reads it.

## 01_field_openloop_20260401: the twelve open-loop runs (57 MB)

Pool, 2026-04-01, open-loop sine gaits (1 to 4 Hz, 5 to 12 mm tendon amplitude; runs 08 and 09 with a
turn offset, run 12 with a different motor P gain). The fluid model was fitted on runs 03, 04, 07, 09, 10, 11
and validated on 01, 02, 05, 06, 08, 12 (run 02 was nudged by hand).

| File per run | What | Produced by | Read by |
|---|---|---|---|
| `raw/<timestamp>_first_swim_<f>Hz_<amp>mm.csv` | the Pi's motor + IMU log | `field_test/openloop_swim.py` | `field_test/pipeline/01…02` |
| `processed/02_fish_log_trimmed.csv` | that log cut to the video window, on the synced clock | `field_test/pipeline/02_process_swimming.py` | `03_prep_field2sim.py` |
| `processed/02_clean_path.csv` | the AprilTag pose track in the world frame | `02_process_swimming.py` | `03_prep_field2sim.py`, `04_plot_field_test.py` |
| `processed/03_field2sim_ready.csv` | pose, motor and IMU resampled to one timeline | `03_prep_field2sim.py` | fishsim `sysid/fluid/opt_field2sim.py` (Table I), paper Fig. 5, Fig. 9, the motor fit |

In the archive only: `raw/VID_*.mp4` (phone video), `raw/<timestamp>/` (phone IMU, sync frames), the other
pipeline outputs and plots.

## 02_deploy_20260520: the closed-loop deployment runs (135 MB)

Pool, 2026-05-20, the deployed policy driving the fish. 43 runs are the paper's deployment set (rows 1 to 43
of the appendix run table, in recording order; `RUN_CATALOG.md` lists them with outcome, target geometry and
detection quality). Two more runs, `151453_hope_didnt_work_1` and `155400_hope_didnt_work_4`, are here because
the closed-loop fits use segments of them.

| File | What | Produced by | Read by |
|---|---|---|---|
| `<run>/raw/run.csv` | Pi log at 50 Hz: observation, action, CPG parameters and phase, motor state, pose age | `deploy/pi/deploy_main.py` | paper Fig. 6 (`full_success_2`), Fig. 9 (the 43), the appendix table, `analysis/`, the motor fit |
| `paper_runs.txt` | the 43 run folders of the paper's deployment set, one per line (the figures iterate over this list, not over the folder) | by hand | paper Fig. 9, Fig. 12, the domain figures |
| `<run>/raw/run_config.json`, `frame_timestamps.csv` | env config, policy hash (`7bbe08b8` in all), clock offset; camera frame times | `deploy_main.py`, `deploy/laptop/vision_streamer.py` | `analysis/build_analysis.py`, the fits |
| `analysis/cache/<name>.npz`, `cache_redetect/<name>.npz` | tracked fish, anchor and target positions per frame, corrected timeline | `analysis/build_analysis.py`, `redetect_trajectories.py` (from the videos) | paper Fig. 7, Fig. 8 |
| `analysis/cache_heading/initial_heading.json` | initial heading per reach run | `analysis/make_heading_cache.py` (from the videos in the archive) | paper Fig. 8 |
| `analysis/reachability/run_geometry.csv`, `reachability_summary.md` | target radius, angle, outcome, closest approach; the 13/22, 7/8, 20/30 counts | `analysis/build_reach.py` | paper Fig. 8, the report |
| `analysis/report_strobes/*.png` | strobe composites | `analysis/gen_report_strobes.py` (from the videos) | paper Fig. 1, Fig. 7 |
| `canonical/<part_id>.csv` | 18 curated segments resampled to 1 kHz: pose, yaw, velocities, motor position and velocity, gyro | `analysis/canonical/convert_segments.py` (from the videos and `run.csv`, following `layer2_parts.yml`) | fishsim `sysid/closed_loop/refit_fluid.py` (the direct-action fluid), `fit_motor.py` (Table "motor") |
| `canonical/layer2_parts.yml`, `layer1_runs.yml` | the curation manifest: which runs, which [t_start, t_end] parts, starts-at-rest flags | hand-curated in the private fluidID workspace | `convert_segments.py`, both fits |
| `canonical/motor_zeros.json`, `motor_zero.json`, `anchor_calibration.json`, `detection/*.json` | motor-zero corrections, the session anchor, per-run AprilTag detection parameters | the fluidID workspace | `convert_segments.py`, the motor fit |

In the archive only: `<run>/raw/video.avi`, `<run>/processed/*.png`, `analysis/videos/`, `strobe_plots/`,
`tag_pipeline_frames/`, the reachability plots.

## 03_sysid: the bench measurements (4 MB)

| Folder | What | Produced by | Read by |
|---|---|---|---|
| `tail_benchmark/air/` | 100 motor sweeps with the tail in air (`20260320_*_r17.5_d{05,07,09}_Benchmark_air_<f>Hz_<amp>Amp.csv`: commanded and measured position, velocity, load) | `sysid/tail/run_dynamic_benchmark.py --env air` | fishsim `sysid/tail_stiffness/step1_stiffness.py` (hinge stiffness and damping, paper Fig. 4) |
| `tail_benchmark/stiffness/` | 10 static stiffness ramps | `sysid/tail/calibrate_stiffness.py` | `sysid/tail/plot_stiffness.py` |
| `imu_noise/` | two BNO055 logs with the fish still (9-DOF mode) | `sysid/imu/log_imu.py --mode 9dof` | `sysid/imu/analyze_imu.py` |
| `sensor_noise_20260507/` | the sensor-noise measurement behind the training noise model: a 30 s shadow run with the motor silent and the fish still (`raw/run.csv` at 50 Hz, 14-D observation of the May policy, `run_config.json`, `frame_timestamps.csv`), and `processed/noise_config.json` + `noise_report.md` | `deploy/pi/open_loop_main.py --trajectory zero` with the laptop streamer; `deploy/laptop/post_run.py` regenerates `processed/` identically | fishsim `rl/models/sensor_model.py` took its per-channel noise from it |

In the archive only: the 108 water sweeps and the 60 r12.5 air sweeps (`sysid/tail/compare_air_water.py`
reads them), the 17 benchmark videos, the 18 MB 6-DOF IMU log, the 276 camera-calibration frames.

## Notes

- The Pi and laptop CSVs are CRLF; `.gitattributes` disables line-ending normalisation so they stay byte-identical.
- `finetune_run_4` (`194717`) is a fail (closest approach 246 mm, heavy frame drops); the 20/30 reach count assumes that.
- The file name recorded in each `run_config.json` (`SAC_curriculum_stage0_fwd_wide.zip`) is the name the
  checkpoint had on the Pi that day; its hash `7bbe08b8` is `policies/sofi_cpg_rl_policy.zip`.
