# Data

Rule: a file is in this repository if a figure, a table or a fit in the paper reads it. Everything larger
(video, raw phone data, calibration frames, identification checkpoints) is in the archive
`SOFI_ICRA2026` on OneDrive, <https://1drv.ms/f/c/b311444f04e9154f/IgA5FN-kerWHS5HrusoArlQrAQ6lR6rIJepderhEuW8ZCB8?e=016YIx>,
whose folders mirror the paths below. `SHA256SUMS` in the archive verifies a download.

## In this repository (about 150 MB)

| Path | Files | What | Produced by | Read by |
|---|---|---|---|---|
| `field_test/data/20260401_first_swim/run_<n>_<f>Hz_<amp>mm[_<offset>mm]/raw/*.csv` | 12 | The Pi's motor + IMU log of each open-loop run | `field_test/unified_test.py` | `field_test/pipeline/01…03` |
| `…/processed/02_clean_path.csv` | 12 | AprilTag pose track in the world frame | `field_test/pipeline/02_process_swimming.py` | `03_prep_field2sim.py`, `04_plot_field_test.py` |
| `…/processed/03_field2sim_ready.csv` | 12 | Resampled pose + motor + IMU on one timeline | `field_test/pipeline/03_prep_field2sim.py` | fishsim `sysid/fluid/opt_field2sim.py` (Table I), paper Fig. 5, Fig. 9 |
| `deploy/runs/20260520_<hhmmss>_<name>/raw/run.csv` | 43 | Pi log at 50 Hz: observation, action, CPG parameters and phase, motor state, pose age | `deploy/pi/deploy_main.py` | paper Fig. 6 (`full_success_2`), Fig. 9 (all 43), appendix run table, `analysis/` |
| `…/raw/run_config.json`, `frame_timestamps.csv` | 43 + 43 | Env config, policy hash, clock offset; camera frame times | `deploy_main.py`, `deploy/laptop/vision_streamer.py` | `analysis/build_analysis.py` |
| `analysis/cache/<name>.npz`, `cache_redetect/<name>.npz` | 43 + 42 | Tracked fish, anchor and target positions per frame, corrected timeline | `analysis/build_analysis.py`, `redetect_trajectories.py` | paper Fig. 7, Fig. 8 |
| `analysis/cache_heading/initial_heading.json` | 1 | Initial heading of each run from the first tag detections | fishsim `paper/tools/make_heading_cache.py` | paper Fig. 8 |
| `analysis/reachability/run_geometry.csv`, `reachability_summary.md` | 2 | Target radius, angle, outcome, closest approach per run; the 13/22, 7/8, 20/30 reach counts | `analysis/build_reach.py` | paper Fig. 8, report table |
| `analysis/report_strobes/*.png` | 6 | Strobe composites of six runs | `analysis/gen_report_strobes.py` (needs the videos) | paper Fig. 1, Fig. 7 |
| `analysis/RUN_CATALOG.md` | 1 | Every run of the session with outcome, target, frame drops, detection quality | `analysis/build_catalog.py` | the appendix run table |
| `sysid/tail/data/*_Benchmark_air_*.csv` | 160 | Motor sweeps with the tail in air (commanded and measured position, velocity, load) | `sysid/tail/run_dynamic_benchmark.py` | fishsim `sysid/tail_stiffness/` (the 100 `r17.5` files; paper Fig. 4), `sysid/motor_noise/` |
| `sysid/tail/data/*_Benchmark_water_*.csv` | 108 | The same sweeps in water | same | `sysid/tail/compare_air_water.py` |
| `sysid/tail/data/*_Stiffness_Calibration.csv` | 10 | Static stiffness ramps | `sysid/tail/calibrate_stiffness.py` | `sysid/tail/plot_stiffness.py` |
| `sysid/imu/data/imu_9DOF_*.csv` | 2 | BNO055 logs, fish still | `sysid/imu/log_9dof.py` | `sysid/imu/analyze_imu.py`, the sensor noise model in fishsim |
| `policies/sofi_cpg_rl_policy.zip` | 1 | The deployed policy, md5 `7bbe08b8e0999fb4324b073d324d8c78` | fishsim `rl/train_rl_curriculum.py` | `deploy/pi/policy_runner.py` |

## In the archive only

| Archive folder | Repo path it completes | What |
|---|---|---|
| `01_field_openloop_20260401/run_<nn>_…/raw/` | `field_test/data/20260401_first_swim/<run>/raw/` | Phone video, phone IMU streams, sync frames |
| `…/processed/` | `…/processed/` | All other pipeline outputs: trimmed logs, quality report, swim clip, plots |
| `02_deploy_20260520/<run>/raw/video.avi`, `processed/*.png` | `deploy/runs/<run>/` | Overhead video of every run, post-run summary and diagnostics plots |
| `02_deploy_20260520/analysis/{videos,strobe_plots,tag_pipeline_frames,reachability}` | `analysis/` | Rendered trace videos, strobe plots, vision-tuning frames, reachability plots |
| `03_sysid/tail_benchmark/videos/` | `sysid/tail/` | 17 videos of the benchmark sweeps |
| `03_sysid/imu_noise/imu_6DOF_*.csv` | `sysid/imu/data/` | The 37-minute 6-DOF static log (18 MB) |
| `03_sysid/camera_calibration/` | `field_test/pipeline/calibration_evidence/` | 276 checkerboard frames behind `vision_settings.json` |
| `03_sysid/motor_identification/` | (fishsim `sysid/motor/results/`) | Result JSONs of the motor study |
| `04_policies/` | `policies/` | The deployed policy and the direct-action policy |
| `05_fluid_identification/` | (fishsim `sysid/fluid/`) | CMA-ES checkpoints and convergence plots of the fluid fits; refitA/B/C on the deployment runs |

Not kept anywhere public: deployment sessions before 2026-05-20, aborted runs, recordings without a Pi log,
field runs 13 to 15, intermediate policy checkpoints.

## Notes on the deployment set

- The 43 runs are numbered 1 to 43 in the paper's appendix table in recording order; `analysis/RUN_CATALOG.md`
  has the same order with the folder names.
- `finetune_run_4` (`20260520_194717`) is a **fail** (closest approach 246 mm, heavy frame drops); the 20/30
  reach count for static plus disturbance targets assumes that. An older catalogue line that marked it a success is wrong.
- All 43 runs carry `"model_hash": "7bbe08b8"`; the file name recorded in `run_config.json`
  (`SAC_curriculum_stage0_fwd_wide.zip`) is the name the checkpoint had on the Pi that day and is the same bytes as
  `policies/sofi_cpg_rl_policy.zip`.
- The Pi and laptop CSVs are CRLF. `.gitattributes` disables line-ending normalisation so they stay byte-identical.
