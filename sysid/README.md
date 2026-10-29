# sysid: measurements on the fish

Three measurements, each a folder, each writing into `data/03_sysid/` through `paths.py`. The Pi scripts run
on the Pi (`./scripts/deploy_to_pi.sh` first), the analysis scripts on the laptop after `./scripts/fetch_data.sh`.

| Folder | Measurement | Feeds |
|---|---|---|
| `tail/` | the tail bench: a static stiffness ramp, then the frequency × amplitude sweep of the motor-tendon-tail chain in air and in water | the hinge stiffness and damping fit in fishsim (`sysid/tail_stiffness/`, paper Fig. 4) reads the 100 air sweeps `20260320_*_r17.5_*` |
| `imu/` | the BNO055 noise floor with the fish still | the IMU rows of the training noise model were later re-measured in the deployment stack (`data/03_sysid/sensor_noise_20260507/`); these logs are the March baseline |
| `imu_axis_check.py` | the gyro axis and sign check after any re-mounting of the IMU | `deploy/pi/imu_reader.py` (see the IMU section of `deploy/README.md`) |

Everything below is in tendon travel (mm), converted to motor ticks with `--pulley_radius`; `--dataset` labels
the files so that two pulleys or two fish never overwrite each other. The paper's fish: pulley 17.5 mm,
dataset `01_r17.5` (the file names carry the travel as `_d05.0`, `_d07.0`, `_d09.0`).

## Tail bench

**1. Static stiffness ramp (in air).** Finds the safe tendon travel and shows stick-slip.
```bash
python3 sysid/tail/calibrate_stiffness.py --max_current 300 --pulley_radius 17.5 --dataset my_fish   # Pi; raise the current step by step
python  sysid/tail/plot_stiffness.py --file data/03_sysid/tail_benchmark/stiffness/<file>.csv         # laptop
```
Start at a low current and step up until the tail reaches the bend you want; the highest `Tendon Pull`
printed is the travel to pass to the sweep. If the tendon crept, re-tighten it and repeat.
The ten ramps of the paper's fish are in `data/03_sysid/tail_benchmark/stiffness/`.

**2. Frequency × amplitude sweep (in air).** 1 to 4 Hz, 20 to 100 % of the travel, one CSV per point with
commanded and measured position, velocity and load.
```bash
python3 sysid/tail/run_dynamic_benchmark.py --env air --pulley_radius 17.5 --max_tendon_travel 9.0 --dataset my_fish   # Pi
python  sysid/tail/analyze_dynamic_benchmark.py --env air --dataset my_fish      # laptop: Bode curves, load map, speed saturation
```
Files land in `data/03_sysid/tail_benchmark/air/` as `<date>_<dataset>_Benchmark_air_<f>Hz_<amp>Amp.csv`;
plots in `sysid/tail/plots/` (gitignored). The fit in fishsim reads the sweeps directly.

**3. The same sweep in water, head clamped.** Quantifies added mass and drag against the air sweep.
```bash
python3 sysid/tail/run_dynamic_benchmark.py --env water --pulley_radius 17.5 --max_tendon_travel 9.0 --dataset my_fish
python  sysid/tail/compare_air_water.py                                          # laptop: air against water, per point
```
The 108 water sweeps of the paper's fish are in the archive only (`03_sysid/tail_benchmark/water/`);
copy them next to the air files to run the comparison.

## IMU noise floor

```bash
python3 sysid/imu/log_imu.py --mode 6dof      # Pi; 6dof = IMUPLUS, the mode the deployment uses; 9dof adds the magnetometer
python  sysid/imu/analyze_imu.py data/03_sysid/imu_noise/<file>.csv              # laptop: std per axis, histograms
```
The logger waits for the sensor's calibration, gives you 15 s to tape it down, then logs at about 100 Hz
until Ctrl+C. The two March logs are in `data/03_sysid/imu_noise/`.

## IMU axes

```bash
python3 sysid/imu_axis_check.py              # Pi, motor unpowered; turn the fish nose-left: wz must go positive
```
Prints the software remap of `imu_reader.py` next to the sensor's hardware remap; both columns must agree.
