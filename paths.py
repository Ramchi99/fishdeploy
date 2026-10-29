"""Where fishdeploy keeps its data.

`data/` holds the subset of the measured data that the paper's figures, tables and fits read, laid out
exactly like the public archive on OneDrive (which adds the videos, the raw phone data and the
calibration frames under the same folder names). New experiments land in `data/<YYYYMMDD>_<topic>/`.

Every script locates data through this module, so moving a folder means changing one line here.
Importing it also puts the repository root on sys.path.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "data"

FIELD = DATA / "01_field_openloop_20260401"            # the 12 open-loop runs, run_01 … run_12
DEPLOY = DATA / "02_deploy_20260520"                     # the deployment runs of the paper, <timestamp>_<name>/raw/run.csv
ANALYSIS = DEPLOY / "analysis"                           # cache/, cache_redetect/, cache_heading/, reachability/, report_strobes/
CANONICAL = DEPLOY / "canonical"                         # the curated 1 kHz segments of the closed-loop fits + their manifest
TAIL_BENCH = DATA / "03_sysid" / "tail_benchmark"        # air/ (the 100 r17.5 sweeps), stiffness/
IMU_NOISE = DATA / "03_sysid" / "imu_noise"

POLICIES = REPO / "policies"
NEW_RUNS = DATA / "deploy_runs"                          # where scripts/fetch_data.sh puts new deployment runs
NEW_FIELD = DATA / "field_runs"                          # where new open-loop sessions go: <session>/run_*/raw/ for the pipeline
FIELD_LOGS = DATA / "field_logs"                         # where field_test/openloop_swim.py writes its log on the Pi

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
