#!/bin/bash
# Smartly syncs ONLY data/plots from the Raspberry Pi back to the PC.
# It places the data exactly in the same folder structure they were generated in.

cd "$(dirname "$0")/.."

PI_USER="${PI_USER:-sofi}"
PI_HOST="${PI_HOST:-raspberrypi.local}"   # override: PI_HOST=<ip> ./scripts/fetch_data.sh
SRC_DIR="~/fishdeploy/"
DEST_DIR="./"

if ! command -v rsync &> /dev/null
then
    echo "rsync could not be found, please install it to use this script."
    exit
fi

# Optional: pass "rl <run_name>" to fetch a specific RL run, otherwise fetch everything
if [ "$1" = "rl" ]; then
    if [ -z "$2" ]; then
        echo "Usage: ./scripts/fetch_data.sh rl <run_name>"
        echo "  run_name: e.g. 20260508_143022_shadow_zero_1"
        exit 1
    fi
    RUN_NAME="$2"
    RAW_DEST="./data/deploy_runs/${RUN_NAME}/raw"
    mkdir -p "$RAW_DEST"

    echo "Fetching Pi:deploy/data/${RUN_NAME}/ → data/deploy_runs/${RUN_NAME}/raw/ ..."
    rsync -avz --prune-empty-dirs \
               --include="*/" \
               --include="*.csv" \
               --include="*.json" \
               --exclude="*" \
               "$PI_USER@$PI_HOST:~/fishdeploy/deploy/data/$RUN_NAME/" \
               "$RAW_DEST/" \
    || { echo "ERROR: rsync failed (exit $?). Aborting."; exit 1; }

    # Move recording files from recordings/ if present
    REC_DIR="./deploy/recordings/${RUN_NAME}"
    if [ -d "$REC_DIR" ]; then
        echo "Moving recording from $REC_DIR → $RAW_DEST/"
        [ -f "$REC_DIR/video.avi" ] && mv "$REC_DIR/video.avi" "$RAW_DEST/"
        [ -f "$REC_DIR/frame_timestamps.csv" ] && mv "$REC_DIR/frame_timestamps.csv" "$RAW_DEST/"
        rmdir "$REC_DIR" 2>/dev/null || true
    fi

    # Use mujoco conda env python if available (has matplotlib etc.), else fall back
    PYTHON_BIN="${HOME}/micromamba/envs/mujoco/bin/python3"
    [ -x "$PYTHON_BIN" ] || PYTHON_BIN="python3"

    echo "Running post_run.py..."
    "$PYTHON_BIN" deploy/laptop/post_run.py "data/deploy_runs/${RUN_NAME}"
    echo "Fetch + analysis complete: data/deploy_runs/${RUN_NAME}/"
else
    echo "Fetching data files (*.csv, *.png, *.pdf, *.mat, *.log, *.txt, *.json) from $PI_USER@$PI_HOST..."
    rsync -avz --prune-empty-dirs \
               --include="*/" \
               --include="*.csv" \
               --include="*.png" \
               --include="*.pdf" \
               --include="*.mat" \
               --include="*.log" \
               --include="*.txt" \
               --include="*.json" \
               --exclude="*" \
               $PI_USER@$PI_HOST:$SRC_DIR $DEST_DIR
fi

echo "Data fetch complete! Your plots and data should be right next to your scripts."