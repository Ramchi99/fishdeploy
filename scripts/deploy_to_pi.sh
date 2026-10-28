#!/bin/bash
# Syncs code (and existing data) from the host PC to the Raspberry Pi

# Go to project root (one dir up from scripts)
cd "$(dirname "$0")/.."

PI_USER="${PI_USER:-sofi}"
PI_HOST="${PI_HOST:-raspberrypi.local}"   # override: PI_HOST=<ip> ./scripts/deploy_to_pi.sh
DEST_DIR="~/fishdeploy"

if ! command -v rsync &> /dev/null
then
    echo "rsync could not be found, please install it to use this script."
    exit
fi

echo "Deploying to $PI_USER@$PI_HOST:$DEST_DIR..."
# The Pi needs the code, the policy and the configs, not the measured data or the analysis caches.
rsync -avz --exclude '.git/' \
           --exclude 'data/' \
           --exclude 'analysis/' \
           --exclude '__pycache__/' \
           --exclude '*.mp4' \
           --exclude '*.avi' \
           --exclude 'venv/' \
           --exclude '.venv/' \
           ./ $PI_USER@$PI_HOST:$DEST_DIR

echo "Deployment complete."