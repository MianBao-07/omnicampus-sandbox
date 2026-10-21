#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD GPU BURST RUNNER (NVIDIA L4)"
echo "=================================================="

nvidia-smi

echo "Installing / Verifying python dependencies..."
if ! pip install -q -r requirements.txt 2>/dev/null; then
    echo "Managed environment detected: applying --break-system-packages..."
    pip install -q --break-system-packages -r requirements.txt
fi

echo "Starting Multi-Seed Cloud Burst Zoo v21..."
if command -v python3 &>/dev/null; then
    python3 train_cloud_burst_zoo_v21.py $1
else
    python train_cloud_burst_zoo_v21.py $1
fi

echo "Cloud Burst Finished. Generated prediction artifacts:"
ls -lh output/
