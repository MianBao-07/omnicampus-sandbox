#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD GPU BURST RUNNER (NVIDIA L4)"
echo "=================================================="

nvidia-smi

echo "Installing / Verifying python dependencies..."
pip install -q -r requirements.txt

echo "Starting Multi-Seed Cloud Burst Zoo v21..."
python train_cloud_burst_zoo_v21.py $1

echo "Cloud Burst Finished. Generated prediction artifacts:"
ls -lh output/
