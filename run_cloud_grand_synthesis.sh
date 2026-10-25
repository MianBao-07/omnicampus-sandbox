#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD: 10-FOLD GRAND SYNTHESIS 595"
echo "=================================================="

mkdir -p output

echo "Starting 10-Fold Grand Synthesis Early-Fusion Training..."
python3 -u train_cloud_grand_synthesis_595.py

echo "Grand Synthesis Training Complete. Generated output artifacts:"
ls -lh output/
