#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD: 10-FOLD LGBM-DAE-491 STREAM"
echo "=================================================="

mkdir -p output

echo "Starting 10-Fold LGBM-DAE-491 Training..."
python3 -u train_cloud_lgbm_dae491.py

echo "LGBM-DAE Training Complete. Generated output artifacts:"
ls -lh output/
