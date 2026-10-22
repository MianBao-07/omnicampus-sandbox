#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD GPU TABULAR RESNET (NVIDIA L4)"
echo "=================================================="

nvidia-smi

echo "Installing / Verifying python dependencies..."
if ! pip install -q torch -r requirements.txt 2>/dev/null; then
    echo "Managed environment detected: applying --break-system-packages..."
    pip install -q --break-system-packages torch -r requirements.txt
fi

echo "Starting 10-Fold Pre-Activation Tabular ResNet..."
if command -v python3 &>/dev/null; then
    python3 train_cloud_tabular_resnet.py
else
    python train_cloud_tabular_resnet.py
fi

echo "Tabular ResNet Training Complete. Generated artifacts:"
ls -lh output/
