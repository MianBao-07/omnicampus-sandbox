#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD GPU TABM-32 (NVIDIA L4)"
echo "=================================================="

nvidia-smi

echo "Installing / Verifying python dependencies with CUDA 12.4 support..."
pip install -q --break-system-packages --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || pip install -q --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
pip install -q --break-system-packages -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt

python3 -c "import torch; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "Starting 10-Fold TabM-32 + In-Fold k-NN 500 Manifold Injection..."
if command -v python3 &>/dev/null; then
    python3 train_cloud_tabm.py
else
    python train_cloud_tabm.py
fi

echo "TabM-32 Training Complete. Generated artifacts:"
ls -lh output/
