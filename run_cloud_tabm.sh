#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD GPU TABM-32 (NVIDIA L4)"
echo "=================================================="

nvidia-smi

echo "Verifying PyTorch CUDA acceleration..."
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "CUDA not active: Installing PyTorch with CUDA 12.4 support..."
    pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "PyTorch CUDA acceleration already active and verified! Skipping download."
fi
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
