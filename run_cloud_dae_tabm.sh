#!/usr/bin/env bash
set -e

echo "=================================================="
echo "OMNICAMPUS CLOUD GPU: DAE + TABM-32 (463 FEATURES)"
echo "=================================================="

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "Warning: nvidia-smi not found. Training may fall back to CPU."
fi

echo "Verifying PyTorch CUDA acceleration..."
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Installing PyTorch with CUDA 12.4 support..."
    pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || pip install torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "PyTorch CUDA acceleration is active and verified."
fi

pip install -q --break-system-packages -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt

python3 -c "import torch; print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

mkdir -p output

echo "Starting Transductive Swap-Noise DAE + 10-Fold TabM-32 Training..."
python3 -u train_cloud_dae_tabm463.py

echo "TabM-32 + DAE Training Complete. Generated output artifacts:"
ls -lh output/
