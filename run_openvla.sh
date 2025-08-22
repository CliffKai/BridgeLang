#!/bin/bash
set -euo pipefail
source /root/anaconda3/etc/profile.d/conda.sh
conda activate openvla
cd /mnt/openvla

echo "[INFO] Python: $(python -V)"
echo "[INFO] CUDA: $(python - <<'PY'
import torch; print(torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
)"

export PYTORCH_ENABLE_SDPA=1
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1
unset PYTORCH_SDP_BACKEND

python -u run_openvla_demo.py
