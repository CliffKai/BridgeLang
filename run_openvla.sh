#!/bin/bash

# 初始化 Conda
source /root/anaconda3/etc/profile.d/conda.sh

# 激活 openvla
conda activate openvla

echo "[INFO] Running OpenVLA with Flash Attention disabled, using math backend and head_dim padding"

# 环境变量设置
export PYTORCH_SDP_BACKEND=math
export PYTORCH_ENABLE_SDPA=0
export PT_SDPA_ENABLE_HEAD_DIM_PADDING=1

# 启动脚本
python run_openvla_demo.py
