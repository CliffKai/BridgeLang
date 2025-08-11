# --- force per-script deps for OpenVLA ---
import sys, os
OPENVLA_DEPS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".deps_openvla"))
if OPENVLA_DEPS not in sys.path:
    sys.path.insert(0, OPENVLA_DEPS)
# --- end of per-script deps ---


from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import datetime

import os
os.environ["PYTORCH_SDP_BACKEND"] = "math"
os.environ["PYTORCH_ENABLE_SDPA"] = "0"
os.environ["PT_SDPA_ENABLE_HEAD_DIM_PADDING"] = "1"

import torch
torch.backends.cuda.sdp_kernel.enable_math = True
torch.backends.cuda.sdp_kernel.enable_flash = False
torch.backends.cuda.sdp_kernel.enable_mem_efficient = False


# ✅ 强制禁用 Flash Attention 实现，避免 head_dim 不被 32 整除的问题
torch.backends.cuda.sdp_kernel.enable_math = True
torch.backends.cuda.sdp_kernel.enable_flash = False
torch.backends.cuda.sdp_kernel.enable_mem_efficient = False

# 模型和图片路径
model_path = "/mnt/openvla/openvla-7b"
image_path = "/mnt/openvla/test.jpg"
log_file = "run_log.txt"

# 加载模型和处理器
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

# 加载图像
image = Image.open(image_path).convert("RGB")

# 自定义 prompt（根据图像实际内容调整）
prompt = "In: What action should the robot take to grasp the snack bag?\nOut:"

# 执行推理
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# 打印到终端
print("Predicted action:", action.tolist())

# 写入日志（追加模式）
with open(log_file, "a") as f:
    f.write(f"[{datetime.datetime.now()}]\n")
    f.write(f"Prompt: {prompt.strip()}\n")
    f.write(f"Predicted action: {action.tolist()}\n\n")
