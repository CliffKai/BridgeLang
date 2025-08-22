# ① 让脚本使用本地旧版 transformers/tokenizers
import sys, os
sys.path.insert(0, os.path.abspath(".deps_openvla"))

# ② 在 import torch 之前打开 SDPA + 头维 padding（给 timm/IXDNN 用）
os.environ["PYTORCH_ENABLE_SDPA"] = "1"
os.environ["PT_SDPA_ENABLE_HEAD_DIM_PADDING"] = "1"
# 别设 PYTORCH_SDP_BACKEND=math

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import datetime

# 模型和图片路径
model_path = "/mnt/BridgeLang/openvla-7b"
image_path = "/mnt/BridgeLang/test.jpg"
log_file = "/mnt/BridgeLang/run_log.txt"

# 加载模型和处理器
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation="sdpa",   # ← 关键：HF 侧用 SDPA，避免 eager 的掩码对齐坑
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda:0").eval()

# 加载图像
image = Image.open(image_path).convert("RGB")

# 自定义 prompt
prompt = "In: What action should the robot take to grasp the snack bag?\nOut:"

# 执行推理
with torch.no_grad():
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(
        **inputs,
        unnorm_key="bridge_orig",
        do_sample=False,
        use_cache=False,           # ← 关键：关掉 KV cache，彻底规避 276/275 的 off-by-one
    )

# 打印到终端
print("Predicted action:", action.tolist())

# 写入日志
with open(log_file, "a") as f:
    f.write(f"[{datetime.datetime.now()}]\n")
    f.write(f"Prompt: {prompt.strip()}\n")
    f.write(f"Predicted action: {action.tolist()}\n\n")
