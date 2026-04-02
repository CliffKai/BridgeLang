# supplementor.py — BridgeLang structured reasoning supplementor
#
# Single inference:
#   python supplementor.py \
#     --model Qwen/Qwen2.5-VL-3B-Instruct \
#     --image /path/to/frame.png \
#     --task "put the apple in the box"
#
# HTTP server mode:
#   python supplementor.py \
#     --model Qwen/Qwen2.5-VL-3B-Instruct \
#     --server --host 127.0.0.1 --port 18081

import argparse
import base64
import io
import os
import re
import sys
import time
from typing import Optional

import torch
from PIL import Image

# 兼容 Qwen2.5-VL (Transformers ≥ 4.41 常见写法)
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as QwenVL
except Exception:
    # 旧版本命名回退
    from transformers import Qwen2VLForConditionalGeneration as QwenVL

# ===== 可选: Server 相关依赖（仅 --server 时需要） =====
try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
    import uvicorn
    _FASTAPI_OK = True
except Exception:
    _FASTAPI_OK = False


SYSTEM_PROMPT = (
    "You are a robot perception supplementer. "
    "Output ONLY the three schema fields in the exact order and format. "
    "Keys like 'name:', 'color:' are FORBIDDEN in entries. "
    "Any extra text or missing section is prohibited."
)

USER_TEXT_TEMPLATE = """<image>
<task>{task}</task>
Rules (strict):
1) You MUST output ALL sections in EXACTLY this order:
   <objects>...</objects>
   <relations>...</relations>
   <subgoals>...</subgoals>
2) Inside each section, entries MUST follow the signatures below. Do NOT invent keys like 'name:' or 'color:'.

   <objects>  entries:  object_name:color:count; object_name:color:count; ...
               - Exactly TWO colons per entry (3 fields).
               - NO numbering like '1:'; NO keys like 'name:'.
               - Examples (correct):  box:brown:1; coca_cola:red:1
               - Examples (forbidden): name:box:color:brown:1  (✗), color:brown; count:1 (✗)

   <relations> entries:  subject:predicate:object; subject:predicate:object; ...
               - Exactly TWO colons per entry (3 fields).
               - NO numbering; NO keys.
               - predicate ∈ {{on, under, left_of, right_of, near, inside}}
               - Examples (correct):  coca_cola:right_of:box
               - Forbidden: name:box:left_of:name:coca_cola (✗)

   <subgoals>  entries:  1:action object; 2:action object; ...
               - MUST be numbered 1,2,3,...
               - action ∈ {{grasp, open, close, pour, place, push, pull}}
               - Example (correct):  1:grasp coca_cola; 2:place coca_cola inside box

3) Use ONLY entities visible in the CURRENT image and the TASK.
4) NO natural-language sentences. NO extra spaces around tags. Stop IMMEDIATELY after "</subgoals>".
"""


END_TAG = "</subgoals>"


def _build_messages(pil_image: Image.Image, task: str):
    """
    Qwen2.5-VL 官方多模态对话结构:
    messages = [
      {"role": "system", "content": "..."},
      {"role": "user", "content": [{"type":"image", "image": pil}, {"type":"text", "text": "..."}]}
    ]
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": USER_TEXT_TEMPLATE.format(task=task)},
            ],
        },
    ]


def _truncate_to_end_tag(text: str) -> str:
    if END_TAG in text:
        return text[: text.index(END_TAG) + len(END_TAG)]
    return text


def _single_line(text: str) -> str:
    return " ".join(text.strip().split())


class QwenSupplementor:
    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        min_pixels: int = 256_000,
        max_pixels: int = 1_024_000,
        use_flash_attn2: bool = False,
    ):
        self.device = device
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True,
        )
        self.model = QwenVL.from_pretrained(
            model_id_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if use_flash_attn2 else "eager",
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        pil_image: Image.Image,
        task: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        messages = _build_messages(pil_image, task)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 直接把 PIL 交给 processor；其内部会做图像预处理与尺寸控制
        inputs = self.processor(
            text=[prompt],
            images=[pil_image],
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
        )
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        # 只保留新生成的部分
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw = _truncate_to_end_tag(raw)
        return _single_line(raw)


def _load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def _load_image_from_b64(b64: str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


# -------------------- CLI 路径（单次） --------------------
def main_cli(args):
    sup = QwenSupplementor(
        args.model,
        device=args.device,
        dtype=args.dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_flash_attn2=args.use_flash_attn2,
    )
    if bool(args.image) ^ bool(args.image_b64):
        if args.image:
            img = _load_image_from_path(args.image)
        else:
            img = _load_image_from_b64(args.image_b64)
    else:
        print("Exactly one of --image or --image_b64 must be provided.", file=sys.stderr)
        sys.exit(2)

    out = sup.generate(
        img,
        args.task,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    # 与旧脚本保持一致：直接把三段式打印到 stdout
    print(out)


# -------------------- Server 路径（FastAPI） --------------------
class Req(BaseModel):
    task: str = Field(..., description="原任务文本")
    image_b64: Optional[str] = Field(None, description="base64 编码的 jpg/png（与 image_path 二选一）")
    image_path: Optional[str] = Field(None, description="服务器本地可见的图像绝对路径（与 image_b64 二选一）")
    max_new_tokens: int = 256
    temperature: float = 0.0


def build_app(args):
    if not _FASTAPI_OK:
        raise RuntimeError("FastAPI/uvicorn 未安装，请在 Qwen 环境中: pip install fastapi uvicorn pydantic")

    sup = QwenSupplementor(
        args.model,
        device=args.device,
        dtype=args.dtype,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_flash_attn2=args.use_flash_attn2,
    )

    app = FastAPI()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/supplement")
    def supplement(req: Req):
        t0 = time.time()
        if bool(req.image_b64) == bool(req.image_path):
            return {"error": "exactly one of image_b64 or image_path must be provided"}
        try:
            img = _load_image_from_b64(req.image_b64) if req.image_b64 else _load_image_from_path(req.image_path)  # noqa
            text = sup.generate(
                img,
                req.task,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
            )
            return {"supplement": text, "elapsed_ms": int(1000 * (time.time() - t0))}
        except Exception as e:
            return {"error": str(e)}

    return app


def parse_args():
    p = argparse.ArgumentParser()
    # 通用
    p.add_argument("--model", type=str, required=True, help="Qwen2.5-VL-3B-Instruct 路径或 HF 名称")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--min_pixels", type=int, default=256_000)
    p.add_argument("--max_pixels", type=int, default=1_024_000)
    p.add_argument("--use_flash_attn2", action="store_true")

    # CLI 单次
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--image_b64", type=str, default=None)
    p.add_argument("--task", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)

    # Server
    p.add_argument("--server", action="store_true")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=18081)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.server:
        if not _FASTAPI_OK:
            print("请先安装: pip install fastapi uvicorn pydantic", file=sys.stderr)
            sys.exit(2)
        app = build_app(args)
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
    else:
        # 兼容旧用法: 需要 --image 或 --image_b64 + --task
        if not args.task:
            print("--task 必填（非 --server 模式）", file=sys.stderr)
            sys.exit(2)
        main_cli(args)
