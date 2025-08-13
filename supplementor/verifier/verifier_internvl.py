# supplementor/verifier/verifier_internvl.py
from __future__ import annotations
import re, torch
from dataclasses import dataclass
from typing import List
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

_Y = re.compile(r"\b(yes|true|correct)\b", re.I)
_N = re.compile(r"\b(no|false|incorrect)\b", re.I)
_U = re.compile(r"\b(uncertain|not sure|unsure|can.?t tell|ambiguous)\b", re.I)
def _ynu(x:str) -> str:
    x = x.strip()
    if _U.search(x): return "UNCERTAIN"
    if _Y.search(x): return "YES"
    if _N.search(x): return "NO"
    return "UNCERTAIN"

@dataclass
class InternVLConfig:
    model_id: str = "/mnt/models/InternVL2_5-26B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    offline: bool = True
    max_new_tokens: int = 16

class InternVLVerifier:
    def __init__(self, cfg: InternVLConfig):
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True, local_files_only=cfg.offline)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True, local_files_only=cfg.offline)
        dtype = torch.bfloat16 if cfg.torch_dtype in ("auto","bfloat16","bf16") else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, trust_remote_code=True,
                                torch_dtype=dtype, device_map=cfg.device_map, local_files_only=cfg.offline)

    def ask(self, image_paths: List[str], question: str) -> str:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        # 具体 prompt/输入格式请参考该模型卡，以下是通用骨架：
        messages = [{"role":"user","content":[*[
                        {"type":"image", "image": im} for im in images
                    ], {"type":"text","text": question + "\nAnswer ONLY: YES or NO or UNCERTAIN."}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False, eos_token_id=self.tokenizer.eos_token_id)
        ans = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _ynu(ans)
