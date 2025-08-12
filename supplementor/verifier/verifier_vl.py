# supplementor/verifier/verifier_vl.py
from __future__ import annotations
import re, os, torch
from typing import List, Tuple
from dataclasses import dataclass
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

from ..factlist import FactList, Obj

# 统一 YES/NO/UNCERTAIN
_Y = re.compile(r"\b(yes|true|correct)\b", re.I)
_N = re.compile(r"\b(no|false|incorrect)\b", re.I)
_U = re.compile(r"\b(uncertain|not sure|unsure|can.?t tell|ambiguous)\b", re.I)

def _ynu(text: str) -> str:
    t = text.strip()
    if _U.search(t): return "UNCERTAIN"
    if _Y.search(t): return "YES"
    if _N.search(t): return "NO"
    # 贪心兜底
    return "UNCERTAIN"

@dataclass
class VerifyConfig:
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    offline: bool = False
    temperature: float = 0.0
    max_new_tokens: int = 16

class VisualVerifier:
    """
    视觉一致性审稿人：对 FACT_LIST 中的断言进行 YES/NO/UNCERTAIN 判定。
    规则：YES 保留，NO 删除，UNCERTAIN -> 加入 factlist.uncertain（文本层按 UNCERTAIN 输出）。
    """
    def __init__(self, cfg: VerifyConfig):
        dtype = torch.bfloat16 if cfg.torch_dtype in ("auto","bfloat16","bf16") else torch.float16
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id, torch_dtype=dtype, device_map=cfg.device_map, local_files_only=cfg.offline
        )
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, local_files_only=cfg.offline)
        self.cfg = cfg

    def _ask(self, image_paths: List[str], q: str) -> str:
        content = [{"type":"image","image": f"file://{p}"} for p in image_paths]
        content.append({"type":"text","text": q})
        messages = [
            {"role":"system","content":"You are a strict visual fact verifier. Answer ONLY one token: YES or NO or UNCERTAIN."},
            {"role":"user","content": content}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[Image.open(p).convert("RGB") for p in image_paths],
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens,
                                          do_sample=False, eos_token_id=self.processor.tokenizer.eos_token_id)
        new_ids = out_ids[:, inputs.input_ids.shape[1]:]
        ans = self.processor.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
        return _ynu(ans)

    # --- 断言模板 ---
    def _q_obj(self, o: Obj) -> str:
        desc = o.name
        col = o.attributes.get("color")
        if col: desc = f"{col} {desc}"
        return f"Is there a {desc} visible?"

    def _q_pose(self, pose: str, obj_name: str) -> str:
        return f"Is the {obj_name} posed as: {pose}?"

    def _q_obst(self, target: str, obst: str) -> str:
        # 简化成存在性验证（更稳），可选再加空间关系
        return f"Is there a {obst} visible near the {target}?"

    def verify(self, image_paths: List[str], facts: FactList) -> Tuple[FactList, dict]:
        log = {"kept": [], "removed": [], "uncertain": []}
        # 复制一份
        kept = FactList(objects=list(facts.objects), location=facts.location, pose=facts.pose,
                        grasp_points=list(facts.grasp_points), obstacles=list(facts.obstacles),
                        state_text=list(facts.state_text), uncertain=list(facts.uncertain))

        # 1) 对象存在
        valid_objs = []
        for o in kept.objects:
            y = self._ask(image_paths, self._q_obj(o))
            if y == "YES":
                valid_objs.append(o); log["kept"].append(f"object:{o.name}")
            elif y == "UNCERTAIN":
                log["uncertain"].append(f"object:{o.name}"); kept.uncertain.append(f"object:{o.name}")
            else:
                log["removed"].append(f"object:{o.name}")
        kept.objects = valid_objs

        # 2) 姿态/朝向（若有对象且 pose 非空）
        if kept.objects and kept.pose:
            y = self._ask(image_paths, self._q_pose(kept.pose, kept.objects[0].name))
            if y == "NO":
                log["removed"].append(f"pose:{kept.pose}"); kept.pose = ""
            elif y == "UNCERTAIN":
                log["uncertain"].append(f"pose:{kept.pose}"); kept.uncertain.append(f"pose:{kept.pose}")

        # 3) 障碍（逐项）
        v_obst = []
        tname = kept.objects[0].name if kept.objects else "target"
        for ob in kept.obstacles:
            y = self._ask(image_paths, self._q_obst(tname, ob))
            if y == "YES":
                v_obst.append(ob); log["kept"].append(f"obstacle:{ob}")
            elif y == "UNCERTAIN":
                log["uncertain"].append(f"obstacle:{ob}"); kept.uncertain.append(f"obstacle:{ob}")
            else:
                log["removed"].append(f"obstacle:{ob}")
        kept.obstacles = v_obst

        # 4) 抓取点与可视性：仅保留可见抓取点的**存在性**断言
        v_gp = []
        for gp in kept.grasp_points:
            y = self._ask(image_paths, f"Is the grasp point '{gp}' visible on the {tname}?")
            if y == "YES":
                v_gp.append(gp); log["kept"].append(f"grasp:{gp}")
            elif y == "UNCERTAIN":
                log["uncertain"].append(f"grasp:{gp}"); kept.uncertain.append(f"grasp:{gp}")
            else:
                log["removed"].append(f"grasp:{gp}")
        kept.grasp_points = v_gp

        return kept, log
