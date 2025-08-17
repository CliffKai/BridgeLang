# supplementor/verifier/verifier_vl.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 引入我们自己的 FactList/Obj 定义
from ..factlist import FactList

_Y = re.compile(r"\b(yes|true|correct)\b", re.I)
_N = re.compile(r"\b(no|false|incorrect)\b", re.I)
_U = re.compile(r"\b(uncertain|not sure|unsure|can.?t tell|ambiguous)\b", re.I)

def _ynu(text: str) -> str:
    t = (text or "").strip()
    if _U.search(t): return "UNCERTAIN"
    if _Y.search(t): return "YES"
    if _N.search(t): return "NO"
    return "UNCERTAIN"

@dataclass
class VerifyConfig:
    model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    device_map: str = "balanced"      # 交给 accelerate 均衡切片
    torch_dtype: str = "bfloat16"
    offline: bool = False
    max_new_tokens: int = 16
    # 量化可选（多数非NVIDIA环境不建议开；默认关闭）
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

class VisualVerifier:
    """
    72B 视觉一致性主裁判：
    - 输入：图片路径 + Draft 解析得到的 FactList
    - 输出：校正后的 FactList + 逐条判定日志 vlog
    """
    def __init__(self, cfg: VerifyConfig):
        self.cfg = cfg
        dtype = torch.bfloat16 if cfg.torch_dtype in ("auto","bfloat16","bf16") else torch.float16

        # 用整型 GPU 编号声明每卡上限（适配 accelerate）
        try:
            n = torch.cuda.device_count()
            max_mem = {i: "28GiB" for i in range(n)}  # 不够就改成 27GiB 或 26GiB
        except Exception:
            max_mem = None

        q_kwargs = {}
        if cfg.load_in_8bit or cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            q_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=cfg.load_in_8bit,
                load_in_4bit=cfg.load_in_4bit,
                bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            )

        # 关键：low_cpu_mem_usage + device_map="balanced" + max_memory(整型键)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            local_files_only=cfg.offline,
            low_cpu_mem_usage=True,
            device_map=cfg.device_map,
            max_memory=max_mem,
            attn_implementation="eager",   # ← 同样强制 eager
            **q_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, local_files_only=cfg.offline)

    # ===== 低级问答接口 =====
    def _ask(self, image_paths: List[str], q: str) -> str:
        content = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": q + "\nAnswer ONLY: YES or NO or UNCERTAIN."})
        messages = [
            {"role": "system", "content": "You are a strict visual fact verifier. Answer ONLY YES/NO/UNCERTAIN."},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [Image.open(p).convert("RGB") for p in image_paths]
        inputs = self.processor(text=[text], images=images, padding=True, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            # 主裁判建议确定性：不采样 -> 不传 temperature/top_p
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        dec = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return _ynu(dec)

    # ===== 问句模板（可按需扩展）=====
    def q_object(self, name: str, color: str | None = None) -> str:
        desc = f"{color} {name}".strip() if color else name
        return f"Is there a {desc} visible in the image?"

    def q_pose(self, obj_name: str, pose: str) -> str:
        return f"Is the {obj_name} oriented/posed as: {pose}?"

    def q_obstacle_near(self, target: str, obst: str) -> str:
        return f"Is there a {obst} visible near the {target}?"

    def q_grasp(self, target: str, gp: str) -> str:
        return f"Is the grasp point '{gp}' visible on the {target}?"

    # ===== 高级接口：校正 FACT_LIST =====
    def verify(self, image_paths: List[str], facts: FactList) -> Tuple[FactList, Dict[str, Any]]:
        """
        对传入的 FactList 逐项做可视核验，返回：
          - facts_v: 校正后的 FactList
          - vlog: 逐条判定日志（YES/NO/UNCERTAIN）
        """
        # 取字段（全部做健壮兜底）
        objects      = list(getattr(facts, "objects", []) or [])
        location     = getattr(facts, "location", "") or ""
        pose         = getattr(facts, "pose", "") or ""
        grasp_points = list(getattr(facts, "grasp_points", []) or [])
        obstacles    = list(getattr(facts, "obstacles", []) or [])
        state_text   = list(getattr(facts, "state_text", []) or [])
        uncertain    = list(getattr(facts, "uncertain", []) or [])

        vlog: Dict[str, Any] = {"objects": [], "pose": None, "obstacles": [], "grasp_points": [], "added_uncertain": []}

        # 1) 对象存在性
        kept_objects = []
        for o in objects:
            attrs = getattr(o, "attributes", {}) or {}
            color = attrs.get("color")
            y = self._ask(image_paths, self.q_object(getattr(o, "name", "object"), color))
            vlog["objects"].append({"name": getattr(o, "name", "object"), "color": color, "decision": y})
            if y == "YES":
                kept_objects.append(o)
            elif y == "UNCERTAIN":
                tag = f"object:{(color+' ') if color else ''}{getattr(o, 'name', 'object')}".strip()
                uncertain.append(tag)
            # NO -> 丢弃

        # 2) 姿态
        pose_v = pose
        if pose:
            target_name = getattr(kept_objects[0], "name", None) if kept_objects else (getattr(objects[0], "name", None) if objects else "target object")
            y = self._ask(image_paths, self.q_pose(target_name or "target object", pose))
            vlog["pose"] = {"value": pose, "target": target_name, "decision": y}
            if y == "NO":
                pose_v = ""
            elif y == "UNCERTAIN":
                uncertain.append(f"pose:{pose}")

        # 3) 障碍
        kept_obstacles = []
        target_name = getattr(kept_objects[0], "name", None) if kept_objects else (getattr(objects[0], "name", None) if objects else "target object")
        for ob in obstacles:
            y = self._ask(image_paths, self.q_obstacle_near(target_name or "target object", ob))
            vlog["obstacles"].append({"name": ob, "decision": y})
            if y == "YES":
                kept_obstacles.append(ob)
            elif y == "UNCERTAIN":
                uncertain.append(f"obstacle:{ob}")

        # 4) 抓取点
        kept_gp = []
        for gp in grasp_points:
            y = self._ask(image_paths, self.q_grasp(target_name or "target object", gp))
            vlog["grasp_points"].append({"name": gp, "decision": y})
            if y == "YES":
                kept_gp.append(gp)
            elif y == "UNCERTAIN":
                uncertain.append(f"grasp:{gp}")

        # 合并 UNCERTAIN（去重）
        seen = set()
        merged_uncertain = []
        for x in uncertain:
            if x and x not in seen:
                merged_uncertain.append(x); seen.add(x)

        # 生成校正后的 FactList（保持原 location/state_text，不随意删除）
        facts_v = FactList(
            objects=kept_objects,
            location=location,
            pose=pose_v,
            grasp_points=kept_gp,
            obstacles=kept_obstacles,
            state_text=state_text,
            uncertain=merged_uncertain
        )
        return facts_v, vlog
