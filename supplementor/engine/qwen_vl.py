# supplementor/engine/qwen_vl.py
from __future__ import annotations

import os
import json, re
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ===== Runtime guards: fix head_dim=80 with IXDNN/Flash path =====
# Enable head-dim padding and prefer math SDPA to avoid flash-80-dim crash.
os.environ.setdefault("PT_SDPA_ENABLE_HEAD_DIM_PADDING", "1")
try:
    torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    )
except Exception:
    pass  # Non-CUDA / older torch — safe to ignore

# qwen_vl_utils is optional; we fallback to local parser if missing
try:
    from qwen_vl_utils import process_vision_info as _process_vision_info  # type: ignore
except Exception:
    _process_vision_info = None  # fallback later

from ..api import SupplementRequest, SupplementResult

_FACT_RE = re.compile(r"\[FACT_LIST\](.*?)\[/FACT_LIST\]", re.S)
_SUP_RE = re.compile(r"\[SUPPLEMENT\](.*?)\[/SUPPLEMENT\]", re.S)

# ban common "instructional/restatement" phrases to keep pure visible facts
_BAN = re.compile(
    r"\b(needs to|should|effectively|in order to|perform an action|must|required to)\b",
    re.I,
)

# === 在文件顶端已有的 import 和 _SUP_RE 后面粘贴 ===
_BAN = re.compile(
    r"\b(needs to|should|effectively|in order to|perform an action|must|required to)\b",
    re.I,
)

_BAD_SENTENCE = re.compile(
    r"\b(robot|reach out|should|needs to|ensure|in order to|manipulat|transport|plan to)\b",
    re.I,
)


def _strip_instructional_sentences(t: str) -> str:
    # 句号/问号/感叹号断句；丢掉包含“指令/复述触发词”的句子
    parts = re.split(r'(?<=[.!?])\s+', t.strip())
    keep = [p for p in parts if p and not _BAD_SENTENCE.search(p)]
    return " ".join(keep).strip()

def _clean_supplement(text: str) -> str:
    t = text.strip()
    t = _BAN.sub("", t)                 # 删除常见的“建议式/复述”词
    t = re.sub(r"\s+", " ", t)          # 压缩多余空白
    t = re.sub(r"\s([.,;:])", r"\1", t) # 去掉标点前的空格
    return t.strip()

# —— 规范化 SUPPLEMENT 内部文本（去内嵌标签、统一大小写等）——
def _normalize_block(body: str) -> str:
    """
    规范化 SUPPLEMENT 内部文本：
    - 去掉任何大小写混用的 [supplement] 内嵌标签
    - 统一 UNCERTAIN: 大写
    - 统一常用字段名
    - 强制每个字段换行
    - 压缩空白但保留换行
    """
    t = body.strip()

    # 1) 去掉内嵌 [supplement] 标签（大小写、是否带斜杠都匹配）
    t = re.sub(r"\[/?\s*supplement\s*\]", "", t, flags=re.I)

    # 2) 统一 UNCERTAIN 大写
    t = re.sub(r"\buncertain\s*:", "UNCERTAIN:", t, flags=re.I)

    # 3) 字段名归一（容错大小写与多余空格）
    field_map = {
        r"\bTarget object\(s\)\s*:\s*": "Target object(s): ",
        r"\bLocation\s*:\s*": "Location: ",
        r"\bPose/Orientation\s*:\s*": "Pose/Orientation: ",
        r"\bGraspable parts\s*\(if visible\)\s*:\s*": "Graspable parts (if visible): ",
        r"\bObstacles/Free space\s*:\s*": "Obstacles/Free space: ",
        r"\bState/Labels/Text\s*:\s*": "State/Labels/Text: ",
        r"\bUNCERTAIN\s*:\s*": "UNCERTAIN: ",
    }
    for pat, rep in field_map.items():
        t = re.sub(pat, rep, t, flags=re.I)

    # 4) 强制每个字段独立成行（给每个字段名前面加换行；首行再去掉多余换行）
    labels = [
        "Target object(s):", "Location:", "Pose/Orientation:",
        "Graspable parts (if visible):", "Obstacles/Free space:",
        "State/Labels/Text:", "UNCERTAIN:"
    ]
    for label in labels:
        # 任意空白开头接 label -> 换行 + label；否则如果已经在行首就不处理
        t = re.sub(rf"\s*{re.escape(label)}\s*", f"\n{label} ", t)
    t = t.lstrip("\n")  # 去掉首部多余换行

    # 5) 压缩空白但保留换行：只合并行内的空格，不合并 '\n'
    t = re.sub(r"[ \t]+", " ", t)       # 合并一行内多空格
    t = re.sub(r"[ \t]+\n", "\n", t)    # 去掉行尾空格
    t = re.sub(r"\n{3,}", "\n\n", t)    # 多个空行压到最多一个
    t = re.sub(r"\s([.,;:])", r"\1", t) # 去掉标点前空格

    return t.strip()

def _truncate_tokens(processor, text: str, target_max_tokens: int = 120) -> str:
    toks = processor.tokenizer.encode(text, add_special_tokens=False)
    if len(toks) <= target_max_tokens:
        return text.strip()
    toks = toks[:target_max_tokens]
    return processor.tokenizer.decode(toks, skip_special_tokens=True).strip()

def _extract_block(pattern: re.Pattern, raw: str) -> str | None:
    """在 raw 文本里抓取第一个匹配的方块内容；没有则返回 None。"""
    m = pattern.search(raw)
    return m.group(1).strip() if m else None

def _extract_images_from_messages(messages: List[dict]) -> List[Image.Image]:
    """Fallback parser when qwen_vl_utils is unavailable (image-only)."""
    imgs: List[Image.Image] = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content", [])
        if not isinstance(content, list):
            continue
        for seg in content:
            if isinstance(seg, dict) and seg.get("type") == "image":
                src = seg.get("image")
                if isinstance(src, Image.Image):
                    imgs.append(src.convert("RGB"))
                elif isinstance(src, str):
                    path = src[7:] if src.startswith("file://") else src
                    imgs.append(Image.open(path).convert("RGB"))
    return imgs

def _safe_factlist_json(text: str) -> str:
    """确保 FACT_LIST 内是合法 JSON，并补齐所有键。"""
    try:
        data = json.loads(text)
    except Exception:
        data = {}
    # 强制补齐 schema
    def _s(x): return x if isinstance(x, str) else ""
    def _L(x): return x if isinstance(x, list) else []
    def _objs(x):
        out = []
        if isinstance(x, list):
            for o in x:
                if not isinstance(o, dict): continue
                name = _s(o.get("name",""))
                attrs = o.get("attributes") if isinstance(o.get("attributes"), dict) else {}
                out.append({"name": name, "attributes": attrs})
        return out
    data = {
        "objects": _objs(data.get("objects", [])),
        "location": _s(data.get("location","")),
        "pose": _s(data.get("pose","")),
        "grasp_points": _L(data.get("grasp_points", [])),
        "obstacles": _L(data.get("obstacles", [])),
        "state_text": _L(data.get("state_text", [])),
        "uncertain": _L(data.get("uncertain", [])),
    }
    return json.dumps(data, ensure_ascii=False)

class QwenVLSupplementor:
    """
    Qwen/Qwen2.5-VL 本地推理引擎（Transformers）
    - 支持离线加载
    - 自动清洗与长度控制
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        prompt_path: str = "prompts/supplementor_en.txt",
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        max_new_tokens: int = 160,  # 上限（含标签），正文我们再控到 ~120
        temperature: float = 0.0,   # 起草更稳，默认贪心；需要多样性可设 0.2
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        offline: bool = False,      # ← 离线模式
        use_bnb_8bit: bool = False, # ← 可选 8bit 量化
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.offline = offline

        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

        dtype = torch.bfloat16 if torch_dtype in ("auto", "bfloat16", "bf16") else torch.float16

        quantization_config = None
        if use_bnb_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                dtype = None  # 量化时由 quantization_config 决定精度
            except Exception as e:
                raise RuntimeError(
                    "bitsandbytes 未安装或不可用；请先 `pip install bitsandbytes`"
                ) from e

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            local_files_only=self.offline,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=self.offline
        )
        
        # 这些是很容易导致“指令/复述”的触发词；注意前导空格避免误伤词干
        bad_words = [" robot", " Robot", " should", " needs", " ensure", " reach", " plan to"]
        self.bad_words_ids = [
            self.processor.tokenizer.encode(w, add_special_tokens=False) for w in bad_words
        ]
        # 过滤空列表（极少数 tokenizer 可能把短片段编码为空）
        self.bad_words_ids = [ids for ids in self.bad_words_ids if ids]


    def _build_messages(self, image_paths: List[str], task: str) -> list[dict]:
        content = [{"type": "image", "image": f"file://{p}"} for p in image_paths]
        content.append({"type": "text", "text": (
            f"{task}\n\n"
            "Return TWO blocks in this exact order:\n"
            "[FACT_LIST]{JSON with keys: objects(list of {name, attributes{color?}}), "
            "location, pose, grasp_points(list), obstacles(list), state_text(list), uncertain(list)}[/FACT_LIST]\n"
            "[SUPPLEMENT]Concise, task-aligned description[/SUPPLEMENT]"
        )})
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]

    def _postprocess(self, raw_text: str, target_max_tokens: int = 120) -> str:
        fact_body = _extract_block(_FACT_RE, raw_text)
        supp_body = _extract_block(_SUP_RE,  raw_text)

        # 1) FACT_LIST 兜底
        if fact_body is None:
            # 没有 FACT_LIST，就给个空表占位（方便 pipeline 不中断）
            fact_json = {
                "objects": [], "location":"", "pose":"", "grasp_points":[],
                "obstacles": [], "state_text": [], "uncertain": []
            }
            fact_str = json.dumps(fact_json, ensure_ascii=False)
        else:
            fact_str = _safe_factlist_json(fact_body)

        # 2) SUPPLEMENT 兜底 + 截断
        if supp_body is None:
            supp_body = "No additional details beyond the task description."
        supp_body = _truncate_tokens(self.processor, supp_body, target_max_tokens)

        return f"[FACT_LIST]{fact_str}[/FACT_LIST]\n[SUPPLEMENT]{supp_body}[/SUPPLEMENT]"

    @torch.inference_mode()
    def infer(self, req: SupplementRequest) -> SupplementResult:
        messages = self._build_messages(req.images, req.task)

        # 文本模板
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 解析多模态输入：优先官方工具；无则走本地 fallback（图片）
        if _process_vision_info is not None:
            image_inputs, video_inputs = _process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            images = _extract_images_from_messages(messages)
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)

        # —— 关键改动：按需传采样参数，避免 do_sample=False 时仍传 temperature/top_p —— #
        do_sample = (self.temperature is not None) and (float(self.temperature) > 0.0)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            bad_words_ids=self.bad_words_ids,  # 硬屏蔽禁词
            eos_token_id=self.processor.tokenizer.eos_token_id,
            # pad_token_id 可不传；若要更稳，可以显式等于 eos：
            # pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=float(self.temperature),
                top_p=float(self.top_p),
            )
        else:
            gen_kwargs.update(do_sample=False)

        out_ids = self.model.generate(**gen_kwargs)

        # 只取新生成的部分并解码
        new_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        raw = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if os.getenv("SUPP_DEBUG", "0") == "1":
            print("=== RAW_DRAFT ===\n", raw)

        text = self._postprocess(raw, target_max_tokens=req.max_tokens)
        return SupplementResult(text=text, score=1.0, flags=[])

