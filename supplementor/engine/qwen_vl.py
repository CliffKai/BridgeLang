# supplementor/engine/qwen_vl.py
from __future__ import annotations

import os
import json, re
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoConfig

# ===== Runtime guards: prefer Flash-Attention v2 when available =====
os.environ.setdefault("PT_SDPA_ENABLE_HEAD_DIM_PADDING", "1")  # allow head-dim padding
try:
    # 开启三种 SDPA 内核，由框架/模型自动选择；装了 FA2 时会优先走 flash-attn v2
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
except Exception:
    pass


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

# —— 从规范化 SUPPLEMENT 兜底构造 FACT_LIST JSON —— 
def _fact_from_structured_supp(supp_text: str) -> dict:
    """
    输入：已被 _normalize_block 规范化后的 SUPPLEMENT 文本
    输出：符合 FACT_LIST schema 的 dict
    """
    def grab(label: str) -> str:
        m = re.search(rf"{re.escape(label)}\s*(.*)", supp_text, flags=re.I)
        return (m.group(1).strip() if m else "")

    # objects: “a, b, c”/“a; b; c”/“a and b” 等都尽量切出来
    objs_line = grab("Target object(s):")
    objs = []
    if objs_line:
        parts = re.split(r"[;,]| and ", objs_line, flags=re.I)
        for name in (p.strip() for p in parts):
            if name:
                objs.append({"name": name, "attributes": {}})

    def split_list(s: str):
        return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]

    return {
        "objects": objs,
        "location": grab("Location:"),
        "pose": grab("Pose/Orientation:"),
        "grasp_points": split_list(grab("Graspable parts (if visible):")),
        "obstacles": split_list(grab("Obstacles/Free space:")),
        "state_text": split_list(grab("State/Labels/Text:")),
        "uncertain": split_list(grab("UNCERTAIN:")),
    }


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

_LABEL_TOKENS = {
    "Target object(s):",
    "Location:",
    "Pose/Orientation:",
    "Graspable parts (if visible):",
    "Obstacles/Free space:",
    "State/Labels/Text:",
    "UNCERTAIN:",
}
def _is_labelish(x: str) -> bool:
    if not isinstance(x, str):
        return False
    t = x.strip()
    # 纯标签或“看起来像标签（以冒号结尾的短语）”
    return (t in _LABEL_TOKENS) or (t.endswith(":") and len(t) <= 40)

def _safe_factlist_json(text: str) -> str:
    """
    将 FACT_LIST 块强制规整为合法 JSON（强类型 + 去标签 + 去空）。
    - 保证 schema 完整
    - 任何“值=标签”的内容视为无效 -> 置空
    - 列表字段中过滤空串/标签串
    """
    try:
        raw = json.loads(text)
    except Exception:
        raw = {}

    # 小工具
    def _s(x):
        return x if isinstance(x, str) else ""

    def _clean_str(x):
        t = _s(x).strip()
        return "" if _is_labelish(t) else t

    def _L(x):
        return x if isinstance(x, list) else []

    def _clean_list_str(xs):
        out = []
        for it in _L(xs):
            if isinstance(it, str):
                t = it.strip()
                if t and not _is_labelish(t):
                    out.append(t)
        return out

    def _objs(xs):
        out = []
        if isinstance(xs, list):
            for o in xs:
                if not isinstance(o, dict):
                    # 兼容把对象名直接写成字符串的情况
                    if isinstance(o, str):
                        name = _clean_str(o)
                        if name:
                            out.append({"name": name, "attributes": {}})
                    continue
                name = _clean_str(o.get("name", ""))
                if not name or _is_labelish(name):
                    continue
                attrs = o.get("attributes")
                attrs = attrs if isinstance(attrs, dict) else {}
                out.append({"name": name, "attributes": attrs})
        return out

    data = {
        "objects": _objs(raw.get("objects", [])),
        "location": _clean_str(raw.get("location", "")),
        "pose": _clean_str(raw.get("pose", "")),
        "grasp_points": _clean_list_str(raw.get("grasp_points", [])),
        "obstacles": _clean_list_str(raw.get("obstacles", [])),
        "state_text": _clean_list_str(raw.get("state_text", [])),
        "uncertain": _clean_list_str(raw.get("uncertain", [])),
    }

    return json.dumps(data, ensure_ascii=False)


# === Pose/Objects/Location 规范化 ===
_POSE_CANON = {
    "upright","lying on side","tilted","handle left","handle right",
    "open","closed","in container","on surface","inside","under","stacked","hanging"
}

def _canon_pose_text(p: str) -> str:
    t = (p or "").strip().lower()
    if not t or t == "uncertain":
        return ""
    hits = []
    def add(tag):
        if tag in _POSE_CANON and tag not in hits:
            hits.append(tag)

    # 同义归一
    if re.search(r"\bupright|standing(\s+upright)?|vertical\b", t): add("upright")
    if re.search(r"\blying\b|on (its|the) side|sideways", t):        add("lying on side")
    if re.search(r"\btilt|angled|slanted", t):                        add("tilted")
    if re.search(r"handle (facing )?left|left[- ]?handle|handle to the left", t):  add("handle left")
    if re.search(r"handle (facing )?right|right[- ]?handle|handle to the right", t): add("handle right")
    if re.search(r"\bopen(ed)?\b", t):                                add("open")
    if re.search(r"\bclosed?\b|shut\b", t):                           add("closed")
    if re.search(r"\bin (a |the )?(pan|pot|bowl|cup|mug|container|box|bin)\b", t):  add("in container")
    if re.search(r"\bon (a |the )?(table|board|surface|tray|plate|stove|stovetop|counter|floor)\b", t): add("on surface")
    if re.search(r"\binside\b", t):                                   add("inside")
    if re.search(r"\bunder(neath)?\b|below\b", t):                    add("under")
    if re.search(r"\bstack(ed|ing)?\b", t):                           add("stacked")
    if re.search(r"\bhang(ing)?\b", t):                               add("hanging")

    return ", ".join(hits[:2])

def _filter_visible_objects(objs):
    """丢弃占位/不确定/泛化名；保留 {name, attributes{}} 结构。"""
    out = []
    if not isinstance(objs, list):
        return out
    for o in objs:
        if not isinstance(o, dict):
            # 兼容字符串对象名
            if isinstance(o, str):
                name = o.strip()
                if name and name.lower() not in {"uncertain","unknown","target","target object","object"}:
                    out.append({"name": name, "attributes": {}})
            continue
        name = str(o.get("name","")).strip()
        if not name or name.lower() in {"uncertain","unknown","target","target object","object"}:
            continue
        attrs = o.get("attributes") if isinstance(o.get("attributes"), dict) else {}
        item = {"name": name, "attributes": attrs}
        if "count" in o:
            item["count"] = o["count"]
        out.append(item)
    return out

def _is_place_phrase(loc: str) -> bool:
    """位置必须是 '地点'；排除 open/left/right 等状态/方向词。"""
    t = (loc or "").strip().lower()
    if not t:
        return False
    if t in {"left","right","open","closed","up","down","front","back"}:
        return False
    return True



def _ensure_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])




def _ensure_nonempty_facts(fact: dict) -> dict:
    """
    只兜底 location/pose/grasp_points；objects 不自动补任务名，只保留可见对象。
    - location/pose 为空或不合规 -> "UNCERTAIN"（并在 uncertain 写原因）
    - pose 若非空，规范化到小词表；失败则置 "UNCERTAIN"
    - grasp_points 为空 -> ["UNCERTAIN"]
    - objects 仅做“可见对象”过滤，不补占位
    """
    if not isinstance(fact, dict):
        fact = {}

    def add_unc(msg):
        if not msg:
            return
        u = fact.get("uncertain")
        if not isinstance(u, list):
            u = []
        if msg not in u:
            u.append(msg)
        fact["uncertain"] = u

    # 读取并统一类型
    objects      = fact.get("objects")
    location     = (fact.get("location") or "").strip()
    pose         = (fact.get("pose") or "").strip()
    grasp_points = fact.get("grasp_points")
    obstacles    = fact.get("obstacles")
    state_text   = fact.get("state_text")
    uncertain    = fact.get("uncertain")

    if not isinstance(obstacles, list):  obstacles = []
    if not isinstance(state_text, list): state_text = []
    if not isinstance(uncertain, list):  uncertain = []

    # 对象：仅保留“可见对象”
    objects = _filter_visible_objects(objects)

    # 位置：必须是地点
    if not location or not _is_place_phrase(location):
        location = "UNCERTAIN"
        add_unc("location:not visible or not a place")

    # 姿态：规范到小词表
    if not pose:
        pose = "UNCERTAIN"
        add_unc("pose:not visible")
    else:
        p2 = _canon_pose_text(pose)
        if p2:
            pose = p2
        else:
            pose = "UNCERTAIN"
            add_unc("pose:out of vocabulary")

    # 抓取点：至少一个
    if not isinstance(grasp_points, list) or len([g for g in grasp_points if (g or "").strip()]) == 0:
        grasp_points = ["UNCERTAIN"]
        add_unc("grasp_points:not visible")

    return {
        "objects": objects,
        "location": location,
        "pose": pose,
        "grasp_points": grasp_points,
        "obstacles": obstacles,
        "state_text": state_text,
        "uncertain": fact.get("uncertain", []),
    }




def _empty_fact():
    return {
        "objects": [],
        "location": "",
        "pose": "",
        "grasp_points": [],
        "obstacles": [],
        "state_text": [],
        "uncertain": [],
    }


# —— 统一 schema -> “永不留空”强制器 -> 生成简短 SUPPLEMENT —— #

def _to_fact_dict(text: str) -> dict:
    """将 FACT_LIST 文本安全转为 dict，并补齐全部键（允许为空）。"""
    try:
        data = json.loads(text)
    except Exception:
        data = {}
    def _s(x): return x if isinstance(x, str) else ""
    def _L(x): return x if isinstance(x, list) else []
    def _objs(x):
        out = []
        if isinstance(x, list):
            for o in x:
                if isinstance(o, dict):
                    name = _s(o.get("name", ""))
                    attrs = o.get("attributes") if isinstance(o.get("attributes"), dict) else {}
                    out.append({"name": name, "attributes": attrs})
        return out
    return {
        "objects": _objs(data.get("objects", [])),
        "location": _s(data.get("location", "")),
        "pose": _s(data.get("pose", "")),
        "grasp_points": _L(data.get("grasp_points", [])),
        "obstacles": _L(data.get("obstacles", [])),
        "state_text": _L(data.get("state_text", [])),
        "uncertain": _L(data.get("uncertain", [])),
    }


_LABEL_LINE = re.compile(r"^\s*(Target object\(s\)|Location|Pose/Orientation|Graspable parts\s*\(if visible\)|Obstacles/Free space|State/Labels/Text|UNCERTAIN)\s*:\s*$", re.I | re.M)

def _compose_prose_from_facts(f: dict) -> str:
    """
    由 facts 合成 1-2 句简洁 SUPPLEMENT 文本。
    避免把 'UNCERTAIN' 当作事实，只在需要时提一嘴不确定。
    """
    names = ", ".join([o.get("name","") for o in f.get("objects", []) if o.get("name")]) or "target object(s)"
    loc   = f.get("location", "")
    pose  = f.get("pose", "")
    gps   = ", ".join(f.get("grasp_points", [])) if isinstance(f.get("grasp_points", []), list) else str(f.get("grasp_points", ""))

    parts1 = []
    if names and names.upper() != "UNCERTAIN":
        parts1.append(names)
    else:
        parts1.append("target object(s)")
    if loc and loc.upper() != "UNCERTAIN":
        parts1.append(f"at {loc}")
    sent1 = " ".join(parts1).strip().capitalize() + "."

    parts2 = []
    if pose and pose.upper() != "UNCERTAIN":
        parts2.append(f"Pose: {pose}.")
    if gps and gps.upper() != "UNCERTAIN":
        parts2.append(f"Grasp points: {gps}.")
    if not parts2:
        parts2.append("Some details are UNCERTAIN from the image.")
    return " ".join([sent1] + parts2)

def _clean_and_maybe_rewrite_supp(supp_body: str, fact: dict, target_max_tokens: int, processor) -> str:
    """
    清洗 SUPPLEMENT：
    - 去掉纯标签行/空行
    - 过滤指令化句子
    - 若清洗后过短或像模板，就用 facts 合成 1-2 句
    - 截断到目标 token 上限
    """
    t = supp_body or ""
    # 去内嵌标签与多余空白
    t = _normalize_block(t)  # 已有函数：去内嵌标签、压空白等
    # 去掉形如 “Location:” 这种空标签行
    t = _LABEL_LINE.sub("", t).strip()
    # 丢掉“指令/复述”句子
    t = _strip_instructional_sentences(t)
    # 若太短或仍像模板，就用 facts 合成
    if len(t) < 8 or re.search(r"Target object|Pose/Orientation|Graspable parts|Obstacles/Free space", t, re.I):
        t = _compose_prose_from_facts(fact)
    # 最终清洁
    t = _clean_supplement(t)
    # 截断
    return _truncate_tokens(processor, t, target_max_tokens)


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

        # 先备好 config，并指定使用 Flash-Attention v2
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=self.offline)
        try:
            cfg._attn_implementation = "flash_attention_2"
        except Exception:
            pass

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            config=cfg,                   # ← 关键：把带 FA2 指定的 config 传进去
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

    def _postprocess(self, raw_text: str, target_max_tokens: int = 120, task: str | None = None) -> str:
        fact_body = _extract_block(_FACT_RE, raw_text)
        supp_body = _extract_block(_SUP_RE,  raw_text)

        # 1) 解析 FACT_LIST：用“安全 JSON 化”兜底
        if fact_body is None:
            fact = _empty_fact()
        else:
            safe = _safe_factlist_json(fact_body)
            try:
                fact = json.loads(safe)
            except Exception:
                fact = _empty_fact()

        # 2) 统一强制不留空 + 规范化
        fact = _ensure_nonempty_facts(fact)

        # 3) SUPPLEMENT 清洗/兜底/截断
        if supp_body is None:
            supp_body = ""
        supp_body = _clean_and_maybe_rewrite_supp(supp_body, fact, target_max_tokens, self.processor)

        # 4) 回写为文本块
        fact_str = json.dumps(fact, ensure_ascii=False)
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
        do_sample = (self.temperature > 0.0)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            repetition_penalty=self.repetition_penalty,
            bad_words_ids=self.bad_words_ids,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(dict(temperature=self.temperature, top_p=self.top_p))

        out_ids = self.model.generate(**inputs, **gen_kwargs)


        # 只取新生成的部分并解码
        new_ids = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        raw = self.processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if os.getenv("SUPP_DEBUG", "0") == "1":
            print("=== RAW_DRAFT ===\n", raw)

        text = self._postprocess(raw, target_max_tokens=req.max_tokens, task=req.task)
        return SupplementResult(text=text, score=1.0, flags=[])

