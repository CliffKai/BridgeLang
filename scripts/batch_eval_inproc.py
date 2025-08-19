#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-process batch evaluator with optional heterogeneous cross-check (e.g., LLaVA v1.6-34B).
- 只加载一次 Draft(3B)/Verifier(72B)
- 可选：加载异构复核模型，对关键事实逐条 YES/NO/UNCERTAIN 抽检
- 产出 JSONL，包含 alt_review 与 disagreement
"""

import argparse, csv, json, os, sys, time
from pathlib import Path

# —— 避免某些 flash 内核在 80 维等场景下崩溃，统一偏好 math SDPA —— #
import torch

os.environ.setdefault("PYTORCH_SDP_DISABLE_FLASH", "1")
os.environ.setdefault("PYTORCH_SDP_DISABLE_MEM_EFFICIENT", "1")

# 让 PyTorch SDPA 支持 head_dim padding（保险起见，FA2 也不冲突）
os.environ.setdefault("PT_SDPA_ENABLE_HEAD_DIM_PADDING", "1")
# 不再屏蔽 flash；三种内核都打开，由模型自行选择（FA2 时这行其实不起作用）
try:
    torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
except Exception:
    pass

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

from supplementor.api import SupplementRequest
from supplementor.engine.qwen_vl import QwenVLSupplementor
from supplementor.factlist import parse_factlist_from_output, FactList
from supplementor.consistency import check_format
from supplementor.verifier.verifier_vl import VisualVerifier, VerifyConfig


def to_dict_safe(facts: FactList|None):
    if not facts:
        return {"objects": [], "location": "", "pose": "", "grasp_points": [], "obstacles": [], "state_text": [], "uncertain": []}
    return {
        "objects": [ {"name": o.name, "attributes": getattr(o, "attributes", {}), "count": getattr(o, "count", None)} for o in getattr(facts, "objects", []) ],
        "location": getattr(facts, "location", ""),
        "pose": getattr(facts, "pose", ""),
        "grasp_points": getattr(facts, "grasp_points", []),
        "obstacles": getattr(facts, "obstacles", []),
        "state_text": getattr(facts, "state_text", []),
        "uncertain": getattr(facts, "uncertain", []),
    }


# ---------- Heterogeneous cross-check (generic VL YES/NO) ----------

def _ynu(s: str) -> str:
    t = (s or "").strip().upper()
    if "YES" in t and "NO" not in t: return "YES"
    if "NO" in t and "YES" not in t: return "NO"
    if "UNCERTAIN" in t: return "UNCERTAIN"
    # fallback：首词截断
    for tok in t.replace(".", " ").split():
        if tok in ("YES","NO","UNCERTAIN"): return tok
    return "UNCERTAIN"


class GenericVLYesNo:
    """
    最小可用的多模态 YES/NO 验证器：
    - AutoProcessor + (LLaVA 系列优先，否则回退 AutoModelForCausalLM)
    - 使用 chat_template；若模型不支持，会抛异常（外层捕获并跳过）
    - 仅用于“异构复核”，逐条问：Answer ONLY: YES or NO or UNCERTAIN.
    """
    def __init__(self, model_id: str, offline: bool = False, dtype: str = "bfloat16", use_fast: bool = True):
        self.model_id = model_id
        self.offline = offline

        torch_dtype = torch.bfloat16 if dtype in ("bfloat16","bf16","auto") else torch.float16

        # 识别是否为 LLaVA 家族
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=offline)
        # 指定使用 Flash-Attention v2
        try:
            cfg._attn_implementation = "flash_attention_2"
        except Exception:
            pass

        model_type = getattr(cfg, "model_type", None)

        if model_type == "llava_next":
            from transformers import LlavaNextForConditionalGeneration as _Class
        elif model_type == "llava":
            from transformers import LlavaForConditionalGeneration as _Class
        else:
            _Class = AutoModelForCausalLM  # 其它家族尝试通用加载

        self.model = _Class.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch_dtype,
            local_files_only=offline,
            config=cfg,  
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=offline, use_fast=use_fast
        )

        # 采样开关（可用环境变量控制）
        self.sample = bool(int(os.getenv("ALT_YN_SAMPLE","0")))
        self.temperature = float(os.getenv("ALT_YN_TEMPERATURE","0.2"))
        self.top_p = float(os.getenv("ALT_YN_TOP_P","0.9"))

    @torch.inference_mode()
    def ask(self, image_path: str, q: str, force_sample: bool|None=None) -> str:
        img = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a strict visual fact verifier. Answer ONLY: YES or NO or UNCERTAIN."},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text": q + "\nAnswer ONLY: YES or NO or UNCERTAIN."}
            ]},
        ]
        # 需要 chat template 支持多模态
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt").to(self.model.device)

        use_sample = self.sample if force_sample is None else bool(force_sample)
        gen_kwargs = dict(max_new_tokens=16, eos_token_id=self.processor.tokenizer.eos_token_id)
        if use_sample:
            gen_kwargs.update(dict(do_sample=True, temperature=self.temperature, top_p=self.top_p))
        else:
            gen_kwargs.update(dict(do_sample=False))

        out = self.model.generate(**inputs, **gen_kwargs)
        dec = self.processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        return _ynu(dec)


# === 放在文件顶端或 _build_checks 前：一些短语规范化工具 ===
def _normalize_loc(loc: str) -> str:
    t = (loc or "").strip().lower()
    # 更口语化
    repl = {
        "on table": "on the table",
        "on a table": "on the table",
        "at table": "on the table",
        "towards the table": "toward the table",
        "towards table": "toward the table",
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    return t

def _normalize_pose(pose: str) -> str:
    t = (pose or "").strip().lower()
    # 常见姿态短语 -> 更自然的表达
    t = t.replace("angled towards the", "tilted toward the")
    t = t.replace("angled toward the", "tilted toward the")
    t = t.replace("angled towards", "tilted toward")
    t = t.replace("angled toward", "tilted toward")
    t = t.replace("pointing at", "pointing to")
    t = t.replace("towards the", "toward the")
    return t

def _object_phrase(name: str, color: str|None, relax_color: bool) -> str:
    name = (name or "").strip()
    color = (color or "").strip() or None
    if not name:
        return ""
    if relax_color or not color:
        return name
    return f"{color} {name}".strip()


def _normalize_loc(loc: str) -> str:
    t = (loc or "").strip().lower()
    # 更自然的口语化同义
    repl = {
        "on table": "on the table",
        "on a table": "on the table",
        "at table": "on the table",
        "on countertop": "on the countertop",
        "on counter": "on the counter",
        "on top of stove": "on the stove",
        "on top of the stove": "on the stove",
        "countertop next to sink": "on the countertop next to the sink",
        "countertop": "on the countertop",  # 常见短写
        "stove top": "on the stove",
        "towards the table": "toward the table",
        "towards table": "toward the table",
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    # 去重空格
    t = " ".join(t.split())
    return t

def _normalize_pose(pose: str) -> str:
    t = (pose or "").strip().lower()
    t = t.replace("angled towards the", "tilted toward the")
    t = t.replace("angled toward the", "tilted toward the")
    t = t.replace("angled towards", "tilted toward")
    t = t.replace("angled toward", "tilted toward")
    t = t.replace("pointing at", "pointing to")
    t = t.replace("towards the", "toward the")
    t = " ".join(t.split())
    return t

def _merge_color_name(name: str, color: str|None) -> str:
    """把颜色与名称合并，避免 yellow yellow block 等重复；大小写/前缀安全。"""
    nn = (name or "").strip()
    cc = (color or "").strip() if color else ""
    if not cc:
        return nn
    low = nn.lower()
    ccl = cc.lower()
    # 如果名称已经以颜色开头，或颜色词已在名称的首两个 token 中，就不再重复
    toks = low.split()
    if low.startswith(ccl + " ") or (toks and ccl in toks[:2]):
        return nn
    return f"{cc} {nn}".strip()


def _build_checks(facts_after: dict) -> dict:
    """
    从主裁判确认后的 facts_after 生成一组待复核问题。
    环境变量控制开关：
      ALT_SKIP_OBJECTS=1   跳过 objects
      ALT_SKIP_LOCATION=1  跳过 location
      ALT_SKIP_POSE=1      跳过 pose
      ALT_SKIP_GP=1        跳过 grasp_points
      ALT_RELAX_COLOR=1    问 objects 时不强制颜色（默认 1）
    """
    skip_obj = bool(int(os.getenv("ALT_SKIP_OBJECTS", "0")))
    skip_loc = bool(int(os.getenv("ALT_SKIP_LOCATION", "0")))
    skip_pose = bool(int(os.getenv("ALT_SKIP_POSE", "0")))
    skip_gp  = bool(int(os.getenv("ALT_SKIP_GP", "0")))
    relax_color = bool(int(os.getenv("ALT_RELAX_COLOR", "1")))

    checks = {"objects": [], "pose": None, "location": None, "grasp_points": []}

    # objects
    targets = []
    if not skip_obj:
        for o in facts_after.get("objects", []) or []:
            name = (o.get("name") or "").strip()
            color = (o.get("attributes", {}) or {}).get("color")
            desc = _object_phrase(name, color, relax_color)
            if desc:
                q = f"Is there a {desc} visible in the image?"
                checks["objects"].append({"name": name, "color": None if relax_color else color, "question": q})
                targets.append(name)

    # 选一个 target（只用名称，避免颜色过度约束）
    target = targets[0] if targets else None

    # location
    if not skip_loc:
        loc_raw = facts_after.get("location") or ""
        loc = _normalize_loc(loc_raw)
        if loc and target:
            checks["location"] = {
                "value": loc,
                "question": f"Is the {target} {loc}?",
            }

    # pose
    if not skip_pose:
        pose_raw = facts_after.get("pose") or ""
        pose = _normalize_pose(pose_raw)
        if pose and target:
            checks["pose"] = {
                "value": pose,
                "question": f"Is the {target} {pose}?",
            }

    # grasp points（只问“是否可见”，不问“是否适合抓取”）
    if not skip_gp and target:
        for gp in (facts_after.get("grasp_points") or []):
            gp = (gp or "").strip()
            if gp:
                q = f"Is the {target}'s {gp} visible?"
                checks["grasp_points"].append({"name": gp, "question": q})

    return checks


    
def run_alt_review(alt: GenericVLYesNo, image: str, facts_after: dict, force_sample: bool|None=None) -> dict:
    """
    用异构模型对关键事实逐条复核；返回 alt_review 结构与 disagreement 标记。
    - 默认只在 NO 时记为分歧；可用 ALT_COUNT_UNCERTAIN=1 把 UNCERTAIN 也当分歧
    """
    checks = _build_checks(facts_after)

    verdicts = {
        "objects": [],
        "location": None,
        "pose": None,
        "grasp_points": [],
    }
    disagree = False
    count_uncertain = bool(int(os.getenv("ALT_COUNT_UNCERTAIN", "0")))

    def _is_disagree(dec: str) -> bool:
        if dec == "NO":
            return True
        if dec == "UNCERTAIN" and count_uncertain:
            return True
        return False

    # objects
    for c in checks["objects"]:
        dec = alt.ask(image, c["question"], force_sample=force_sample)
        verdicts["objects"].append({"name": c["name"], "color": c.get("color"), "alt_decision": dec})
        if _is_disagree(dec):
            disagree = True

    # location
    if checks["location"]:
        dec = alt.ask(image, checks["location"]["question"], force_sample=force_sample)
        verdicts["location"] = {"value": checks["location"]["value"], "alt_decision": dec}
        if _is_disagree(dec):
            disagree = True

    # pose
    if checks["pose"]:
        dec = alt.ask(image, checks["pose"]["question"], force_sample=force_sample)
        verdicts["pose"] = {"value": checks["pose"]["value"], "alt_decision": dec}
        if _is_disagree(dec):
            disagree = True

    # grasp points
    for g in checks["grasp_points"]:
        dec = alt.ask(image, g["question"], force_sample=force_sample)
        verdicts["grasp_points"].append({"name": g["name"], "alt_decision": dec})
        if _is_disagree(dec):
            disagree = True

    return {
        "verdicts": verdicts,
        "disagreement": disagree,
    }


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with headers: image,task")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model_id", default="/mnt/BridgeLang/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--verify_model_id", default="/mnt/BridgeLang/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--with_verify", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=120)

    # --- 异构复核 ---
    ap.add_argument("--with_alt_verify", action="store_true")
    ap.add_argument("--alt_verify_model_id", default="/mnt/BridgeLang/LLaVA-v1.6-34B")
    ap.add_argument("--alt_sample", action="store_true", help="let alt do sampling (optional)")

    args = ap.parse_args()

    # Load main draft & verifier once
    engine = QwenVLSupplementor(model_id=args.model_id, offline=args.offline)
    verifier = VisualVerifier(VerifyConfig(model_id=args.verify_model_id, offline=args.offline)) if args.with_verify else None

    # ALT verifier init (once)
    alt = None
    alt_status = None
    if args.with_alt_verify:
        try:
            alt = GenericVLYesNo(model_id=args.alt_verify_model_id, offline=args.offline)
            # CLI 覆盖环境变量
            if args.alt_sample:
                alt.sample = True
            alt_status = "ALT_VERIFIER_READY"
        except Exception as e:
            alt = None
            alt_status = f"ALT_VERIFIER_UNAVAILABLE: {repr(e)}"

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()

    # 读 manifest
    with open(args.manifest, "r", encoding="utf-8") as f_in, open(outp, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            n += 1
            image = row["image"].strip()
            task = row["task"].strip()

            t_sample = time.time()
            req = SupplementRequest(images=[image], task=task, lang="auto", max_tokens=args.max_tokens)
            draft = engine.infer(req)
            out_text = draft.text
            flags = ["DRAFT"]

            facts = parse_factlist_from_output(draft.text)
            facts_before = to_dict_safe(facts)

            vlog = None
            facts_after = None
            if verifier is not None:
                facts_v, vlog = verifier.verify([image], facts if facts else FactList())
                flags.append("VERIFIED")
                # 规则式回写
                names = ", ".join([
                    _merge_color_name(
                        o.name,
                        (getattr(o, "attributes", {}) or {}).get("color")
                    )
                    for o in getattr(facts_v, "objects", [])
                ]) or ""

                loc_norm = _normalize_loc(getattr(facts_v, 'location','') or "")
                pose_norm = _normalize_pose(getattr(facts_v, 'pose','') or "")

                lines = [
                    f"Target object(s): {names}".strip(),
                    f"Location: {loc_norm}".strip(),
                    f"Pose/Orientation: {pose_norm}".strip(),
                    f"Graspable parts (if visible): {', '.join(getattr(facts_v,'grasp_points',[]) or [])}".strip(),
                    f"Obstacles/Free space: {', '.join(getattr(facts_v,'obstacles',[]) or [])}".strip(),
                    f"State/Labels/Text: {', '.join(getattr(facts_v,'state_text',[]) or [])}".strip(),
                    f"UNCERTAIN: {', '.join(getattr(facts_v,'uncertain',[]) or [])}".strip(),
                ]
                out_text = "[SUPPLEMENT]" + "\n".join(lines) + "[/SUPPLEMENT]"
                facts_after = to_dict_safe(facts_v)

            # --- 异构复核 ---
            alt_review = None
            disagreement = None
            if alt is not None and facts_after:
                try:
                    rr = run_alt_review(alt, image, facts_after, force_sample=args.alt_sample)
                    alt_review = {"model_id": args.alt_verify_model_id, **rr}
                    disagreement = bool(rr.get("disagreement", False))
                except Exception as e:
                    alt_review = {"model_id": args.alt_verify_model_id, "error": repr(e)}
                    flags.append("ALT_REVIEW_ERROR")
            elif args.with_alt_verify and alt is None:
                flags.append(alt_status or "ALT_VERIFIER_UNAVAILABLE")

            ok, extra_flags = check_format(out_text)
            if not ok: flags += extra_flags

            rec = {
                "images": [image],
                "task": task,
                "engine_model_id": args.model_id,
                "verify_model_id": args.verify_model_id if verifier else None,
                "alt_verify_model_id": args.alt_verify_model_id if args.with_alt_verify else None,
                "flags": flags,
                "facts_before": facts_before,
                "facts_after": facts_after,
                "vlog": vlog,
                "supplement": out_text,
                "elapsed_s": time.time() - t_sample,
                "alt_review": alt_review,
                "disagreement": disagreement,  # 可能为 True/False 或 None
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            print(f"[{n}] done {image}  time={rec['elapsed_s']:.1f}s  flags={flags}"
                  + (f"  disagreement={disagreement}" if disagreement is not None else ""))

    print(f"ALL DONE. samples={n}  total_time={time.time()-t0:.1f}s  out={outp}")


if __name__ == "__main__":
    main()
