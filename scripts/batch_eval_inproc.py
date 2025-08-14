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
os.environ.setdefault("PT_SDPA_ENABLE_HEAD_DIM_PADDING", "1")
try:
    torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_mem_efficient=False, enable_math=True
    )
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

# === 替换原来的 _build_checks ===
def _build_checks(facts_after: dict) -> dict:
    """
    从主裁判确认后的 facts_after 生成一组“待复核”的 YES/NO 问题。
    - 默认不带颜色限定（可通过 ALT_RELAX_COLOR 控制）
    - 位置/姿态短语更自然
    - 抓取点只问是否可见（不问 suitable，以减少主观性）
    """
    checks = {"objects": [], "pose": None, "location": None, "grasp_points": []}

    relax_color = bool(int(os.getenv("ALT_RELAX_COLOR", "1")))

    # objects
    for o in facts_after.get("objects", []) or []:
        name = (o.get("name") or "").strip()
        color = (o.get("attributes", {}) or {}).get("color")
        desc = _object_phrase(name, color, relax_color)
        if desc:
            q = f"Is there a {desc} visible in the image?"
            checks["objects"].append({"name": name, "color": None if relax_color else color, "question": q})

    # 选一个 target 描述
    target = None
    if checks["objects"]:
        c0 = checks["objects"][0]
        target = c0["name"].strip()

    # location
    loc_raw = facts_after.get("location") or ""
    loc = _normalize_loc(loc_raw)
    if loc and target:
        checks["location"] = {
            "value": loc,
            "question": f"Is the {target} {loc}?",
        }

    # pose
    pose_raw = facts_after.get("pose") or ""
    pose = _normalize_pose(pose_raw)
    if pose and target:
        checks["pose"] = {
            "value": pose,
            "question": f"Is the {target} {pose}?",
        }

    # grasp points
    for gp in (facts_after.get("grasp_points") or []):
        gp = (gp or "").strip()
        if gp and target:
            q = f"Is the {gp} of the {target} visible?"
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
                    ((o.attributes.get("color")+" "+o.name).strip()
                     if getattr(o, "attributes", {}) and o.attributes.get("color") else o.name)
                    for o in getattr(facts_v, "objects", [])
                ]) or ""
                lines = [
                    f"Target object(s): {names}".strip(),
                    f"Location: {getattr(facts_v, 'location','')}".strip(),
                    f"Pose/Orientation: {getattr(facts_v, 'pose','')}".strip(),
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
