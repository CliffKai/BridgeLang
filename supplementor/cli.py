# supplementor/cli.py
import argparse, json, os
from dataclasses import asdict, is_dataclass

from .api import SupplementRequest
from .engine.qwen_stub import QwenVLSupplementorStub
from .engine.qwen_vl import QwenVLSupplementor
from .consistency import check_format
from .factlist import parse_factlist_from_output, FactList
from .verifier.verifier_vl import VisualVerifier, VerifyConfig


def _to_dict_safe(x):
    if x is None:
        return None
    try:
        if is_dataclass(x):
            return asdict(x)
    except Exception:
        pass
    try:
        return x.to_dict()  # 若你在 FactList 里实现了 to_dict
    except Exception:
        pass
    try:
        return json.loads(json.dumps(x, ensure_ascii=False))
    except Exception:
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--prompt", default="prompts/supplementor_en.txt")
    ap.add_argument("--max_tokens", type=int, default=120)

    ap.add_argument("--engine", choices=["stub","qwen_vl"], default="qwen_vl")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--offline", action="store_true")

    # Visual Verifier
    ap.add_argument("--with_verify", action="store_true")
    ap.add_argument("--verify_model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")

    # Reviewer（可选，gpt-oss 做文本清洁）
    ap.add_argument("--with_review", action="store_true")
    ap.add_argument("--review_model_id", default="openai/gpt-oss-20b")
    ap.add_argument("--review_prompt", default="prompts/review_en.txt")

    # —— 新增：vlog 输出控制 —— #
    ap.add_argument("--print_vlog", action="store_true",
                    help="Print vlog/facts_before/facts_after along with supplement JSON.")
    ap.add_argument("--dump_vlog", default=None,
                    help="Append a JSONL record (facts_before/after, vlog, supplement) to this path.")

    args = ap.parse_args()

    # Draft
    engine = (QwenVLSupplementor(
                model_id=args.model_id if args.engine=="qwen_vl" else "stub",
                prompt_path=args.prompt,
                offline=args.offline
              )
              if args.engine=="qwen_vl" else
              QwenVLSupplementorStub(prompt_path=args.prompt, max_tokens=args.max_tokens))

    req = SupplementRequest(images=args.images, task=args.task, lang=args.lang, max_tokens=args.max_tokens)
    draft = engine.infer(req)  # SupplementResult
    out_text = draft.text
    flags = ["DRAFT"]

    # 解析 FACT_LIST
    facts = parse_factlist_from_output(draft.text)
    no_facts = False
    if not facts or (
        not getattr(facts, "objects", None) and
        not getattr(facts, "pose", "") and
        not getattr(facts, "grasp_points", []) and
        not getattr(facts, "obstacles", []) and
        not getattr(facts, "state_text", []) and
        not getattr(facts, "location", "")
    ):
        flags.append("NO_FACTS_FROM_DRAFT")
        no_facts = True

    facts_before_dict = _to_dict_safe(facts)

    vlog = None
    facts_after_dict = None

    # Verify（可选）
    if args.with_verify:
        vf = VisualVerifier(VerifyConfig(model_id=args.verify_model_id, offline=args.offline))
        facts_v, vlog = vf.verify(args.images, facts if facts else FactList())
        flags.append("VERIFIED")

        # 1) 回写标准 FACT_LIST（JSON）
        fact_json_str = json.dumps({
            "objects": [
                {"name": o.name, "attributes": o.attributes} for o in getattr(facts_v, "objects", [])
            ],
            "location": getattr(facts_v, "location", ""),
            "pose": getattr(facts_v, "pose", ""),
            "grasp_points": getattr(facts_v, "grasp_points", []) or [],
            "obstacles": getattr(facts_v, "obstacles", []) or [],
            "state_text": getattr(facts_v, "state_text", []) or [],
            "uncertain": getattr(facts_v, "uncertain", []) or [],
        }, ensure_ascii=False)

        # 2) 规则式回写 SUPPLEMENT（沿用你原本顺序）
        names = ", ".join([
            ((o.attributes.get("color", "") + " " + o.name).strip()
            if getattr(o, "attributes", {}) and o.attributes.get("color") else o.name)
            for o in getattr(facts_v, "objects", [])
        ]) or ""
        supplement_lines = []
        supplement_lines.append(f"Target object(s): {names}".strip())
        supplement_lines.append(f"Location: {getattr(facts_v, 'location', '')}".strip())
        supplement_lines.append(f"Pose/Orientation: {getattr(facts_v, 'pose', '')}".strip())
        gp = ", ".join(getattr(facts_v, "grasp_points", []) or [])
        supplement_lines.append(f"Graspable parts (if visible): {gp}".strip())
        obs = ", ".join(getattr(facts_v, 'obstacles', []) or [])
        supplement_lines.append(f"Obstacles/Free space: {obs}".strip())
        st = ", ".join(getattr(facts_v, 'state_text', []) or [])
        supplement_lines.append(f"State/Labels/Text: {st}".strip())
        un = ", ".join(getattr(facts_v, 'uncertain', []) or [])
        supplement_lines.append(f"UNCERTAIN: {un}".strip())

        # 3) 同时输出两个区块
        out_text = "[FACT_LIST]" + fact_json_str + "[/FACT_LIST]\n" + \
                "[SUPPLEMENT]" + "\n".join(supplement_lines) + "[/SUPPLEMENT]"

        facts_after_dict = _to_dict_safe(facts_v)


    # Review（可选）
    if args.with_review:
        try:
            from .review.gpt_oss import GPTOssReviewer
            reviewer = GPTOssReviewer(model_id=args.review_model_id,
                                      prompt_path=args.review_prompt,
                                      offline=args.offline)
            rr = reviewer.review(out_text)
            out_text = rr.text
            flags += rr.flags
        except ImportError:
            flags.append("REVIEW_MODULE_MISSING")

    # 最终格式校验
    ok, f2 = check_format(out_text)
    if not ok:
        flags += f2

    # —— 主输出 —— #
    payload = {"supplement": out_text, "score": 1.0, "flags": flags}
    if args.print_vlog:
        payload["vlog"] = vlog
        payload["facts_before"] = facts_before_dict
        payload["facts_after"]  = facts_after_dict
    print(json.dumps(payload, ensure_ascii=False), flush=True)

    # —— 可选：落盘 JSONL —— #
    if args.dump_vlog:
        rec = {
            "images": args.images,
            "task": args.task,
            "engine_model_id": args.model_id,
            "verify_model_id": args.verify_model_id if args.with_verify else None,
            "flags": flags,
            "facts_before": facts_before_dict,
            "facts_after": facts_after_dict,
            "vlog": vlog,
            "supplement": out_text
        }
        os.makedirs(os.path.dirname(args.dump_vlog), exist_ok=True)
        with open(args.dump_vlog, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
