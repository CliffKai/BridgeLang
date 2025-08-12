# supplementor/cli.py
import argparse, json
from .api import SupplementRequest
from .engine.qwen_stub import QwenVLSupplementorStub
from .engine.qwen_vl import QwenVLSupplementor
from .consistency import check_format
from .factlist import parse_factlist_from_output, FactList
from .verifier.verifier_vl import VisualVerifier, VerifyConfig
from .review.gpt_oss import GPTOssReviewer

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

    # Reviewer
    ap.add_argument("--with_review", action="store_true")
    ap.add_argument("--review_model_id", default="openai/gpt-oss-20b")
    ap.add_argument("--review_prompt", default="prompts/review_en.txt")

    args = ap.parse_args()

    # Draft
    engine = QwenVLSupplementor(
        model_id=args.model_id if args.engine=="qwen_vl" else "stub",
        prompt_path=args.prompt,
        offline=args.offline
    ) if args.engine=="qwen_vl" else QwenVLSupplementorStub(prompt_path=args.prompt, max_tokens=args.max_tokens)

    req = SupplementRequest(images=args.images, task=args.task, lang=args.lang, max_tokens=args.max_tokens)
    draft = engine.infer(req)  # SupplementResult
    out_text = draft.text
    flags = ["DRAFT"]

    # —— 解析 FACT_LIST（若无则给空表）——
    facts = parse_factlist_from_output(draft.text)

    # Verify（可选）
    if args.with_verify:
        vf = VisualVerifier(VerifyConfig(model_id=args.verify_model_id, offline=args.offline))
        facts_v, vlog = vf.verify(args.images, facts)
        flags += ["VERIFIED"]
        # 用校正后的 FACT_LIST 回写 SUPPLEMENT（结构化到文本）
        # 简单重拼：保持你 SUPPLEMENT 的字段次序
        supplement_lines = []
        names = ", ".join([ (o.attributes.get("color")+" "+o.name).strip() if o.attributes.get("color") else o.name for o in facts_v.objects]) or ""
        supplement_lines.append(f"Target object(s): {names}".strip())
        supplement_lines.append(f"Location: {facts_v.location}".strip())
        supplement_lines.append(f"Pose/Orientation: {facts_v.pose}".strip())
        gp = ", ".join(facts_v.grasp_points)
        supplement_lines.append(f"Graspable parts (if visible): {gp}".strip())
        obs = ", ".join(facts_v.obstacles)
        supplement_lines.append(f"Obstacles/Free space: {obs}".strip())
        st = ", ".join(facts_v.state_text)
        supplement_lines.append(f"State/Labels/Text: {st}".strip())
        un = ", ".join(facts_v.uncertain)
        supplement_lines.append(f"UNCERTAIN: {un}".strip())
        out_text = "[SUPPLEMENT]" + "\n".join(supplement_lines) + "[/SUPPLEMENT]"

    # Review（可选）
    if args.with_review:
        reviewer = GPTOssReviewer(model_id=args.review_model_id, prompt_path=args.review_prompt, offline=args.offline)
        rr = reviewer.review(out_text)
        out_text = rr.text
        flags += rr.flags

    # 最终格式校验
    ok, f2 = check_format(out_text)
    flags += (f2 if not ok else [])
    print(json.dumps({"supplement": out_text, "score": 1.0, "flags": flags}, ensure_ascii=False))

if __name__ == "__main__":
    main()
