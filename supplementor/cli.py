# openvla-ext/supplementor/cli.py
import argparse, json
from .api import SupplementRequest
from .engine.qwen_stub import QwenVLSupplementorStub
from .engine.qwen_vl import QwenVLSupplementor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--prompt", default="prompts/supplementor_en.txt")
    ap.add_argument("--max_tokens", type=int, default=120)
    ap.add_argument("--engine", choices=["stub", "qwen_vl"], default="qwen_vl")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--use_bnb_8bit", action="store_true")
    args = ap.parse_args()

    if args.engine == "stub":
        engine = QwenVLSupplementorStub(prompt_path=args.prompt, max_tokens=args.max_tokens)
    else:
        engine = QwenVLSupplementor(
            model_id=args.model_id,
            prompt_path=args.prompt,
        )

    req = SupplementRequest(images=args.images, task=args.task, lang=args.lang, max_tokens=args.max_tokens)
    result = engine.infer(req)

    out = {
        "supplement": result.text,
        "score": result.score,
        "flags": result.flags or []
    }
    print(json.dumps(out, ensure_ascii=False))

if __name__ == "__main__":
    main()
