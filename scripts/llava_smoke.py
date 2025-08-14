# scripts/llava_smoke.py
import os, sys, torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

os.environ.setdefault("PT_SDPA_ENABLE_HEAD_DIM_PADDING","1")
try:
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    pass

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/llava_smoke.py <MODEL_DIR> <IMAGE_PATH>")
        sys.exit(1)

    mid = sys.argv[1]              # e.g. /mnt/BridgeLang/LLaVA-v1.6-34B
    img_path = sys.argv[2]         # e.g. test.jpg

    cfg = AutoConfig.from_pretrained(mid, local_files_only=True)
    proc = AutoProcessor.from_pretrained(mid, local_files_only=True, use_fast=True)

    if cfg.model_type == "llava_next":
        from transformers import LlavaNextForConditionalGeneration as LlavaClass
    elif cfg.model_type == "llava":
        from transformers import LlavaForConditionalGeneration as LlavaClass
    else:
        raise ValueError(f"Not a LLaVA family model: {cfg.model_type}")

    model = LlavaClass.from_pretrained(
        mid, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )

    img = Image.open(img_path).convert("RGB")
    messages = [{
        "role":"user","content":[
            {"type":"image","image":img},
            {"type":"text","text":"Is there a snack bag visible in the image? Answer ONLY YES or NO or UNCERTAIN."}
        ]
    }]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs, max_new_tokens=8, do_sample=False,
        eos_token_id=proc.tokenizer.eos_token_id
    )
    ans = proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(ans.strip())

if __name__ == "__main__":
    main()
