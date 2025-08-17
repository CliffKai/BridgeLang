#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, glob, csv, hashlib
from pathlib import Path
import tensorflow as tf

# 这组默认键与您 peek 的字段对齐：
#   - 文本：steps/language_instruction（可能每个 step 一条，取首条非空）
#   - 图像：steps/observation/image_0..3（任选一个可用相机，默认优先 0）
CAND_TEXT = ["steps/language_instruction", "language_instruction", "instruction", "task", "text"]
CAND_IMG  = [
    "steps/observation/image_0",
    "steps/observation/image_1",
    "steps/observation/image_2",
    "steps/observation/image_3",
    "observation/rgb_static", "rgb_static", "image", "front_rgb", "rgb"
]

def _get_feature(ex, name):
    return ex.features.feature.get(name)

def _first_text(ex, keys):
    """从候选键里找文本；如果 bytes_list 有多条（按 step 存），取首个非空"""
    for k in keys:
        f = _get_feature(ex, k)
        if not f or not f.bytes_list.value:
            continue
        for b in f.bytes_list.value:
            if not b: 
                continue
            try:
                s = b.decode("utf-8", errors="ignore").strip()
            except Exception:
                s = ""
            if s:
                return s
    return None

def _first_image_bytes(ex, keys):
    """从候选键里找图像；bytes_list.value 可能有多条，取首个长度较大的条目"""
    for k in keys:
        f = _get_feature(ex, k)
        if not f or not f.bytes_list.value:
            continue
        for b in f.bytes_list.value:
            if b and len(b) > 1000:  # 粗略过滤非图像小字节串
                return b, k
    return None, None

def export_val(tfrec_glob, out_frames, out_manifest, max_n, text_keys, img_keys):
    out_frames = Path(out_frames)
    out_frames.mkdir(parents=True, exist_ok=True)
    rows = []

    files = sorted(glob.glob(tfrec_glob))
    assert files, f"No TFRecord matched: {tfrec_glob}"

    n = 0
    for path in files:
        for raw in tf.compat.v1.io.tf_record_iterator(path):
            if n >= max_n:
                break
            ex = tf.train.Example()
            ex.ParseFromString(raw)

            txt = _first_text(ex, text_keys)
            img_bytes, hit_key = _first_image_bytes(ex, img_keys)
            if not (txt and img_bytes):
                continue

            h = hashlib.sha1(img_bytes).hexdigest()[:12]
            img_path = out_frames / f"val_{h}.jpg"
            with open(img_path, "wb") as f:
                f.write(img_bytes)

            rows.append({"image": str(img_path), "task": txt})
            n += 1
        if n >= max_n:
            break

    outp = Path(out_manifest)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image","task"])
        w.writeheader()
        w.writerows(rows)

    print(f"Exported {len(rows)} examples to:\n  frames: {out_frames}\n  manifest: {outp}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrec_glob", default="/mnt/BridgeLang/data/bridgedata_v2/raw/tfds/bridge_dataset-val.tfrecord-*")
    ap.add_argument("--out_frames", default="/mnt/BridgeLang/data/bridgedata_v2/frames/val")
    ap.add_argument("--out_manifest", default="/mnt/BridgeLang/data/bridgedata_v2/manifests/val_mini.csv")
    ap.add_argument("--max_n", type=int, default=200)
    ap.add_argument("--text_keys", default=",".join(CAND_TEXT))
    ap.add_argument("--img_keys",  default=",".join(CAND_IMG))
    args = ap.parse_args()

    export_val(
        tfrec_glob=args.tfrec_glob,
        out_frames=args.out_frames,
        out_manifest=args.out_manifest,
        max_n=args.max_n,
        text_keys=[k.strip() for k in args.text_keys.split(",") if k.strip()],
        img_keys=[k.strip() for k in args.img_keys.split(",") if k.strip()],
    )

if __name__ == "__main__":
    main()
