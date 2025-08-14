#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the supplementor CLI K times and check output stability.
- 默认测试“确定性”（不采样）：期望 K 次 hash 全一致
- 如传 --sample：开启采样（见下方“可选 patch”），期望 hash 多样
"""
import argparse, hashlib, json, os, subprocess, sys, time

def run_once(args, extra_env=None):
    cmd = [
        sys.executable, "-u", "-m", "supplementor.cli",
        "--engine", "qwen_vl",
        "--model_id", args.model_id,
        "--images", args.image,    # 单图即可，与你当前用法一致
        "--task", args.task,
        "--max_tokens", str(args.max_tokens),
    ]
    if args.offline: cmd.append("--offline")
    if args.with_verify:
        cmd += ["--with_verify", "--verify_model_id", args.verify_model_id, "--print_vlog"]

    env = os.environ.copy()
    if extra_env: env.update(extra_env)

    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"CLI failed (rc={p.returncode}):\n{p.stderr[:500]}")
    # 容忍杂讯：拿到最后一行 JSON
    lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError("No JSON found in stdout. stderr head:\n" + p.stderr[:300])
    out = json.loads(lines[-1])
    sup = out["supplement"]
    h = hashlib.sha256(sup.encode("utf-8")).hexdigest()
    return h, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model_id", default="/mnt/BridgeLang/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--with_verify", action="store_true")
    ap.add_argument("--verify_model_id", default="/mnt/BridgeLang/Qwen2.5-VL-72B-Instruct")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=120)
    ap.add_argument("-k", "--runs", type=int, default=2)
    ap.add_argument("--sample", action="store_true", help="turn on sampling (see patch notes)")
    ap.add_argument("--draft_temp", type=float, default=0.0, help="sampling temperature for Draft (env override)")
    args = ap.parse_args()

    extra_env = {}
    if args.sample:
        # Draft 采样开关（见下方 patch；若未打 patch，设置也无影响）
        if args.draft_temp > 0:
            extra_env["SUPP_DRAFT_TEMP"] = str(args.draft_temp)
        # Verifier 采样开关（见下方 patch；未打 patch则无影响）
        extra_env["SUPP_VF_SAMPLE"] = "1"
        extra_env["SUPP_VF_TEMPERATURE"] = "0.2"
        extra_env["SUPP_VF_TOP_P"] = "0.9"

    hashes = []
    t0 = time.time()
    for i in range(args.runs):
        h, _ = run_once(args, extra_env=extra_env)
        print(f"[{i+1}/{args.runs}] sha256={h[:12]}")
        hashes.append(h)
    dt = time.time() - t0

    uniq = len(set(hashes))
    mode = "SAMPLING" if args.sample else "DETERMINISTIC"
    print(f"Mode={mode}  runs={args.runs}  unique={uniq}  time={dt:.1f}s")
    if not args.sample and uniq != 1:
        sys.exit(1)  # 确定性应一致
    if args.sample and uniq == 1:
        print("WARNING: sampling produced identical outputs (可能未打采样 patch 或温度太低)")
    print("OK")

if __name__ == "__main__":
    main()
