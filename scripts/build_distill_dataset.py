#!/usr/bin/env python3
import json, sys, pathlib

def as_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except: return {}
    return {}

def verdict(dec):
    d = (dec or "").strip().upper()
    return d if d in ("YES", "NO", "UNCERTAIN") else "UNCERTAIN"

def merge(d1, d2):
    # 双老师合并：一票 NO 否决；两票 YES 才 YES；其余 UNCERTAIN
    if "NO" in (d1, d2):
        return "NO"
    if d1 == "YES" and d2 == "YES":
        return "YES"
    return "UNCERTAIN"

def _decisions_from_block(block):
    """
    统一把 verdicts[key] 展平为 ['YES'|'NO'|'UNCERTAIN', ...]
    - dict           -> 取 .alt_decision / .decision
    - list[dict/str] -> 逐个取上述字段或自身
    - 其它/缺失      -> []
    """
    out = []
    if isinstance(block, dict):
        v = verdict(block.get("alt_decision") or block.get("decision"))
        if v: out.append(v)
    elif isinstance(block, list):
        for e in block:
            if isinstance(e, dict):
                v = verdict(e.get("alt_decision") or e.get("decision"))
                if v: out.append(v)
            elif isinstance(e, str):
                v = verdict(e)
                if v: out.append(v)
    elif isinstance(block, str):
        v = verdict(block)
        if v: out.append(v)
    return out

def _aggregate_teacher(block):
    """
    单老师聚合：
    - 任何 NO -> NO
    - 非空且全 YES -> YES
    - 其他 -> UNCERTAIN
    """
    decs = _decisions_from_block(block)
    if any(d == "NO" for d in decs):
        return "NO"
    if decs and all(d == "YES" for d in decs):
        return "YES"
    return "UNCERTAIN"

def dec(key):
    # 注意：该函数需与 for ln in ...: rec = json.loads(ln) 处于同一作用域，能访问当前 rec
    v1 = (as_dict(rec.get("review")) or {}).get("verdicts", {})
    v2 = (as_dict(rec.get("alt_review")) or {}).get("verdicts", {})
    t1 = _aggregate_teacher(v1.get(key)) if isinstance(v1, dict) else "UNCERTAIN"
    t2 = _aggregate_teacher(v2.get(key)) if isinstance(v2, dict) else "UNCERTAIN"
    return merge(t1, t2)


def ensure_lists(f):
    f = f or {}
    for k in ("objects","grasp_points","obstacles","state_text","uncertain"):
        if not isinstance(f.get(k), list): f[k] = []
    for k in ("location","pose"):
        f[k] = (f.get(k) or "").strip()
    return f

def main(inp, outp):
    pin  = pathlib.Path(inp)
    pout = pathlib.Path(outp)
    out_lines = []

    system_prompt = pathlib.Path("/mnt/BridgeLang/prompts/supplementor_en.txt").read_text(encoding="utf-8").strip()

    for ln in pin.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        rec = json.loads(ln)

        fact = as_dict(rec.get("facts_after") or rec.get("facts_before") or {})
        fact = ensure_lists(fact)

        rv = as_dict(rec.get("review") or {}).get("verdicts") or {}
        av = as_dict(rec.get("alt_review") or {}).get("verdicts") or {}

        def dec(key):
            v1 = rv.get(key) or {}
            v2 = av.get(key) or {}
            d1 = verdict(v1.get("alt_decision") or v1.get("decision"))
            d2 = verdict(v2.get("alt_decision") or v2.get("decision"))
            return merge(d1, d2)

        # objects：老师非 YES -> 清空 objects，并记原因
        if dec("objects") != "YES":
            if fact.get("objects"):
                fact["uncertain"].append("objects:overruled by teacher")
            fact["objects"] = []

        # location：老师非 YES -> "UNCERTAIN"
        if dec("location") != "YES":
            if fact["location"]:
                fact["uncertain"].append("location:set UNCERTAIN by teacher")
            fact["location"] = "UNCERTAIN"

        # pose：老师非 YES -> "UNCERTAIN"
        if dec("pose") != "YES":
            if fact["pose"]:
                fact["uncertain"].append("pose:set UNCERTAIN by teacher")
            fact["pose"] = "UNCERTAIN"

        # grasp_points：老师非 YES -> ["UNCERTAIN"]
        if dec("grasp_points") != "YES":
            fact["grasp_points"] = ["UNCERTAIN"]
            fact["uncertain"].append("grasp_points:set UNCERTAIN by teacher")

        # uncertain 去重
        fact["uncertain"] = list(dict.fromkeys([u for u in fact["uncertain"] if u]))

        # 生成“老师金标”的目标输出（两段式）
        fact_str = json.dumps(fact, ensure_ascii=False)
        target = (
            f"[FACT_LIST]{fact_str}[/FACT_LIST]\n"
            "[SUPPLEMENT]Visible facts summarized above; UNCERTAIN fields reflect non-visible details.[/SUPPLEMENT]"
        )

        images = [f"file://{p}" for p in (rec.get("images") or [])]
        sample = {
            "messages": [
                {"role":"system","content": system_prompt},
                {"role":"user","content":[*({"type":"image","image":p} for p in images),
                                          {"type":"text","text": rec.get("task","")}]},
                {"role":"assistant","content": target}
            ]
        }
        out_lines.append(json.dumps(sample, ensure_ascii=False))

    pout.parent.mkdir(parents=True, exist_ok=True)
    pout.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"wrote distill set: {pout} | samples={len(out_lines)}")

if __name__ == "__main__":
    assert len(sys.argv)==3, "Usage: build_distill_dataset.py IN.jsonl OUT.jsonl"
    main(sys.argv[1], sys.argv[2])
