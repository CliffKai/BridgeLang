#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read supplementor JSONL and export a compact CSV (and optional LaTeX).
Now includes 'disagreement' column from heterogeneous cross-check.
"""
import argparse, json, csv
from pathlib import Path


import json, re

SUP_RE  = re.compile(r"\[SUPPLEMENT\](.*?)\[/SUPPLEMENT\]", re.S)

def _as_dict(x):
    if isinstance(x, dict): 
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}

def _parse_structured_supp(supp: str):
    """老格式兜底：从规范化 SUPPLEMENT 里的 label 行回填 schema。"""
    def grab(label):
        m = re.search(rf"{re.escape(label)}\s*(.*)", supp, flags=re.I)
        return (m.group(1).strip() if m else "")
    objs_line = grab("Target object(s):")
    objs=[]
    if objs_line:
        for name in [x.strip() for x in re.split(r"[;,]", objs_line) if x.strip()]:
            objs.append({"name": name, "attributes": {}})
    return {
        "objects": objs,
        "location": grab("Location:"),
        "pose": grab("Pose/Orientation:"),
        "grasp_points": [x.strip() for x in re.split(r"[;,]", grab("Graspable parts (if visible):")) if x.strip()],
        "obstacles": [x.strip() for x in re.split(r"[;,]", grab("Obstacles/Free space:")) if x.strip()],
        "state_text": [x.strip() for x in re.split(r"[;,]", grab("State/Labels/Text:")) if x.strip()],
        "uncertain": [x.strip() for x in re.split(r"[;,]", grab("UNCERTAIN:")) if x.strip()],
    }

def get_fact(rec: dict) -> dict:
    """
    统一读取：facts_after → facts_before → 兜底解析 SUPPLEMENT 文本 → 返回完整 schema（可能为空）。
    """
    fact = _as_dict(rec.get("facts_after") or rec.get("facts_before") or {})
    if not fact:
        m = SUP_RE.search(rec.get("supplement") or "")
        fact = _parse_structured_supp(m.group(1)) if m else {}
    # 补齐 schema 键，避免下游 KeyError
    return {
        "objects":       fact.get("objects", []),
        "location":      fact.get("location", "") or "",
        "pose":          fact.get("pose", "") or "",
        "grasp_points":  fact.get("grasp_points", []),
        "obstacles":     fact.get("obstacles", []),
        "state_text":    fact.get("state_text", []),
        "uncertain":     fact.get("uncertain", []),
    }




def _obj_str(facts_after):
    if not facts_after: return ""
    objs = []
    for o in facts_after.get("objects", []):
        name = o.get("name","")
        color = o.get("attributes",{}).get("color")
        objs.append(f"{color+' ' if color else ''}{name}".strip())
    return ", ".join([s for s in objs if s])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--latex_out", default=None)
    args = ap.parse_args()

    rows = []
    n = 0
    nonempty = {"objects":0,"location":0,"pose":0,"grasp_points":0}
    verified = 0
    total_time = 0.0
    disag_cnt = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            rec = json.loads(ln)
            n += 1
            facts = rec.get("facts_after") or rec.get("facts_before") or {}
            disagreement = rec.get("disagreement", None)

            row = {
                "image": Path(rec["images"][0]).name if rec.get("images") else "",
                "objects": _obj_str(facts),
                "location": facts.get("location",""),
                "pose": facts.get("pose",""),
                "grasp_points": ", ".join(facts.get("grasp_points",[]) or []),
                "flags": " ".join(rec.get("flags",[])),
                "elapsed_s": f"{rec.get('elapsed_s', 0.0):.1f}",
                "disagreement": "" if disagreement is None else str(bool(disagreement)),
            }
            rows.append(row)
            # stats
            if row["objects"]: nonempty["objects"] += 1
            if row["location"]: nonempty["location"] += 1
            if row["pose"]: nonempty["pose"] += 1
            if row["grasp_points"]: nonempty["grasp_points"] += 1
            if "VERIFIED" in rec.get("flags",[]): verified += 1
            total_time += rec.get("elapsed_s", 0.0)
            if disagreement is True: disag_cnt += 1

    # write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    headers = ["image","objects","location","pose","grasp_points","flags","elapsed_s","disagreement"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows: writer.writerow(r)

    # summary
    def pct(x): return f"{(100.0*x/n if n else 0):.1f}%"
    summary = {
        "samples": n,
        "verified_ratio": pct(verified),
        "nonempty_objects": pct(nonempty["objects"]),
        "nonempty_location": pct(nonempty["location"]),
        "nonempty_pose": pct(nonempty["pose"]),
        "nonempty_grasp_points": pct(nonempty["grasp_points"]),
        "avg_elapsed_s": f"{(total_time/n if n else 0):.1f}",
        "disagreement_ratio": pct(disag_cnt),
    }
    print("SUMMARY:", summary)

    # optional LaTeX
    if args.latex_out and rows:
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l l l l l l}\n\\toprule\n")
            f.write("Image & Objects & Location & Pose & Grasp & Disagree \\\\\n\\midrule\n")
            for r in rows:
                def esc(x): return (x or "").replace("_","\\_").replace("&","\\&")
                f.write(f"{esc(r['image'])} & {esc(r['objects'])} & {esc(r['location'])} & {esc(r['pose'])} & {esc(r['grasp_points'])} & {esc(r['disagreement'])} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

if __name__ == "__main__":
    main()
