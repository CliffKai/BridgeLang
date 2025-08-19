#!/usr/bin/env python3
import json, argparse, csv, os, pathlib, re

def as_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return {}

def parse_uncertain(unc_list):
    tags = {"pose":[], "location":[], "grasp":[], "objects":[]}
    if not isinstance(unc_list, list): return tags
    for u in unc_list:
        if not isinstance(u, str): continue
        k, v = (u.split(":", 1) + [""])[:2]
        k = k.strip().lower(); v = v.strip()
        if k in tags: tags[k].append(v)
        elif k == "grasp_points": tags["grasp"].append(v)
    return tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_csv",  required=True)
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): continue
            rec = json.loads(ln)
            fact = as_dict(rec.get("facts_after") or rec.get("facts_before") or {})
            unc  = fact.get("uncertain") or []
            if not unc:  # 只输出有 uncertain 的样本
                continue
            tags = parse_uncertain(unc)
            img = os.path.basename((rec.get("images") or [""])[0])
            rows.append({
                "image": img,
                "task": rec.get("task",""),
                "location": fact.get("location",""),
                "pose": fact.get("pose",""),
                "grasp_points": "; ".join(fact.get("grasp_points") or []),
                "unc_pose": "; ".join(tags["pose"]),
                "unc_location": "; ".join(tags["location"]),
                "unc_grasp": "; ".join(tags["grasp"]),
                "unc_objects": "; ".join(tags["objects"]),
            })

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["image","task","location","pose","grasp_points","unc_pose","unc_location","unc_grasp","unc_objects"])
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"Wrote triage: {args.out_csv}  | rows={len(rows)}")

if __name__ == "__main__":
    main()
