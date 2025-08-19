#!/usr/bin/env python3
import json, argparse, pathlib, re

def as_dict(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return {}
    return {}

def has_tag(unc_list, tag):
    if not isinstance(unc_list, list): return False
    tag = tag.lower()
    for u in unc_list:
        if isinstance(u, str) and u.lower().startswith(tag + ":"):
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    p_in  = pathlib.Path(args.in_jsonl)
    p_out = pathlib.Path(args.out_jsonl)

    fixed_loc = fixed_pose = fixed_grasp = 0
    lines = 0

    with p_in.open("r", encoding="utf-8") as fi, p_out.open("w", encoding="utf-8") as fo:
        for ln in fi:
            if not ln.strip():
                continue
            lines += 1
            rec = json.loads(ln)

            # 统一取 facts_after -> facts_before -> {}
            fact = as_dict(rec.get("facts_after") or rec.get("facts_before") or {})
            unc  = fact.get("uncertain") or []

            loc  = (fact.get("location") or "").strip()
            pose = (fact.get("pose") or "").strip()
            gps  = fact.get("grasp_points") or []

            changed = False

            # 空且 uncertain 里出现对应 tag，则补 "UNCERTAIN"
            if not loc and has_tag(unc, "location"):
                fact["location"] = "UNCERTAIN"; fixed_loc += 1; changed = True
            if not pose and has_tag(unc, "pose"):
                fact["pose"] = "UNCERTAIN"; fixed_pose += 1; changed = True
            if (not gps) and has_tag(unc, "grasp") or has_tag(unc, "grasp_points"):
                fact["grasp_points"] = ["UNCERTAIN"]; fixed_grasp += 1; changed = True

            if changed:
                # 写回 facts_after；没有则放 facts_before
                if rec.get("facts_after") is not None:
                    rec["facts_after"] = fact
                else:
                    rec["facts_before"] = fact

            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"wrote: {p_out} | lines={lines} "
          f"| fixed_location={fixed_loc} | fixed_pose={fixed_pose} | fixed_grasp_points={fixed_grasp}")

if __name__ == "__main__":
    main()
