#!/usr/bin/env python3
import argparse, json, re, csv, os, pathlib, collections


def _is_uncertain_str(s: str) -> bool:
    return isinstance(s, str) and s.strip().upper() == "UNCERTAIN"

def _grasp_all_uncertain(lst) -> bool:
    if not isinstance(lst, list) or not lst:
        return False
    return all(_is_uncertain_str((x or "")) for x in lst)


# ========= 统一的解析工具 =========
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
    统一读取：facts_after → facts_before → 兜底解析 SUPPLEMENT 文本，
    然后在返回前做一次“强制不留空”的兜底（与生成端保持一致）。
    """
    FILL_OBJECTS_UNCERTAIN = False  # 如需 objects 也 100%，改为 True

    fact = _as_dict(rec.get("facts_after") or rec.get("facts_before") or {})
    if not fact:
        m = SUP_RE.search(rec.get("supplement") or "")
        fact = _parse_structured_supp(m.group(1)) if m else {}

    # --- 补齐 schema ---
    objects      = fact.get("objects", [])
    location     = (fact.get("location") or "").strip()
    pose         = (fact.get("pose") or "").strip()
    grasp_points = fact.get("grasp_points", [])
    obstacles    = fact.get("obstacles", [])
    state_text   = fact.get("state_text", [])
    uncertain    = fact.get("uncertain", [])

    # --- 规范 objects ---
    norm_objs = []
    for o in (objects if isinstance(objects, list) else []):
        if isinstance(o, dict):
            name = o.get("name", "")
            attrs = o.get("attributes") if isinstance(o.get("attributes"), dict) else {}
            out = {"name": name, "attributes": attrs}
            if "count" in o: out["count"] = o["count"]
            norm_objs.append(out)
        elif isinstance(o, str):
            norm_objs.append({"name": o, "attributes": {}})
    objects = norm_objs

    # --- 不留空兜底（与生成端保持一致）---
    if not location:
        location = "UNCERTAIN"
        if isinstance(uncertain, list) and "location:not visible" not in uncertain:
            uncertain.append("location:not visible")

    if not pose:
        pose = "UNCERTAIN"
        if isinstance(uncertain, list) and "pose:not visible" not in uncertain:
            uncertain.append("pose:not visible")

    if not (isinstance(grasp_points, list) and any((g or "").strip() for g in grasp_points)):
        grasp_points = ["UNCERTAIN"]
        if isinstance(uncertain, list) and "grasp_points:not visible" not in uncertain:
            uncertain.append("grasp_points:not visible")

    if not isinstance(obstacles, list):  obstacles = []
    if not isinstance(state_text, list): state_text = []
    if not isinstance(uncertain, list):  uncertain = []

    # 可选：把 objects 也追到 100%
    if FILL_OBJECTS_UNCERTAIN and not objects:
        objects = [{"name": "UNCERTAIN", "attributes": {}}]
        if "objects:not visible" not in uncertain:
            uncertain.append("objects:not visible")

    return {
        "objects": objects,
        "location": location,
        "pose": pose,
        "grasp_points": grasp_points,
        "obstacles": obstacles,
        "state_text": state_text,
        "uncertain": uncertain,
    }


def _tex(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    s = s.replace("_", r"\_").replace("{", r"\{").replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    return s

# ========= 主流程 =========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--latex_out", required=False)
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip(): 
                continue
            rec = json.loads(ln)

            # 统一从结构化 facts 读取；仅用于摘要展示时再提 SUPPLEMENT 文本
            fact = get_fact(rec)
            m = SUP_RE.search(rec.get("supplement") or "")
            supp_txt = m.group(1).strip() if m else ""

            img = os.path.basename((rec.get("images") or [""])[0])
            task = rec.get("task","")
            rows.append({
                "image": img,
                "task": task,
                "objects": "; ".join([o.get("name","") for o in fact["objects"]]),
                "location": fact["location"],
                "pose": fact["pose"],
                "grasp_points": "; ".join(fact["grasp_points"]),
                "flags": " ".join(rec.get("flags", [])),
                "elapsed_s": rec.get("elapsed_s"),
                "disagreement": bool(rec.get("disagreement")),
                "supplement_excerpt": (supp_txt[:200].replace("\n"," ") if supp_txt else "")
            })


    # 统计 —— 双指标：Coverage（含 UNCERTAIN）与 Confirmed-Visible（不含 UNCERTAIN）
    n = len(rows)
    covered = collections.Counter()
    confirmed = collections.Counter()

    # 为了做 confirmed 统计，需要重新拿一次结构化 fact
    # 注意：rows 里只有字符串形式，我们重新读取 rec 计算更准确
    # 简单起见，再读一遍文件（IO 开销很小）
    uncertain_tags = collections.Counter()
    verified_cnt = 0
    disagree_cnt = 0
    elapsed_vals = []

    with open(args.in_jsonl, "r", encoding="utf-8") as f2:
        for ln in f2:
            if not ln.strip(): continue
            rec = json.loads(ln)
            fact = get_fact(rec)

            objs = fact.get("objects") or []
            loc  = (fact.get("location") or "")
            pose = (fact.get("pose") or "")
            gps  = fact.get("grasp_points") or []
            unc  = fact.get("uncertain") or []

            # Coverage：只要“有值/有列表”就计入（UNCERTAIN 也算）
            if objs: covered["objects"] += 1
            if loc.strip(): covered["location"] += 1
            if pose.strip(): covered["pose"] += 1
            if gps: covered["grasp_points"] += 1

            # Confirmed-Visible：排除 UNCERTAIN
            if objs: confirmed["objects"] += 1  # objects 不用占位，等同于 coverage
            if loc.strip()  and not _is_uncertain_str(loc):   confirmed["location"] += 1
            if pose.strip() and not _is_uncertain_str(pose):  confirmed["pose"] += 1
            if gps and not _grasp_all_uncertain(gps):         confirmed["grasp_points"] += 1

            # uncertain 标签分布
            for u in unc:
                if isinstance(u, str):
                    tag = u.split(":", 1)[0].strip().lower()
                    if tag: uncertain_tags[tag] += 1

            # 其它指标（保持与你原来一致）
            if rec.get("disagreement"): disagree_cnt += 1
            if "VERIFIED" in " ".join(rec.get("flags", [])): verified_cnt += 1
            if rec.get("elapsed_s") is not None:
                try: elapsed_vals.append(float(rec["elapsed_s"]))
                except: pass

    def _pct(x): 
        return f"{(x/n*100):.1f}%" if n else "0.0%"

    summary = {
        "samples": n,
        "verified_ratio": _pct(verified_cnt),
        # Coverage（含 UNCERTAIN）
        "coverage_objects": _pct(covered["objects"]),
        "coverage_location": _pct(covered["location"]),
        "coverage_pose": _pct(covered["pose"]),
        "coverage_grasp_points": _pct(covered["grasp_points"]),
        # Confirmed-Visible（排除 UNCERTAIN）
        "confirmed_objects": _pct(confirmed["objects"]),
        "confirmed_location": _pct(confirmed["location"]),
        "confirmed_pose": _pct(confirmed["pose"]),
        "confirmed_grasp_points": _pct(confirmed["grasp_points"]),
        "avg_elapsed_s": f"{(sum(elapsed_vals)/len(elapsed_vals)):.1f}" if elapsed_vals else "NA",
        "disagreement_ratio": _pct(disagree_cnt)
    }
    print("SUMMARY:", summary)

    # 也可以顺带打印 uncertain 的 top-10 分布，便于快速诊断
    if uncertain_tags:
        print("UNCERTAIN breakdown (top 10):",
              dict(uncertain_tags.most_common(10)))


    # 写 CSV
    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image","task","objects","location","pose","grasp_points","flags","elapsed_s","disagreement","supplement_excerpt"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 可选：LaTeX（简表）
    if args.latex_out:
        with open(args.latex_out, "w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{l l l l l l}\n\\toprule\n")
            f.write("Image & Objects & Location & Pose & Grasp & Disagree \\\\\n\\midrule\n")
            for r in rows:
                img   = _tex(r.get("image", ""))
                objs  = _tex(r.get("objects", ""))
                loc   = _tex(r.get("location", ""))
                pose  = _tex(r.get("pose", ""))
                grasp = _tex(r.get("grasp_points", ""))
                dis   = "True" if r.get("disagreement") else "False"
                f.write(f"{img} & {objs} & {loc} & {pose} & {grasp} & {dis} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n")

if __name__ == "__main__":
    main()
