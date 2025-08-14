# scripts/check_with_verify.py
import sys, json, re, hashlib
raw = sys.stdin.read().strip()
out = json.loads(raw)

# 1) supplement 同时含两块
s = out["supplement"]
assert "[FACT_LIST]" in s and "[SUPPLEMENT]" in s, "both blocks not found"

# 2) FACT_LIST 可解析
m = re.search(r"\[FACT_LIST\](.*?)\[/FACT_LIST\]", s, re.S)
data = json.loads(m.group(1))
need = ["objects","location","pose","grasp_points","obstacles","state_text","uncertain"]
for k in need: assert k in data, f"missing key: {k}"

# 3) Flags 要有 VERIFIED
flags = set(out.get("flags", []))
assert "VERIFIED" in flags, "missing VERIFIED flag"

# 4) vlog 存在 & 结构基本项
vlog = out.get("vlog")
assert isinstance(vlog, dict), "vlog missing"
for k in ["objects","pose","obstacles","grasp_points","added_uncertain"]:
    assert k in vlog, f"vlog missing key: {k}"

# 5) 补一个稳定性哈希，方便回归检查
h = hashlib.sha256(out["supplement"].encode()).hexdigest()[:12]
print("OK (verify). vlog keys ok. hash(supplement)=", h)
