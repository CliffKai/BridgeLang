# scripts/check_factlist.py
import sys, json, re
raw = sys.stdin.read().strip()
if not raw:
    print("NO STDOUT. Did the CLI crash? Check stderr log."); sys.exit(2)
out = json.loads(raw)

s = out["supplement"]
m = re.search(r"\[FACT_LIST\](.*?)\[/FACT_LIST\]", s, re.S)
assert m, "FACT_LIST missing in supplement"
data = json.loads(m.group(1))
need = ["objects","location","pose","grasp_points","obstacles","state_text","uncertain"]
for k in need: assert k in data, f"missing key: {k}"
print("OK (draft). FACT_LIST keys:", list(data.keys()))
