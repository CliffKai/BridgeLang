# supplementor/factlist.py
from __future__ import annotations
import json, re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

_FACT_RE = re.compile(r"\[FACT_LIST\](.*?)\[/FACT_LIST\]", re.S)

@dataclass
class Obj:
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    count: int | None = None

@dataclass
class FactList:
    objects: List[Obj] = field(default_factory=list)
    location: str = ""
    pose: str = ""
    grasp_points: List[str] = field(default_factory=list)
    obstacles: List[str] = field(default_factory=list)
    state_text: List[str] = field(default_factory=list)
    uncertain: List[str] = field(default_factory=list)

    @staticmethod
    def from_json_str(s: str) -> "FactList":
        j = json.loads(s)
        objs = [Obj(**o) if isinstance(o, dict) else Obj(name=str(o)) for o in j.get("objects", [])]
        return FactList(
            objects=objs,
            location=j.get("location", "") or "",
            pose=j.get("pose", "") or "",
            grasp_points=[str(x) for x in j.get("grasp_points", [])],
            obstacles=[str(x) for x in j.get("obstacles", [])],
            state_text=[str(x) for x in j.get("state_text", [])],
            uncertain=[str(x) for x in j.get("uncertain", [])],
        )

    def to_json(self) -> str:
        d = asdict(self)
        d["objects"] = [asdict(o) for o in self.objects]
        return json.dumps(d, ensure_ascii=False)

def extract_factlist_block(text: str) -> str | None:
    m = _FACT_RE.search(text)
    return m.group(1).strip() if m else None

def parse_factlist_from_output(model_text: str) -> FactList:
    js = extract_factlist_block(model_text)
    if js:
        try:
            return FactList.from_json_str(js)
        except Exception:
            pass
    # 兜底：没有 FACT_LIST 就给空
    return FactList()
