from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SupplementRequest:
    images: List[str]            # 图片文件路径（先用路径，后续可改成ndarray或bytes）
    task: str
    lang: str = "auto"           # "auto" 或 显式 "en"/"zh" 等
    max_tokens: int = 120

@dataclass
class SupplementResult:
    text: str                    # 必须含 [SUPPLEMENT]...[/SUPPLEMENT]
    score: float = 1.0           # 置信占位，后续由引擎给
    flags: List[str] = None      # ["UNCERTAIN", "LOW_ALIGNMENT"] 等
