from . import __name__ as pkgname
from ..api import SupplementRequest, SupplementResult

SYSTEM_PLACEHOLDER = "[SUPPLEMENT]No additional details beyond the task description.[/SUPPLEMENT]"

class QwenVLSupplementorStub:
    """
    占位引擎：不看图，只返回占位SUPPLEMENT。
    目的：尽快打通“接口—拼接—OpenVLA推理入口”的链路。
    后续你只需把本类替换为真正的Qwen2.5-VL调用即可。
    """
    def __init__(self, prompt_path: str, max_tokens: int = 120):
        self.prompt_path = prompt_path
        self.max_tokens = max_tokens
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

    def infer(self, req: SupplementRequest) -> SupplementResult:
        # 这里先不做任何真实推理与看图
        return SupplementResult(text=SYSTEM_PLACEHOLDER, score=1.0, flags=[])
