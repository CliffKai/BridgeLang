from .api import SupplementRequest, SupplementResult

SUPPLEMENT_MINI = "[SUPPLEMENT]No additional details beyond the task description.[/SUPPLEMENT]"

def assemble_prompt(task: str, supplement_text: str | None) -> str:
    parts = [f"[TASK] {task.strip()}"]
    if supplement_text and supplement_text.strip():
        parts.append(supplement_text.strip())
    else:
        parts.append(SUPPLEMENT_MINI)
    return "\n".join(parts)
