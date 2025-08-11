import re

SUP_RE = re.compile(r"\[SUPPLEMENT\].*?\[/SUPPLEMENT\]", re.S)

def check_format(s: str, max_chars: int = 1200) -> tuple[bool, list[str]]:
    flags = []
    if not SUP_RE.search(s):
        flags.append("MISSING_SUPPLEMENT_TAGS")
    if len(s) > max_chars:
        flags.append("TOO_LONG")
    return (len(flags) == 0), flags
