"""离线 Token 估算（与 week04 相同的启发式，自包含到本周）。

CJK 字符约 1 token/字，其它字符约 4 字符/token。用于 Mock 的成本统计。
"""
from __future__ import annotations

import math


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x3000 <= o <= 0x303F
        or 0xFF00 <= o <= 0xFFEF
    )


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, math.ceil(cjk + other / 4))
