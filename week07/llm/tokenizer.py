"""离线 Token 粗估：CJK 约 1 token/字，其它字符约 4 字符/token。"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import ChatMessage

MESSAGE_OVERHEAD = 4
REPLY_PRIMER = 2


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0x3000 <= code <= 0x303F or 0xFF00 <= code <= 0xFFEF


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    return max(1, math.ceil(cjk + (len(text) - cjk) / 4))


def estimate_messages_tokens(messages: list["ChatMessage"]) -> int:
    return sum(MESSAGE_OVERHEAD + estimate_text_tokens(m.content) for m in messages) + REPLY_PRIMER
