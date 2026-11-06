"""离线 Token 估算：不依赖 tiktoken，保证 Mock / 单元测试可离线运行。

为什么不用 tiktoken？
1. tiktoken 首次使用要联网下载 BPE 词表，离线环境会失败；
2. 不同模型的 tokenizer 本来就不同，做上下文预算时「稳定、偏保守的估算」
   比「对某个模型精确、对其它模型失真」更实用。

估算规则（对中英混合文本偏保守，即宁可高估）：
- CJK 字符（含中文标点）：约 1 token / 字
- 其它字符（英文/数字/空白）：约 4 字符 / token
- 每条消息有固定开销（role、分隔符等），这里按 4 token 计
- 图片按 detail 估算：low=85，auto/high=1024（与 OpenAI 早期规则同量级）
- 音频无法离线精确估算，按 base64 数据量粗估

生产环境如果追求精确：给对应模型接入官方 tokenizer，然后替换本模块即可，
ContextManager 只依赖这里的函数签名。
"""
from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .client import ChatMessage

MESSAGE_OVERHEAD = 4   # 每条消息的固定开销
REPLY_PRIMER = 2       # 引导模型输出的开销
IMAGE_TOKENS = {"low": 85, "auto": 1024, "high": 1024}


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF      # CJK 统一表意文字
        or 0x3400 <= o <= 0x4DBF   # 扩展 A
        or 0x3000 <= o <= 0x303F   # CJK 标点
        or 0xFF00 <= o <= 0xFFEF   # 全角字符
    )


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return max(1, math.ceil(cjk + other / 4))


def estimate_content_tokens(content: str | list[dict[str, Any]]) -> int:
    """估算 str 或多模态 content parts 的 token 数。"""
    if isinstance(content, str):
        return estimate_text_tokens(content)
    total = 0
    for part in content:
        ptype = part.get("type")
        if ptype == "text":
            total += estimate_text_tokens(part.get("text", ""))
        elif ptype == "image_url":
            detail = (part.get("image_url") or {}).get("detail", "auto")
            total += IMAGE_TOKENS.get(detail, 1024)
        elif ptype == "input_audio":
            data = (part.get("input_audio") or {}).get("data", "")
            total += max(32, len(data) // 400)  # 粗估：音频 base64 体积 → token
        else:
            total += 16  # 未知 part 给一个保守值
    return total


def estimate_message_tokens(message: "ChatMessage") -> int:
    tokens = MESSAGE_OVERHEAD + estimate_content_tokens(message.content)
    if message.tool_calls:
        import json
        tokens += estimate_text_tokens(json.dumps(message.tool_calls, ensure_ascii=False))
    return tokens


def estimate_messages_tokens(messages: "list[ChatMessage]") -> int:
    return sum(estimate_message_tokens(m) for m in messages) + REPLY_PRIMER
