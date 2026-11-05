"""Token 估算工具：理解 Token / Context Window 的第一步。

策略：
1. 优先用 tiktoken（OpenAI 官方 BPE 编码器），得到精确 token 数。
2. 没装 tiktoken 或离线场景下退化为字符启发式估算：
   - 英文 ≈ 1 token / 4 字符
   - 中文 ≈ 1 token / 1.5 字
   - 代码 / 数字 ≈ 1 token / 3 字符

为什么必须自己算？
- 在调用前预估 prompt token，避免超过 Context Window
- 估算成本（in_tokens × 单价 + out_tokens × 单价）
- 后续 RAG 阶段做 Chunking 时按 token 切片
"""
from __future__ import annotations

import re

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # tiktoken 缺失或无法加载
    _ENC = None


_CJK_RE = re.compile(r"[一-鿿]")


def count_tokens(text: str, *, precise: bool = True) -> int:
    """统计一段文本的 token 数。

    Args:
        text: 待统计文本
        precise: True 优先用 tiktoken；False 强制走启发式（便于对比）
    """
    if precise and _ENC is not None:
        return len(_ENC.encode(text))
    return _heuristic_count(text)


def _heuristic_count(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    # 中文每字约 0.7 token，英文每字符约 0.25 token
    return int(cjk / 1.5 + other / 4) + 1


def count_messages_tokens(messages: list[dict]) -> int:
    """统计一组 messages 的 token 数（含每条消息的 role/format 开销）。

    OpenAI 官方建议：每条消息额外加 4 token 的元信息开销。
    """
    per_message_overhead = 4
    total = 0
    for m in messages:
        total += per_message_overhead
        total += count_tokens(m.get("role", ""))
        total += count_tokens(m.get("content", ""))
    total += 2  # 整个对话收尾
    return total


def fits_context(messages: list[dict], *, context_window: int, reserved_output: int = 512) -> bool:
    """判断一组 messages 是否能塞进 Context Window，且留出输出空间。"""
    used = count_messages_tokens(messages)
    return used + reserved_output <= context_window


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  *, in_price_per_1m: float, out_price_per_1m: float) -> float:
    """按 1M token 单价估算单次调用成本（单位与价格表一致，常为 USD）。"""
    return (prompt_tokens / 1_000_000) * in_price_per_1m + \
           (completion_tokens / 1_000_000) * out_price_per_1m


if __name__ == "__main__":
    demo = [
        "Hello, world!",
        "你好，世界！",
        "《AI → 智能体》24周系统学习路线",
        "def add(a, b): return a + b",
    ]
    for t in demo:
        precise = count_tokens(t)
        rough = count_tokens(t, precise=False)
        print(f"{t!r:50s}  tiktoken={precise:3d}  heuristic={rough:3d}  chars={len(t)}")
