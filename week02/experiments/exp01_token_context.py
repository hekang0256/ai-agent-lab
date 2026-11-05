"""实验0-A：Token / Context Window / Messages 三件事的直观理解。

运行：python -m experiments.exp01_token_context
"""
from __future__ import annotations

from llm.tokens import count_tokens, count_messages_tokens, fits_context


def demo_tokenization() -> None:
    print("=" * 60)
    print("1. 同一段意思，不同语言/格式，token 数差异巨大")
    print("=" * 60)
    samples = [
        "Hello, how are you?",
        "你好，你怎么样？",
        "人工智能正在改变世界",
        "AI is changing the world",
        "1+1=2",
        "🙂🙂🙂",
    ]
    for s in samples:
        print(f"  {s!r:30s} → {count_tokens(s):3d} tokens, {len(s):2d} chars")


def demo_messages_overhead() -> None:
    print("\n" + "=" * 60)
    print("2. Messages 列表本身也占 token（role、格式开销）")
    print("=" * 60)
    messages = [
        {"role": "system", "content": "你是一个简洁的助手。"},
        {"role": "user", "content": "什么是 LLM？"},
    ]
    raw = sum(count_tokens(m["content"]) for m in messages)
    with_overhead = count_messages_tokens(messages)
    print(f"  仅 content 字符估算：{raw} tokens")
    print(f"  含 role 与格式开销：{with_overhead} tokens")
    print(f"  元信息额外消耗：{with_overhead - raw} tokens")


def demo_context_window() -> None:
    print("\n" + "=" * 60)
    print("3. Context Window 上限检查")
    print("=" * 60)
    long_text = "人工智能" * 2000  # 约 8000 中文字
    messages = [{"role": "user", "content": long_text}]
    for window in (4096, 8192, 32768, 131072):
        ok = fits_context(messages, context_window=window, reserved_output=512)
        print(f"  window={window:6d} tokens → {'✅ 可容纳' if ok else '❌ 超出'}")


if __name__ == "__main__":
    demo_tokenization()
    demo_messages_overhead()
    demo_context_window()
    print("\n结论：")
    print("  - Token ≠ 字符，中文/英文/代码 token 化效率差异大")
    print("  - Messages 元信息也消耗 token")
    print("  - 调用前必须预估 token，否则会被截断或报错")
