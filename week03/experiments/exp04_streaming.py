"""实验3-D：Streaming —— SSE 流式输出。

为什么需要流式？
1. 用户体验：首字延迟（TTFT）从「等待整段生成」变成「生成第一个 token」
2. 长文本生成避免客户端超时
3. 可以中途取消（用户中断 / 命中敏感词）

运行：python -m experiments.exp04_streaming
"""
from __future__ import annotations

import time

from llm import get_client
from llm.client import ChatMessage


def demo() -> None:
    client = get_client()
    messages = [
        ChatMessage(role="system", content="你是简洁的助手。"),
        ChatMessage(role="user", content="用 100 字介绍一下大语言模型。"),
    ]

    print("=" * 60)
    print("非流式（一次性返回）")
    print("=" * 60)
    started = time.perf_counter()
    resp = client.chat(messages)
    elapsed = (time.perf_counter() - started) * 1000
    print(f"[{elapsed:.0f}ms] {resp.content}")

    print("\n" + "=" * 60)
    print("流式（逐 chunk 返回）")
    print("=" * 60)
    started = time.perf_counter()
    first_chunk_ms = None
    chunks = []
    for chunk in client.chat_stream(messages):
        if first_chunk_ms is None and chunk.delta_content:
            first_chunk_ms = (time.perf_counter() - started) * 1000
        if chunk.delta_content:
            chunks.append(chunk.delta_content)
            print(chunk.delta_content, end="", flush=True)
        if chunk.finish_reason:
            total_ms = (time.perf_counter() - started) * 1000
            print(f"\n\n[TTFT={first_chunk_ms:.0f}ms, 总耗时={total_ms:.0f}ms, "
                  f"chunks={len(chunks)}, finish={chunk.finish_reason}]")


if __name__ == "__main__":
    demo()
    print("\n关键收获：")
    print("  - TTFT (Time To First Token) 是用户体验关键指标")
    print("  - 流式必须在 HTTP 层用 stream=True，不能先 resp.json()")
    print("  - SSE 协议：每行以 'data: ' 开头，'data: [DONE]' 结束")
