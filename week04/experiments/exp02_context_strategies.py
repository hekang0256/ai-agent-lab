"""实验4-B：长上下文管理策略对比（none / sliding / summary）。

场景：14 轮对话，第 1 轮埋下关键信息「项目代号 FALCON-9」。
观察：
- none    不裁剪，token 超预算（真实 API 会直接报 context_length_exceeded）
- sliding 丢掉旧轮次 → 遗忘 FALCON-9
- summary 旧轮次压缩成摘要 → FALCON-9 仍保留在摘要里，且 token 达标
"""
from __future__ import annotations

from llm.client import ChatMessage
from llm.context import ContextManager, heuristic_summary
from llm.mock import MockLLMClient
from llm.session import ChatSession

TURNS = 14


def build_history() -> list[ChatMessage]:
    history: list[ChatMessage] = []
    for i in range(1, TURNS + 1):
        if i == 1:
            user = "第1轮：项目代号 FALCON-9，本周完成了需求评审，下周进入开发阶段。"
        else:
            user = f"第{i}轮：本周完成了模块{i}的开发与联调，修复了若干缺陷，进展符合预期。"
        assistant = f"收到，已记录第{i}轮进展。" + "我会持续跟踪该模块的质量与进度。" * 3
        history.append(ChatMessage(role="user", content=user))
        history.append(ChatMessage(role="assistant", content=assistant))
    return history


def main() -> None:
    history = build_history()
    print(f"历史消息：{len(history)} 条（{TURNS} 轮）")
    print(f"{'策略':<10} {'before':>7} {'after':>7} {'丢弃':>4} {'摘要':>4}  保留FALCON-9?")
    print("-" * 60)

    for strategy in ("none", "sliding", "summary"):
        manager = ContextManager(
            strategy=strategy, max_tokens=1200, reserve_output=200,
            keep_recent=4, summarizer=heuristic_summary,
        )
        msgs, stats = manager.build_messages(history, system="你是项目助理。")
        joined = "".join(m.text for m in msgs)
        print(
            f"{strategy:<10} {stats.before_tokens:>7} {stats.after_tokens:>7} "
            f"{stats.dropped_messages:>4} {stats.summarized_messages:>4}  "
            f"{'✅' if 'FALCON-9' in joined else '❌ 已遗忘'}"
        )

    # 演示 LLM 摘要（Mock）：ChatSession 会自动注入 LLM summarizer
    print("\n== LLM 摘要示例（Mock） ==")
    session = ChatSession(
        MockLLMClient(),
        context=ContextManager(
            strategy="summary", max_tokens=1200, reserve_output=200, keep_recent=4,
        ),
    )
    session.history = history
    reply = session.ask("我们项目的代号是什么？")
    ctx = session.last_context_stats
    print(f"回答：{reply}")
    print(f"摘要消息内容：{[m.text for m in session.context.build_messages(history)[0] if m.role == 'system' and '摘要' in m.text]}")
    print(f"统计：{ctx}")


if __name__ == "__main__":
    main()
