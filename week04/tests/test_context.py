"""ContextManager 的裁剪行为测试。"""
import json

from llm.client import ChatMessage
from llm.context import ContextManager, heuristic_summary
from llm.tokenizer import estimate_messages_tokens


def make_history(turns: int) -> list[ChatMessage]:
    history = []
    for i in range(1, turns + 1):
        history.append(ChatMessage(role="user", content=f"第{i}轮：讨论项目模块{i}的开发进度与质量。"))
        history.append(ChatMessage(role="assistant", content=f"已记录第{i}轮。" + "继续跟踪。" * 5))
    return history


def test_under_budget_returns_everything():
    history = make_history(2)
    manager = ContextManager(strategy="sliding", max_tokens=4096, reserve_output=512)
    msgs, stats = manager.build_messages(history, system="你是助手")
    assert len(msgs) == len(history) + 1
    assert stats.dropped_messages == 0
    assert stats.before_tokens == stats.after_tokens


def test_none_strategy_never_trims():
    history = make_history(30)
    manager = ContextManager(strategy="none", max_tokens=100, reserve_output=10)
    msgs, stats = manager.build_messages(history)
    assert len(msgs) == len(history)
    assert stats.after_tokens == stats.before_tokens


def test_sliding_drops_old_and_keeps_latest():
    history = make_history(30)
    manager = ContextManager(strategy="sliding", max_tokens=200, reserve_output=20)
    msgs, stats = manager.build_messages(history, system="sys")
    assert stats.dropped_messages > 0
    assert stats.after_tokens < stats.before_tokens
    assert msgs[0].role == "system"
    # 最新一轮的 user 发言必须保留
    assert any("第30轮" in m.text for m in msgs)
    # 最早的消息已被丢弃
    assert not any("第1轮：" in m.text for m in msgs)


def test_sliding_does_not_split_tool_turn():
    """assistant(tool_calls) 与对应的 role=tool 结果必须同进同出。"""
    history = make_history(10)
    history += [
        ChatMessage(role="user", content="查一下客户 1001"),
        ChatMessage(
            role="assistant", content="",
            tool_calls=[{"id": "c1", "type": "function",
                         "function": {"name": "get_customer", "arguments": json.dumps({"customer_id": 1001})}}],
        ),
        ChatMessage(role="tool", content='{"ok": true}', tool_call_id="c1", name="get_customer"),
    ]
    manager = ContextManager(strategy="sliding", max_tokens=200, reserve_output=20)
    msgs, _ = manager.build_messages(history)
    roles = [m.role for m in msgs]
    if "tool" in roles:
        # tool 消息的前一条必须是带 tool_calls 的 assistant
        idx = roles.index("tool")
        assert msgs[idx - 1].role == "assistant"
        assert msgs[idx - 1].tool_calls


def test_sliding_keeps_at_least_latest_turn_even_if_oversized():
    history = [ChatMessage(role="user", content="超长输入" * 500)]
    manager = ContextManager(strategy="sliding", max_tokens=100, reserve_output=10)
    msgs, _ = manager.build_messages(history)
    assert len(msgs) == 1  # 至少保留最新一轮，保证对话可继续


def test_summary_compresses_old_messages():
    history = make_history(20)
    manager = ContextManager(
        strategy="summary", max_tokens=300, reserve_output=50,
        keep_recent=4, summarizer=heuristic_summary,
    )
    msgs, stats = manager.build_messages(history, system="sys")
    assert stats.summarized_messages > 0
    assert stats.after_tokens < stats.before_tokens
    # 摘要消息以 system 形式存在
    summary_msgs = [m for m in msgs if m.role == "system" and "摘要" in m.text]
    assert summary_msgs
    # 旧轮次的关键信息通过摘要保留
    assert any("第1轮" in m.text for m in summary_msgs)
    # 最近一轮原文保留
    assert any("第20轮" in m.text for m in msgs)


def test_summary_respects_budget_when_possible():
    history = make_history(20)
    manager = ContextManager(
        strategy="summary", max_tokens=300, reserve_output=50, keep_recent=2,
        summarizer=heuristic_summary,
    )
    msgs, _ = manager.build_messages(history)
    assert estimate_messages_tokens(msgs) <= manager.budget


def test_invalid_strategy_rejected():
    try:
        ContextManager(strategy="magic")
    except ValueError:
        return
    raise AssertionError("应拒绝未知策略")
