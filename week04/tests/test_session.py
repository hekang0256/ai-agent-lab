"""ChatSession 综合测试：工具闭环 / 记忆 / 上下文 / 成本。"""
from llm.client import ChatMessage
from llm.context import ContextManager
from llm.memory import MemoryStore
from llm.mock import MockLLMClient
from llm.session import ChatSession
from llm.tools import ToolRegistry, tool


@tool(description="查询客户信息")
def get_customer(customer_id: int) -> dict:
    return {"id": customer_id, "name": "星辰电商集团", "level": "VIP"}


def make_session(tmp_path, *, strategy="sliding", with_tools=True, **ctx_kwargs) -> ChatSession:
    registry = ToolRegistry()
    if with_tools:
        registry.register(get_customer)
    context = ContextManager(
        strategy=strategy,
        max_tokens=ctx_kwargs.get("max_tokens", 8192),
        reserve_output=ctx_kwargs.get("reserve_output", 1024),
        keep_recent=ctx_kwargs.get("keep_recent", 4),
    )
    return ChatSession(
        MockLLMClient(),
        memory=MemoryStore.load(tmp_path / "mem.json"),
        context=context,
        registry=registry,
    )


def test_simple_chat_roundtrip(tmp_path):
    session = make_session(tmp_path)
    reply = session.ask("你好")
    assert "MOCK" in reply
    assert len(session.history) == 2  # user + assistant
    assert session.stats()["turns"] == 1


def test_tool_calling_closed_loop(tmp_path):
    session = make_session(tmp_path)
    reply = session.ask("查一下客户 1001 的信息")
    assert "根据工具返回" in reply
    roles = [m.role for m in session.history]
    assert "tool" in roles
    # 带 tool_calls 的 assistant 也在历史里
    assert any(m.tool_calls for m in session.history)
    # 首次（tool_calls）+ 回喂后（最终回答）至少两次调用都记账
    assert len(session.tracker.records) >= 2


def test_memory_command_and_recall(tmp_path):
    session = make_session(tmp_path)
    assert "记住了" in session.ask("记住我叫张三")
    assert session.memory.recall("name") == "张三"
    reply = session.ask("我叫什么？")
    assert "张三" in reply


def test_forget_command(tmp_path):
    session = make_session(tmp_path)
    session.ask("记住我叫张三")
    assert "忘掉" in session.ask("忘掉我的名字")
    assert session.memory.recall("name") is None
    assert "没有记住" in session.ask("我叫什么？")


def test_reset_clears_history_but_keeps_memory(tmp_path):
    session = make_session(tmp_path)
    session.ask("记住我叫张三")
    session.ask("你好")
    session.reset()
    assert session.history == []
    assert session.memory.recall("name") == "张三"


def test_summary_strategy_uses_llm_summarizer(tmp_path):
    session = make_session(
        tmp_path, strategy="summary", max_tokens=600, reserve_output=100, keep_recent=2,
    )
    # 预填长历史，触发摘要
    for i in range(1, 15):
        session.history.append(ChatMessage(role="user", content=f"第{i}轮：讨论项目模块{i}的进度与质量。"))
        session.history.append(ChatMessage(role="assistant", content="已记录。" + "继续跟踪。" * 5))
    session.ask("新问题：项目现在怎么样？")
    stats = session.last_context_stats
    assert stats.strategy == "summary"
    assert stats.summarized_messages > 0
    # LLM 摘要的调用也被记账（成本可观测）
    assert any(r.task == "summary" for r in session.tracker.records)


def test_multimodal_ask(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"fake-png")
    session = make_session(tmp_path)
    reply = session.ask("这张图是什么？", images=[str(img)])
    assert reply  # Mock 正常返回
    assert isinstance(session.history[0].content, list)


def test_stats_fields(tmp_path):
    session = make_session(tmp_path)
    session.ask("你好")
    stats = session.stats()
    assert {"turns", "history_messages", "total_tokens", "total_cost", "last_context"} <= set(stats)
    assert stats["total_tokens"] > 0
