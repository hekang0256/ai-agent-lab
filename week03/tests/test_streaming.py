"""Streaming 与 Tool Calling 集成测试（用 MockLLM）。"""
import os

os.environ["LLM_USE_MOCK"] = "true"

from llm.mock import MockLLMClient
from llm.client import ChatMessage
from llm.tools import ToolRegistry, tool


@tool(description="根据客户ID查询客户信息")
def get_customer(customer_id: int) -> dict:
    return {"id": customer_id, "name": "张三", "tier": "VIP"}


def test_stream_yields_chunks():
    client = MockLLMClient()
    msgs = [ChatMessage(role="user", content="讲个故事")]
    chunks = list(client.chat_stream(msgs))
    # 至少 2 个 chunk（内容 + 结束）
    assert len(chunks) >= 2
    # 最后一个 chunk 有 finish_reason
    assert chunks[-1].finish_reason == "stop"
    # 拼接内容非空
    full = "".join(c.delta_content for c in chunks)
    assert len(full) > 0


def test_tool_calling_full_loop():
    """完整的 提问→tool_call→执行→回喂→最终回答 闭环。"""
    client = MockLLMClient()
    registry = ToolRegistry()
    registry.register(get_customer)

    msgs = [
        ChatMessage(role="system", content="CRM 助手"),
        ChatMessage(role="user", content="查一下客户 1001"),
    ]

    # 第 1 次：模型返回 tool_call
    resp1 = client.chat(msgs, tools=registry.to_openai_schema())
    assert resp1.has_tool_calls
    assert resp1.finish_reason == "tool_calls"
    tc = resp1.tool_calls[0]
    assert tc.name == "get_customer"
    assert tc.arguments["customer_id"] == 1001

    # 执行工具
    result = registry.dispatch(tc.id, tc.name, tc.arguments)
    assert result.ok
    assert result.result["name"] == "张三"

    # 回喂
    msgs.append(ChatMessage(role="assistant", content="", tool_calls=[
        {"id": tc.id, "type": "function",
         "function": {"name": tc.name, "arguments": '{"customer_id": 1001}'}}
    ]))
    msgs.append(ChatMessage(
        role="tool", content=result.to_message_content(),
        tool_call_id=tc.id, name=tc.name,
    ))

    # 第 2 次：模型基于工具结果给最终回答
    resp2 = client.chat(msgs, tools=registry.to_openai_schema())
    assert not resp2.has_tool_calls
    assert "张三" in resp2.content or "工具返回" in resp2.content
