"""Tool / ToolRegistry 单元测试。"""
import pytest
from pydantic import ValidationError

from llm.tools import Tool, ToolRegistry, tool


@tool(description="加法计算器")
def add(a: int, b: int) -> int:
    return a + b


@tool(description="会抛错的工具")
def bad_tool(x: int) -> int:
    raise ValueError("模拟业务异常")


def test_tool_decorator_creates_schema():
    assert add.name == "add"
    assert add.description == "加法计算器"
    schema = add.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    params = schema["function"]["parameters"]
    assert "a" in params["properties"]
    assert "b" in params["properties"]


def test_tool_execute_success():
    result = add.execute("call_1", {"a": 3, "b": 4})
    assert result.ok is True
    assert result.result == 7
    assert result.error is None
    assert result.latency_ms >= 0


def test_tool_execute_validation_error():
    """参数类型错误不应 crash，应返回 ok=False。"""
    result = add.execute("call_2", {"a": "not_a_number", "b": 4})
    assert result.ok is False
    assert "参数校验失败" in result.error


def test_tool_execute_business_exception():
    """业务异常也应被捕获，作为 error 回喂给模型。"""
    result = bad_tool.execute("call_3", {"x": 1})
    assert result.ok is False
    assert "ValueError" in result.error


def test_tool_call_result_to_message():
    r = add.execute("call_4", {"a": 1, "b": 2})
    content = r.to_message_content()
    assert '"ok": true' in content
    assert '"result": 3' in content


def test_registry_dispatch_known():
    reg = ToolRegistry()
    reg.register(add)
    result = reg.dispatch("c1", "add", {"a": 10, "b": 20})
    assert result.ok and result.result == 30


def test_registry_dispatch_unknown():
    reg = ToolRegistry()
    result = reg.dispatch("c2", "nonexistent", {})
    assert result.ok is False
    assert "未注册" in result.error


def test_registry_to_openai_schema():
    reg = ToolRegistry()
    reg.register(add)
    reg.register(bad_tool)
    schemas = reg.to_openai_schema()
    assert len(schemas) == 2
    names = {s["function"]["name"] for s in schemas}
    assert names == {"add", "bad_tool"}


def test_requires_approval_flag():
    @tool(description="高风险", requires_approval=True)
    def dangerous(x: int) -> int:
        return x

    assert dangerous.requires_approval is True
    assert add.requires_approval is False
