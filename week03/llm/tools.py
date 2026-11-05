"""Tool Calling 抽象：把 Python 函数变成 LLM 可调用的工具。

设计要点：
1. `Tool` = name + description + JSON Schema 参数 + Python 函数
2. `ToolRegistry` = 工具集合，提供 `to_openai_schema()` 给 LLM、`dispatch()` 给运行时
3. 执行时做参数校验（pydantic），失败不 crash，返回错误给模型让它自己纠正
4. 高风险操作（如发邮件）应走 Human Approval，本模块留 hook

这是后续 week11 (Tool Calling 专题)、week12-13 (MCP)、week14+ (Agent) 的基石。
"""
from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

logger = logging.getLogger(__name__)


# ---------- 数据模型 ----------
@dataclass
class ToolCallResult:
    """一次工具执行的结果，会作为 role=tool 的消息回喂给模型。"""
    tool_call_id: str
    name: str
    ok: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0

    def to_message_content(self) -> str:
        """序列化成字符串，放进 role=tool 消息的 content 字段。"""
        if self.ok:
            return json.dumps({"ok": True, "result": self.result}, ensure_ascii=False, default=str)
        return json.dumps({"ok": False, "error": self.error}, ensure_ascii=False)


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    args_model: type[BaseModel]
    requires_approval: bool = False  # 高风险工具标记，后续 Security 周用

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def execute(self, tool_call_id: str, arguments: dict) -> ToolCallResult:
        started = time.perf_counter()
        try:
            validated = self.args_model.model_validate(arguments)
            result = self.func(**validated.model_dump())
            return ToolCallResult(
                tool_call_id=tool_call_id, name=self.name, ok=True, result=result,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except ValidationError as e:
            return ToolCallResult(
                tool_call_id=tool_call_id, name=self.name, ok=False,
                error=f"参数校验失败: {e}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as e:  # 业务异常也回喂给模型
            logger.exception("Tool %s 执行异常", self.name)
            return ToolCallResult(
                tool_call_id=tool_call_id, name=self.name, ok=False,
                error=f"{type(e).__name__}: {e}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )


# ---------- 装饰器：从函数签名自动生成 Tool ----------
def tool(name: str | None = None, description: str = "", requires_approval: bool = False):
    """把一个普通 Python 函数变成 Tool。

    用法：
        @tool(description="查询客户信息")
        def get_customer(customer_id: int) -> dict: ...
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        # 用 get_type_hints 拿到带类型的参数签名，动态构造 Pydantic 模型
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        fields: dict[str, tuple] = {}
        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            ann = hints.get(param_name, str)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param_name] = (ann, default)
        args_model = create_model(f"{tool_name}_Args", **fields)
        return Tool(
            name=tool_name,
            description=description or (func.__doc__ or "").strip(),
            func=func,
            args_model=args_model,
            requires_approval=requires_approval,
        )
    return decorator


# ---------- 注册表 ----------
@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, t: Tool) -> None:
        self.tools[t.name] = t

    def to_openai_schema(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tools.values()]

    def dispatch(self, tool_call_id: str, name: str, arguments: dict) -> ToolCallResult:
        t = self.tools.get(name)
        if t is None:
            return ToolCallResult(
                tool_call_id=tool_call_id, name=name, ok=False,
                error=f"未注册的工具: {name}",
            )
        return t.execute(tool_call_id, arguments)

    def __len__(self) -> int:
        return len(self.tools)
