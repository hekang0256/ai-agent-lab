"""MockLLMClient（week04）：支持多模态、摘要、记忆问答、工具调用、流式输出。

与 week03 的差异：
- prompt_tokens 用 tokenizer 真实估算（长上下文/成本实验才有意义）
- 新增发票提取（InvoiceExtraction）、上下文摘要（ContextSummary）、记忆问答路由
- 多模态消息通过 ChatMessage.text 取文本，不再假设 content 是 str
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Iterator

from .client import (
    ChatMessage,
    ChatResponse,
    LLMError,
    StreamChunk,
    ToolCall,
)
from .tokenizer import estimate_messages_tokens, estimate_text_tokens


class MockLLMClient:
    def __init__(self, model: str = "mock-llm-v4"):
        self.model = model
        self.calls: list[list[ChatMessage]] = []

    # ---------- 非流式 ----------
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        extra: dict | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        started = time.perf_counter()
        last_user = next((m.text for m in reversed(messages) if m.role == "user"), "")
        system = next((m.text for m in messages if m.role == "system"), "")

        # 工具结果回喂优先：避免无限触发工具调用
        if messages[-1].role == "tool":
            content = f"根据工具返回：{messages[-1].text[:200]}"
            return self._response(content, messages, model, started)

        if tools:
            tc = self._maybe_call_tool(tools, last_user)
            if tc:
                return ChatResponse(
                    content="",
                    model=model or self.model,
                    prompt_tokens=estimate_messages_tokens(messages),
                    completion_tokens=20,
                    total_tokens=estimate_messages_tokens(messages) + 20,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    finish_reason="tool_calls",
                    tool_calls=[tc],
                    raw={"mock": True},
                )

        content = self._route(system, last_user, response_format)
        return self._response(content, messages, model, started)

    def _response(self, content: str, messages, model, started) -> ChatResponse:
        prompt_tokens = estimate_messages_tokens(messages)
        completion_tokens = estimate_text_tokens(content)
        return ChatResponse(
            content=content,
            model=model or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason="stop",
            raw={"mock": True},
        )

    # ---------- 流式 ----------
    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        last_user = next((m.text for m in reversed(messages) if m.role == "user"), "")
        system = next((m.text for m in messages if m.role == "system"), "")
        full = self._route(system, last_user, None)
        for i in range(0, len(full), 8):
            yield StreamChunk(delta_content=full[i : i + 8])
            time.sleep(0.001)
        yield StreamChunk(delta_content="", finish_reason="stop")

    # ---------- 内部路由 ----------
    def _route(self, system: str, user: str, response_format: dict | None) -> str:
        # 发票多模态提取
        if "InvoiceExtraction" in system:
            return self._mock_invoice(user)

        # 上下文摘要（LLM summarizer）
        if "ContextSummary" in system:
            return "[MOCK 摘要] 较早对话围绕项目进度、客户合同与发票展开，用户偏好简洁回答。"

        # 记忆问答：依赖注入到 system 的用户画像
        if "我叫什么" in user or "我是谁" in user:
            m = re.search(r"姓名[:：]\s*(\S+)", system)
            if m:
                return f"你叫{m.group(1)}（来自我的长期记忆）。"
            return "我还没有记住你的名字，可以对我说「记住我叫XXX」。"

        if response_format and response_format.get("type") == "json_object":
            return '{"ok": true, "echo": "mock"}'

        if not user:
            raise LLMError("MockLLMClient 收到空 user 消息")
        return f"[MOCK 回复] 你说：{user[:80]}"

    def _mock_invoice(self, user: str) -> str:
        """根据提示中的文件名返回对应发票数据（与 data/images 中的样例一致）。"""
        if "invoice_02" in user or "INV-2026-0002" in user:
            return json.dumps({
                "invoice_no": "INV-2026-0002",
                "date": "2026-05-03",
                "seller": "NORTHWIND TRADERS",
                "buyer": "EXAMPLE TECH LTD",
                "items": [{"name": "OFFICE CHAIR", "quantity": 20, "amount": 9600.0}],
                "subtotal": 9600.0,
                "tax": 576.0,
                "total": 10176.0,
                "currency": "CNY",
            }, ensure_ascii=False, indent=2)
        return json.dumps({
            "invoice_no": "INV-2026-0001",
            "date": "2026-04-12",
            "seller": "ACME SUPPLY CO",
            "buyer": "EXAMPLE TECH LTD",
            "items": [
                {"name": "CLOUD SERVER", "quantity": 2, "amount": 10000.0},
                {"name": "SSD 4TB", "quantity": 5, "amount": 2800.0},
            ],
            "subtotal": 12800.0,
            "tax": 768.0,
            "total": 13568.0,
            "currency": "CNY",
        }, ensure_ascii=False, indent=2)

    def _maybe_call_tool(self, tools: list[dict], user: str) -> ToolCall | None:
        tool_names = {t["function"]["name"] for t in tools}
        if "get_customer" in tool_names and "客户" in user:
            digits = "".join(ch for ch in user if ch.isdigit())
            cid = int(digits) if digits else 1001
            return ToolCall(
                id="call_mock_001", name="get_customer", arguments={"customer_id": cid},
            )
        if "query_contract" in tool_names and "合同" in user:
            return ToolCall(
                id="call_mock_002", name="query_contract", arguments={"keyword": user[:30]},
            )
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "MockLLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
