"""MockLLMClient：离线演示与单元测试用。

不联网、不消耗 token、响应确定，便于：
- CI / pytest 稳定跑通
- 没有 API key 时也能体验完整调用链
- 后续 RAG/Agent 阶段做故障注入
"""
from __future__ import annotations

import time
from typing import Any

from .client import ChatMessage, ChatResponse, LLMError


class MockLLMClient:
    """与 LLMClient 接口一致，但返回基于规则的固定答案。"""

    def __init__(self, model: str = "mock-llm-v1"):
        self.model = model
        self.calls: list[list[ChatMessage]] = []  # 记录调用，方便断言

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        extra: dict | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        started = time.perf_counter()
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = next((m.content for m in messages if m.role == "system"), "")

        content = self._route(system, last_user, response_format)
        # 模拟 token 计数（粗略：1 token ≈ 1.5 字）
        prompt_tokens = sum(len(m.content) for m in messages) // 2
        completion_tokens = len(content) // 2
        return ChatResponse(
            content=content,
            model=model or self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw={"mock": True},
        )

    def _route(self, system: str, user: str, response_format: dict | None) -> str:
        """根据 prompt 里的关键词路由到不同固定响应，覆盖本周实验场景。"""
        # 日报场景：返回完整 4 章节模板
        if "项目日报" in system or "项目日报" in user or "项目经理" in system:
            return (
                "# 项目日报\n\n"
                "## 一、今日进展\n- 完成 LLM 客户端封装\n- 启动 Prompt 工程实验\n\n"
                "## 二、风险与阻塞\n- 外部 API 偶发超时\n\n"
                "## 三、明日计划\n- 接入 Structured Output\n- 实现 Tool Calling\n\n"
                "## 四、需要的支持\n- 无"
            )
        # 文本工具场景（同时看 system 和 user，因为不同任务模板位置不同）
        if "摘要" in system or "摘要" in user:
            return "【摘要】本文主要讨论大模型应用开发的关键路径。"
        if "翻译" in system or "翻译" in user or "笔译" in system:
            return "Hello, this is a mocked translation."
        if "润色" in system or "编辑" in system:
            return "本文深入探讨大模型应用开发的关键路径。"
        if "情感" in user or "情感分类" in system:
            return "positive"
        if "关键词" in system or "关键词" in user or "SEO" in system:
            return "大模型, Prompt工程, 智能体"
        # JSON 模式
        if response_format and response_format.get("type") == "json_object":
            return '{"ok": true, "echo": "' + user[:30].replace('"', "'") + '"}'
        # 默认回显，方便调试
        if not user:
            raise LLMError("MockLLMClient 收到空 user 消息")
        return f"[MOCK 回复] 你说：{user[:80]}"

    def close(self) -> None:  # 与真实客户端对齐
        return None

    def __enter__(self) -> "MockLLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
