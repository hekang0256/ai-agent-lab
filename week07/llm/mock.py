"""离线 RAG 生成器：从 prompt 的第一个证据块做抽取式回答并保留 [1]。

它不是通用语言模型，只用于验证链路、prompt、引用映射和测试可重复性。
"""
from __future__ import annotations

import re
import time

from .client import ChatMessage, ChatResponse, LLMError
from .tokenizer import estimate_messages_tokens, estimate_text_tokens

INSUFFICIENT_ANSWER = "根据现有资料无法回答。"


class MockLLMClient:
    def __init__(self, model: str = "mock-rag-llm-v1"):
        self.model = model
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self, messages: list[ChatMessage], *, model: str | None = None,
        temperature: float = 0.0, max_tokens: int | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        started = time.perf_counter()
        system = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        if "RAGGroundedQA" not in system:
            raise LLMError("week07 Mock 只处理 RAGGroundedQA 消息")
        blocks = list(re.finditer(
            r"(?m)^\[(\d+)\] document_id=([^|\n]+?) \| title=([^|\n]+?) \| [^\n]*\n"
            r"(.*?)(?=^\[\d+\] document_id=|\Z)",
            user, flags=re.S,
        ))
        if not blocks:
            content = INSUFFICIENT_ANSWER
        else:
            # 离线 Mock 采用 Top-1 抽取，避免假装具备真实模型的推理能力。
            block = blocks[0]
            evidence = block.group(4).strip()
            content = f"{evidence} [{block.group(1)}]" if evidence else INSUFFICIENT_ANSWER
        prompt_tokens = estimate_messages_tokens(messages)
        completion_tokens = estimate_text_tokens(content)
        return ChatResponse(
            content=content, model=model or self.model,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason="stop", raw={"mock": True, "mode": "extractive"},
        )

    def close(self) -> None:
        return None
