"""OpenAI 兼容 Chat Completions 客户端：只保留基础 RAG 所需能力。"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    timeout: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise LLMError("未配置 LLM_API_KEY，或设置 LLM_USE_MOCK=true。")
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
            timeout=self.config.timeout,
        )

    def chat(
        self,
        messages: list[ChatMessage], *, model: str | None = None,
        temperature: float = 0.0, max_tokens: int | None = None,
    ) -> ChatResponse:
        payload = {
            "model": model or self.config.model,
            "messages": [message.to_dict() for message in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        started = time.perf_counter()
        data = self._post_with_retry("/chat/completions", payload)
        choice = data["choices"][0]
        usage = data.get("usage", {}) or {}
        return ChatResponse(
            content=choice["message"].get("content") or "",
            model=data.get("model", model or self.config.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=choice.get("finish_reason", ""), raw=data,
        )

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.post(path, json=payload)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise LLMError(f"LLM 业务错误 {response.status_code}: {response.text}")
                if response.status_code >= 500 or response.status_code == 429:
                    raise httpx.HTTPStatusError("retryable response", request=response.request, response=response)
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                last_error = error
                if attempt + 1 < self.config.max_retries:
                    delay = 2**attempt * 0.5
                    if isinstance(error, httpx.HTTPStatusError):
                        retry_after = error.response.headers.get("Retry-After", "")
                        try:
                            delay = max(delay, min(float(retry_after), 30.0))
                        except ValueError:
                            pass
                    time.sleep(delay)
        if isinstance(last_error, httpx.HTTPStatusError):
            response = last_error.response
            raise LLMError(
                f"LLM 调用失败，已重试 {self.config.max_retries} 次："
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
        raise LLMError(f"LLM 调用失败，已重试 {self.config.max_retries} 次: {last_error}")

    def close(self) -> None:
        self._client.close()


def get_client():
    if os.getenv("LLM_USE_MOCK", "").lower() == "true":
        from .mock import MockLLMClient
        return MockLLMClient()
    return LLMClient()
