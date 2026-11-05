"""OpenAI 兼容的 LLM 客户端（httpx 实现，含超时 / 重试 / Token 统计）。

设计要点：
1. 不绑定具体厂商 SDK，只依赖 OpenAI 兼容的 HTTP 协议。
2. 同步 + 异步双接口（async 留给后续 Streaming / Agent Loop 用）。
3. 内置指数退避重试，区分「可重试」（5xx/429/网络错误）和「不可重试」（4xx 业务错误）。
4. 每次调用返回统一的 ChatResponse，包含 usage 与耗时，为后续 Observability 周打基础。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

Role = Literal["system", "user", "assistant", "tool"]


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
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def cost_hint(self) -> str:
        """粗略成本提示，不绑定具体价格表（价格随时间变）。"""
        return f"in={self.prompt_tokens} out={self.completion_tokens} total={self.total_tokens}"


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    timeout: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMError(Exception):
    """统一 LLM 异常，便于上层 catch。"""


class LLMClient:
    """OpenAI 兼容客户端（同步）。

    用法：
        client = LLMClient()
        resp = client.chat([ChatMessage(role="user", content="你好")])
        print(resp.content, resp.cost_hint)
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise LLMError(
                "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填入，"
                "或设置 LLM_USE_MOCK=true 使用 MockLLMClient。"
            )
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )

    # ---------- 核心接口 ----------
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
        payload: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra:
            payload.update(extra)

        started = time.perf_counter()
        data = self._post_with_retry("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000

        choice = data["choices"][0]
        usage = data.get("usage", {}) or {}
        return ChatResponse(
            content=choice["message"]["content"] or "",
            model=data.get("model", payload["model"]),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            raw=data,
        )

    # ---------- 内部 ----------
    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self._client.post(path, json=payload)
                # 4xx 业务错误不重试
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise LLMError(f"LLM 业务错误 {resp.status_code}: {resp.text}")
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise httpx.HTTPStatusError(
                        f"server {resp.status_code}", request=resp.request, response=resp
                    )
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_err = e
                sleep_s = 2**attempt * 0.5
                time.sleep(sleep_s)
        raise LLMError(f"LLM 调用失败，已重试 {self.config.max_retries} 次: {last_err}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def get_client() -> LLMClient | "MockLLMClient":
    """工厂函数：根据环境变量返回真实客户端或 Mock。"""
    if os.getenv("LLM_USE_MOCK", "").lower() == "true":
        from .mock import MockLLMClient

        return MockLLMClient()
    return LLMClient()
