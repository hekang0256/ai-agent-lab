"""OpenAI 兼容 LLM 客户端（week03 增强版）。

相比 week02 新增：
1. `tools` 参数 —— Function Calling 支持
2. `tool_choice` 参数 —— 强制/禁止工具调用
3. `stream=True` —— SSE 流式响应
4. `tool_calls` 字段解析进 ChatResponse
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ChatMessage:
    role: Role
    content: str
    # Tool Calling 扩展字段
    tool_call_id: str | None = None       # role=tool 时必填
    name: str | None = None               # role=tool 时的函数名
    tool_calls: list[dict] | None = None  # role=assistant 且模型想调工具时

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        return d


@dataclass
class ToolCall:
    """模型返回的一次工具调用请求。"""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def cost_hint(self) -> str:
        return f"in={self.prompt_tokens} out={self.completion_tokens} total={self.total_tokens}"

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class StreamChunk:
    """SSE 流式响应的一个增量。"""
    delta_content: str = ""
    finish_reason: str | None = None


class LLMError(Exception):
    """统一 LLM 异常。"""


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    timeout: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "3"))


class LLMClient:
    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()
        if not self.config.api_key:
            raise LLMError("未配置 LLM_API_KEY。请复制 .env.example 为 .env，或设置 LLM_USE_MOCK=true。")
        self._client = httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )

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
        payload = self._build_payload(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            response_format=response_format, tools=tools, tool_choice=tool_choice, extra=extra,
        )
        started = time.perf_counter()
        data = self._post_with_retry("/chat/completions", payload)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._parse_response(data, latency_ms)

    # ---------- 流式 ----------
    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        """SSE 流式输出，逐 chunk yield。"""
        payload = self._build_payload(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
        )
        payload["stream"] = True
        with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code != 200:
                raise LLMError(f"stream 调用失败 {resp.status_code}: {resp.read().decode()}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                yield StreamChunk(
                    delta_content=delta.get("content", "") or "",
                    finish_reason=choice.get("finish_reason"),
                )

    # ---------- 内部 ----------
    def _build_payload(self, messages, **kwargs) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": kwargs.get("model") or self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": kwargs.get("temperature", 0.7),
        }
        for key in ("max_tokens", "response_format", "tools", "tool_choice"):
            v = kwargs.get(key)
            if v is not None:
                payload[key] = v
        extra = kwargs.get("extra")
        if extra:
            payload.update(extra)
        return payload

    def _parse_response(self, data: dict, latency_ms: float) -> ChatResponse:
        choice = data["choices"][0]
        msg = choice["message"]
        usage = data.get("usage", {}) or {}
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, KeyError):
                args = {"_raw": tc.get("function", {}).get("arguments", "")}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        return ChatResponse(
            content=msg.get("content") or "",
            model=data.get("model", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason", ""),
            tool_calls=tool_calls,
            raw=data,
        )

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                resp = self._client.post(path, json=payload)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise LLMError(f"LLM 业务错误 {resp.status_code}: {resp.text}")
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise httpx.HTTPStatusError(f"server {resp.status_code}", request=resp.request, response=resp)
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_err = e
                time.sleep(2**attempt * 0.5)
        raise LLMError(f"LLM 调用失败，已重试 {self.config.max_retries} 次: {last_err}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def get_client() -> "LLMClient | MockLLMClient":
    if os.getenv("LLM_USE_MOCK", "").lower() == "true":
        from .mock import MockLLMClient
        return MockLLMClient()
    return LLMClient()


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .mock import MockLLMClient
