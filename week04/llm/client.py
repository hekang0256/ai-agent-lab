"""OpenAI 兼容 LLM 客户端（week04 增强版）。

相比 week03 新增：
1. 多模态消息 —— `ChatMessage.content` 支持 content parts（文本/图像/音频）
2. `text_part / image_part / audio_part` —— 一行构造多模态内容，本地文件自动转 base64 data URI
3. `ChatMessage.text` —— 无论 content 是 str 还是 parts，都能取出纯文本（Mock / token 估算用）

保留 week03 能力：tools / tool_choice / stream / 重试 / Token 统计。
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import httpx
from dotenv import load_dotenv

load_dotenv()

Role = Literal["system", "user", "assistant", "tool"]

# content 既可以是纯文本，也可以是 OpenAI 风格的 content parts：
# [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
Content = "str | list[dict[str, Any]]"

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


# ---------- 多模态 content parts ----------
def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _to_base64(source: str | Path | bytes) -> tuple[str, str]:
    """把图片/音频来源统一成 (base64, 扩展名)。source 可以是 bytes / 本地路径。"""
    if isinstance(source, bytes):
        return base64.b64encode(source).decode("ascii"), ""
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"多模态文件不存在: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii"), path.suffix.lower()


def image_part(source: str | Path | bytes, *, detail: str = "auto") -> dict[str, Any]:
    """构造图像 content part。

    source 支持四种形态：
      1. http(s) URL        → 直接引用，不上传
      2. data: URI          → 原样使用
      3. 本地文件路径        → 读文件并编码成 data:image/...;base64,...
      4. bytes              → 按 image/png 编码
    """
    if isinstance(source, str) and source.startswith(("http://", "https://", "data:")):
        url = source
    else:
        b64, ext = _to_base64(source)
        mime = _IMAGE_MIME.get(ext, "image/png")
        url = f"data:{mime};base64,{b64}"
    return {"type": "image_url", "image_url": {"url": url, "detail": detail}}


def audio_part(source: str | Path | bytes, *, format: str | None = None) -> dict[str, Any]:
    """构造音频 content part（OpenAI input_audio 风格）。

    注意：音频输入的字段格式各家供应商差异较大（有的走 audio_url、有的走独立 ASR 接口），
    真实调用前请查阅对应厂商文档；本函数用于演示消息结构与 Mock 流程。
    """
    b64, ext = _to_base64(source)
    fmt = format or ext.lstrip(".") or "wav"
    return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}


@dataclass
class ChatMessage:
    role: Role
    content: str | list[dict[str, Any]]
    # Tool Calling 扩展字段
    tool_call_id: str | None = None       # role=tool 时必填
    name: str | None = None               # role=tool 时的函数名
    tool_calls: list[dict] | None = None  # role=assistant 且模型想调工具时

    @property
    def text(self) -> str:
        """取出纯文本内容（多模态消息只取 text parts）。"""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            p.get("text", "") for p in self.content if p.get("type") == "text"
        )

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
