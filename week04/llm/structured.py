"""Structured Output（week04 版）：在 week03 基础上支持多模态输入。

变化点：`StructuredClient.extract()` 新增 `images` / `audio` 参数，
可以把「图片 + 文本指令」一起发给模型，同时保留 Pydantic 校验与失败回喂重试。
典型场景：发票/合同扫描件 → 直接输出符合 schema 的 JSON。
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .client import ChatMessage, LLMClient, audio_part, image_part, text_part
from .client import LLMError  # noqa: F401  （保持与 week03 相同的导出）

T = TypeVar("T", bound=BaseModel)


class ExtractionError(Exception):
    """结构化提取失败。"""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """从模型输出中稳妥地解析 JSON（直接解析 → 去 markdown 包裹 → 截取花括号）。"""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ExtractionError(f"无法从模型输出中解析 JSON：{text[:200]}...")


class StructuredClient:
    """在 LLMClient 之上提供「返回 Pydantic 模型」的能力。"""

    def __init__(self, llm: LLMClient, *, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def extract(
        self,
        *,
        schema: type[T],
        system: str,
        user: str,
        images: list[str] | None = None,
        audio: list[str] | None = None,
        use_json_mode: bool = True,
        temperature: float = 0.0,
    ) -> T:
        """从（可能是多模态的）输入中提取结构化数据，返回 Pydantic 实例。"""
        json_schema = schema.model_json_schema()
        sys_prompt = (
            f"{system}\n\n"
            f"# 输出约束\n"
            f"你必须只输出一个 JSON 对象，不要任何额外解释、不要 markdown 代码块。\n"
            f"JSON 必须符合以下 JSON Schema：\n"
            f"```json\n{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n```"
        )

        # 有多模态输入时，user 消息用 content parts；否则保持纯文本
        if images or audio:
            parts: list[dict] = [text_part(user)]
            parts += [image_part(src) for src in images or []]
            parts += [audio_part(src) for src in audio or []]
            user_content: str | list[dict] = parts
        else:
            user_content = user

        messages = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user_content),
        ]
        response_format = {"type": "json_object"} if use_json_mode else None

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            resp = self.llm.chat(messages, temperature=temperature, response_format=response_format)
            try:
                data = extract_json(resp.content)
                return schema.model_validate(data)
            except (ExtractionError, ValidationError) as e:
                last_error = e
                messages.append(ChatMessage(role="assistant", content=resp.content))
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            f"上一次输出无法通过校验，错误：{e}\n"
                            f"请重新输出，严格遵守 JSON Schema，不要任何额外文字。"
                        ),
                    )
                )
        raise ExtractionError(f"经过 {self.max_retries + 1} 次尝试仍失败：{last_error}")
