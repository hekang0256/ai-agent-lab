"""Structured Output：让 LLM 稳定输出符合 Pydantic / JSON Schema 的数据。

三种策略（从弱到强）：
1. **prompt_only**：仅靠 Prompt 描述格式 + 后处理 json.loads —— 最弱但最通用
2. **json_object**：API 的 `response_format={"type": "json_object"}` —— 保证合法 JSON，但不保证 schema
3. **json_schema**：API 的 `response_format={"type": "json_schema", ...}` —— 强 schema 约束（OpenAI 4o/DeepSeek 部分支持）

本模块三层都封装，并提供「校验失败重试 + 错误回喂」机制。
"""
from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .client import ChatMessage, LLMClient, LLMError

T = TypeVar("T", bound=BaseModel)


class ExtractionError(Exception):
    """结构化提取失败。"""


# ---------- JSON 提取（处理模型包裹 markdown 代码块的情况）----------
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """从模型输出中稳妥地解析 JSON：
    1) 直接 json.loads
    2) 去掉 markdown ```json ... ``` 包裹
    3) 截取第一个 { 到最后一个 }
    """
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
    # 兜底：抓第一个 { 到最后一个 }
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ExtractionError(f"无法从模型输出中解析 JSON：{text[:200]}...")


# ---------- 结构化客户端 ----------
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
        use_json_mode: bool = True,
        temperature: float = 0.0,
    ) -> T:
        """从用户输入中提取结构化数据，返回 Pydantic 实例。

        流程：
          1) 把 Pydantic schema 转成 JSON Schema 注入 system prompt
          2) 调用 LLM（可选 json_object 模式）
          3) extract_json → schema.model_validate
          4) 失败则把 ValidationError 回喂模型重试（最多 max_retries 次）
        """
        json_schema = schema.model_json_schema()
        sys_prompt = (
            f"{system}\n\n"
            f"# 输出约束\n"
            f"你必须只输出一个 JSON 对象，不要任何额外解释、不要 markdown 代码块。\n"
            f"JSON 必须符合以下 JSON Schema：\n"
            f"```json\n{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n```"
        )

        messages = [
            ChatMessage(role="system", content=sys_prompt),
            ChatMessage(role="user", content=user),
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
                # 把错误回喂，让模型自我修正
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
