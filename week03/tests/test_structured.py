"""StructuredClient 与 extract_json 单元测试。"""
import os

os.environ["LLM_USE_MOCK"] = "true"

import pytest
from pydantic import BaseModel, Field

from llm.structured import StructuredClient, extract_json, ExtractionError
from llm.mock import MockLLMClient


# ---------- extract_json ----------
def test_extract_pure_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_in_markdown():
    text = '好的，结果如下：\n```json\n{"a": 1, "b": 2}\n```\n希望对你有帮助'
    assert extract_json(text) == {"a": 1, "b": 2}


def test_extract_json_no_code_block_marker():
    text = '```\n{"x": 10}\n```'
    assert extract_json(text) == {"x": 10}


def test_extract_json_fallback_braces():
    text = '前缀噪声 {"k": "v"} 后缀噪声'
    assert extract_json(text) == {"k": "v"}


def test_extract_json_failure():
    with pytest.raises(ExtractionError):
        extract_json("完全不是 JSON 的文本")


# ---------- StructuredClient ----------
class Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)


class _UnknownSchema(BaseModel):
    """Mock 不认识此 schema，会走默认 JSON 路径返回 {"ok": true, "echo": "mock"}，
    缺少必需字段 mysterious_field，触发重试。"""
    mysterious_field: str


def test_structured_client_retries_on_bad_output():
    """Mock 对不认识的 schema 返回缺字段 JSON，应触发重试，最终失败。"""
    llm = MockLLMClient()
    client = StructuredClient(llm, max_retries=2)
    with pytest.raises(ExtractionError, match="3 次尝试"):
        client.extract(schema=_UnknownSchema, system="test", user="test")
    # 应该调用了 1 + 2 = 3 次
    assert len(llm.calls) == 3


def test_structured_client_success_on_contract():
    """Mock 对'合同'关键词返回符合 ContractExtraction 的 JSON。"""
    from apps.contract_extractor import ContractExtraction
    llm = MockLLMClient()
    client = StructuredClient(llm)
    result = client.extract(
        schema=ContractExtraction,
        system="法务助手",
        user="这是合同文本：……",
    )
    assert isinstance(result, ContractExtraction)
    assert result.contract_no == "HT-2026-001"
    assert result.amount_cny == 1200000.0
