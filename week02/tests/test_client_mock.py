"""MockLLM 客户端单元测试：保证 Mock 行为稳定，后续阶段可依赖。"""
import os

import pytest

from llm import get_client
from llm.client import ChatMessage, LLMError
from llm.mock import MockLLMClient


@pytest.fixture
def mock() -> MockLLMClient:
    return MockLLMClient()


def test_mock_records_calls(mock):
    msgs = [ChatMessage(role="user", content="hello")]
    mock.chat(msgs)
    mock.chat(msgs)
    assert len(mock.calls) == 2


def test_mock_returns_chat_response_shape(mock):
    resp = mock.chat([ChatMessage(role="user", content="hello")])
    assert resp.content
    assert resp.model == "mock-llm-v1"
    assert resp.total_tokens == resp.prompt_tokens + resp.completion_tokens
    assert resp.latency_ms >= 0


def test_mock_summary_route(mock):
    resp = mock.chat([ChatMessage(role="user", content="请摘要：今天天气很好")])
    assert "摘要" in resp.content


def test_mock_sentiment_route(mock):
    resp = mock.chat([ChatMessage(role="user", content="情感分析：我很开心")])
    assert resp.content.strip() == "positive"


def test_mock_json_mode(mock):
    resp = mock.chat(
        [ChatMessage(role="user", content="随便")],
        response_format={"type": "json_object"},
    )
    assert resp.content.startswith("{")
    assert '"ok": true' in resp.content


def test_mock_empty_user_raises(mock):
    with pytest.raises(LLMError):
        mock.chat([ChatMessage(role="system", content="hi")])


def test_get_client_returns_mock_when_env_set(monkeypatch):
    monkeypatch.setenv("LLM_USE_MOCK", "true")
    client = get_client()
    assert isinstance(client, MockLLMClient)


def test_get_client_raises_without_key(monkeypatch):
    monkeypatch.delenv("LLM_USE_MOCK", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        get_client()
