"""V1 LLM 统一调用层 — 整个24周项目的模型入口雏形。

任何后续阶段（RAG / Tool / Agent / Multi-Agent）都通过这一层访问模型，
从而可以无痛切换云端 API、私有化 vLLM、本地 Ollama 或 Mock。
"""
from .client import LLMClient, LLMConfig, ChatMessage, ChatResponse, get_client
from .mock import MockLLMClient

__all__ = [
    "LLMClient",
    "LLMConfig",
    "ChatMessage",
    "ChatResponse",
    "MockLLMClient",
    "get_client",
]
