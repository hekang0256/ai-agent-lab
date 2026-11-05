"""Week03 LLM 调用层扩展：Structured Output / Tool Calling / Streaming。"""
from .client import LLMClient, LLMConfig, ChatMessage, ChatResponse, LLMError, get_client
from .mock import MockLLMClient
from .structured import StructuredClient, ExtractionError
from .tools import Tool, ToolRegistry, ToolCallResult, tool

__all__ = [
    "LLMClient",
    "LLMConfig",
    "ChatMessage",
    "ChatResponse",
    "LLMError",
    "MockLLMClient",
    "StructuredClient",
    "ExtractionError",
    "Tool",
    "ToolRegistry",
    "ToolCallResult",
    "tool",
    "get_client",
]
