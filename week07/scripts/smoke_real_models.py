"""真实模型最小连接测试；只输出模型、维度、延迟和用量，不输出 API Key。"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv(override=True)

from llm.client import ChatMessage, LLMClient
from llm.embedding import EmbeddingClient


def main() -> None:
    failed = False
    print("== Embedding ==")
    try:
        embedding = EmbeddingClient()
        embedding_response = embedding.embed(["企业制度 RAG 向量维度检测"])
        vector = embedding_response.embeddings[0]
        print(f"model={embedding_response.model}")
        print(f"dimension={len(vector)}")
        print(f"latency_ms={embedding_response.latency_ms:.1f}")
        print(f"total_tokens={embedding_response.total_tokens}")
    except Exception as error:
        failed = True
        print(f"ERROR: {type(error).__name__}: {error}")

    print("\n== Chat ==")
    try:
        llm = LLMClient()
        chat_response = llm.chat(
            [ChatMessage("user", "这是 RAG 验收连接测试，请只回复：连接成功")],
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")), max_tokens=128,
        )
        print(f"model={chat_response.model}")
        print(f"content={chat_response.content.strip()}")
        print(f"latency_ms={chat_response.latency_ms:.1f}")
        print(f"usage={chat_response.total_tokens}")
        llm.close()
    except Exception as error:
        failed = True
        print(f"ERROR: {type(error).__name__}: {error}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
