"""真实 LLM + Mock Embedding 的生成冒烟测试；不是完整真实模型验收。"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

from apps.policy_qa import DOCS_PATH
from llm.client import LLMClient
from llm.document import load_documents
from llm.embedding import MockEmbeddingClient
from llm.rag import RAGService
from llm.vector_store import MemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    llm = LLMClient()
    service = RAGService(
        MemoryVectorStore(), embedding_client=MockEmbeddingClient(),
        llm_client=llm,
    )
    service.index_documents(load_documents(DOCS_PATH))
    responses = [
        service.answer(query) for query in [
            "出差打车费用怎么报销？",
            "离职当天账号权限怎么处理？",
            "今天中午吃什么？",
        ]
    ]
    output = ROOT / "data" / "real_llm_smoke.json"
    output.write_text(json.dumps({
        "scope": "real_llm_with_mock_embedding_not_full_acceptance",
        "responses": [response.to_dict() for response in responses],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for response in responses:
        print(f"query={response.query}")
        print(f"answer={response.answer}")
        print(f"refused={response.refused} citation_status={response.citation_status}")
        print(f"citations={[citation.document_id for citation in response.citations]}")
    print(f"saved={output}")
    llm.close()


if __name__ == "__main__":
    main()
