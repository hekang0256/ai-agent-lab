"""用当前 .env 的 Embedding 重建 week07 主表；不调用生成模型。"""
from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)

from apps.policy_qa import DOCS_PATH, create_service
from llm.document import load_documents


def main() -> None:
    service = create_service("pg")
    stats = service.index_documents(load_documents(DOCS_PATH), clear=True)
    print(f"documents={stats.documents} chunks={stats.chunks} embedding_tokens={stats.embedding_tokens}")
    print(f"embedding_model={getattr(service.embedding_client, 'model', 'unknown')}")
    print(f"pg_count={service.store.count()}")


if __name__ == "__main__":
    main()
