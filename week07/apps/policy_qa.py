"""企业制度 RAG 问答：检索证据 → 受约束生成 → 可核验引用。

用法（离线）：
  set LLM_USE_MOCK=true
  python -m apps.policy_qa "差旅报销需要哪些材料" --store memory
  python -m apps.policy_qa --store pg --rebuild
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.document import load_documents
from llm.pg_store import PgChunkStore
from llm.rag import RAGService
from llm.vector_store import MemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "docs" / "policies.md"


def create_service(store_kind: str = "pg") -> RAGService:
    store = MemoryVectorStore() if store_kind == "memory" else PgChunkStore()
    return RAGService(store)


def ensure_index(service: RAGService, docs_path: str | Path = DOCS_PATH, *, rebuild: bool = False):
    if rebuild or service.store.count() == 0:
        return service.index_documents(load_documents(docs_path), clear=rebuild)
    return None


def print_response(response) -> None:
    print(f"\n问题：{response.query}")
    print(f"回答：{response.answer}")
    if response.refused:
        print(f"状态：拒答（{response.refusal_reason}）")
    for citation in response.citations:
        print(
            f"  [{citation.index}] 《{citation.title}》 {citation.chunk_id} "
            f"score={citation.score:.3f}\n      {citation.source}"
        )
    if response.citation_status in {"missing", "invalid"}:
        print(f"引用校验：{response.citation_status} {response.invalid_citation_indexes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="企业制度基础 RAG 问答")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--store", choices=["pg", "memory"], default="pg")
    parser.add_argument("--docs", type=Path, default=DOCS_PATH)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--category")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    args = parser.parse_args()

    service = create_service(args.store)
    stats = ensure_index(service, args.docs, rebuild=args.rebuild)
    if stats:
        print(f"索引完成：{stats.documents} 份文档 → {stats.chunks} 个 Chunk")
    queries = [args.query] if args.query else ["出差打车费用怎么报销？", "离职当天账号权限怎么处理？", "今天中午吃什么？"]
    responses = [
        service.answer(query, top_k=args.top_k, min_score=args.min_score, category=args.category)
        for query in queries
    ]
    if args.json:
        print(json.dumps([response.to_dict() for response in responses], ensure_ascii=False, indent=2))
    else:
        for response in responses:
            print_response(response)
    if not args.query and not args.json:
        output = ROOT / "data" / "rag_demo.json"
        output.write_text(json.dumps([r.to_dict() for r in responses], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n演示结果已写入 {output}")


if __name__ == "__main__":
    main()
