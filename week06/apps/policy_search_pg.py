"""应用：企业制度语义搜索 · pgvector 版（week05 应用的数据库升级）。

检索逻辑与 week05 完全一致，但存储/过滤/排序全部下推到 PostgreSQL：
- 语义：embedding <=> query（cosine distance）
- Top-K：ORDER BY distance LIMIT k
- Metadata Filter：WHERE category = %s（一条 SQL 完成）
- 拒答：min_score 阈值

前置：docker compose up -d（pgvector 容器，端口 5433）

用法：
  python -m apps.policy_search_pg --rebuild                # 重建索引（14 条制度）
  python -m apps.policy_search_pg "打车费怎么报销"
  python -m apps.policy_search_pg "报销流程" --category 财务
  python -m apps.policy_search_pg                          # 演示查询（含过滤与拒答）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.embedding import get_embedding_client
from llm.pg_store import PgVectorStore, SearchResult

ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "docs" / "policies.json"
DEFAULT_MIN_SCORE = 0.35


class PgPolicySearch:
    def __init__(self, client=None, store: PgVectorStore | None = None):
        self.client = client or get_embedding_client()
        self.store = store or PgVectorStore()

    def build_index(self, docs_path: str | Path = DOCS_PATH) -> int:
        docs = json.loads(Path(docs_path).read_text(encoding="utf-8"))
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        resp = self.client.embed(texts)
        records = [
            {"id": d["id"], "title": d["title"], "category": d["category"],
             "text": text, "vector": vector}
            for d, text, vector in zip(docs, texts, resp.embeddings)
        ]
        self.store.upsert_many(records)
        return len(records)

    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.0,
        category: str | None = None,
    ) -> list[SearchResult]:
        qv = self.client.embed([query]).embeddings[0]
        return self.store.search(qv, top_k=top_k, min_score=min_score, category=category)

    def answer(self, query: str, *, top_k: int = 3,
               min_score: float = DEFAULT_MIN_SCORE, category: str | None = None) -> str:
        results = self.search(query, top_k=top_k, min_score=min_score, category=category)
        if not results:
            scope = f"「{category}」类目下" if category else ""
            return f"{scope}未找到相关制度（最高相关度低于阈值），建议联系 HR / 行政确认。"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. 《{r.metadata['title']}》[{r.metadata['category']}] "
                f"相关度 {r.score:.2f}\n   {r.text.splitlines()[-1]}"
            )
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="企业制度语义搜索（pgvector）")
    parser.add_argument("query", nargs="?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--category", default=None, help="元数据过滤：财务/人事/IT/行政/法务/安全")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    search = PgPolicySearch()
    if args.rebuild:
        n = search.build_index()
        print(f"已写入 {n} 条制度向量（当前库中共 {search.store.count()} 条，类目：{search.store.categories()}）")

    demo: list[tuple[str, str | None]] = [
        ("出差打车费用怎么报销", None),
        ("报销需要哪些材料", "财务"),
        ("报销需要哪些材料", "IT"),   # 同类目过滤：在 IT 里找报销 → 演示过滤的价值
        ("今天中午吃什么", None),      # 拒答
    ]
    queries = [(args.query, args.category)] if args.query else demo

    output: list[dict] = []
    for q, cat in queries:
        scope = f"（仅 {cat}）" if cat else ""
        print(f"\n🔍 {q}{scope}")
        print(search.answer(q, top_k=args.top_k, min_score=args.min_score, category=cat))
        output.append({
            "query": q, "category": cat,
            "results": [
                {"id": r.id, "title": r.metadata["title"], "category": r.metadata["category"],
                 "score": round(r.score, 4)}
                for r in search.search(q, top_k=args.top_k, category=cat)
            ],
        })

    if not args.query:
        out = ROOT / "data" / "pg_search_demo.json"
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n演示结果已写入 {out}")


if __name__ == "__main__":
    main()
