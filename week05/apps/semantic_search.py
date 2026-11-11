"""应用1：企业制度语义搜索（本周主应用）。

流程：制度文档 → Embedding → 向量库 → 查询向量 → 余弦 Top-K → 低于阈值则拒答。
这是 RAG 的「检索」半边（week07 再接入 LLM 生成与引用）。

用法：
  python -m apps.semantic_search                    # 演示 4 个预置查询
  python -m apps.semantic_search "打车费怎么报销"     # 单次查询
  python -m apps.semantic_search --rebuild          # 重建并保存索引 data/policy_index.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.embedding import get_embedding_client
from llm.vector_store import SearchResult, VectorStore

ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "docs" / "policies.json"
INDEX_PATH = ROOT / "data" / "policy_index.json"

# 低于该分数认为「没有相关制度」，直接拒答而不是硬凑答案
DEFAULT_MIN_SCORE = 0.35

DEMO_QUERIES = [
    "出差打车费用怎么报销",
    "年假能休几天",
    "服务器权限怎么申请",
    "今天中午吃什么",  # 无关问题：演示拒答
]


class PolicySearch:
    def __init__(self, client=None):
        self.client = client or get_embedding_client()
        self.store = VectorStore()

    # ---------- 建索引 ----------
    def build_index(self, docs_path: str | Path = DOCS_PATH) -> int:
        docs = json.loads(Path(docs_path).read_text(encoding="utf-8"))
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        resp = self.client.embed(texts)  # 一次批量调用，比逐条省 N-1 次网络往返
        for d, text, vector in zip(docs, texts, resp.embeddings):
            self.store.add(
                d["id"], text, vector,
                metadata={"title": d["title"], "category": d["category"]},
            )
        return len(docs)

    def save_index(self, path: str | Path = INDEX_PATH) -> None:
        self.store.save(path)

    def load_index(self, path: str | Path = INDEX_PATH) -> None:
        self.store = VectorStore.load(path)

    # ---------- 检索 ----------
    def search(self, query: str, *, top_k: int = 3, min_score: float = 0.0) -> list[SearchResult]:
        qv = self.client.embed([query]).embeddings[0]
        return self.store.search(qv, top_k=top_k, min_score=min_score)

    def answer(self, query: str, *, top_k: int = 3, min_score: float = DEFAULT_MIN_SCORE) -> str:
        """带拒答的检索式回答：有相关制度则列出，没有就明说不知道。"""
        results = self.search(query, top_k=top_k, min_score=min_score)
        if not results:
            return "未找到相关制度（最高相关度低于阈值），建议联系 HR / 行政确认。"
        lines = []
        for i, r in enumerate(results, 1):
            title = r.metadata.get("title", r.id)
            lines.append(f"{i}. 《{title}》 相关度 {r.score:.2f}\n   {r.text.splitlines()[-1]}")
        return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="企业制度语义搜索")
    parser.add_argument("query", nargs="?", help="查询问题（缺省跑演示查询）")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--rebuild", action="store_true", help="重建并保存索引")
    args = parser.parse_args()

    search = PolicySearch()
    if args.rebuild or not INDEX_PATH.is_file():
        n = search.build_index()
        search.save_index()
        print(f"已建立索引：{n} 条制度 → {INDEX_PATH}（维度 {search.store.dim}）")
    else:
        search.load_index()

    queries = [args.query] if args.query else DEMO_QUERIES
    demo_results: dict[str, list] = {}
    for q in queries:
        print(f"\n🔍 {q}")
        print(search.answer(q, top_k=args.top_k, min_score=args.min_score))
        demo_results[q] = [
            {"id": r.id, "title": r.metadata.get("title"), "score": round(r.score, 4)}
            for r in search.search(q, top_k=args.top_k)
        ]

    if not args.query:
        out = ROOT / "data" / "search_demo.json"
        out.write_text(json.dumps(demo_results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n演示结果已写入 {out}")


if __name__ == "__main__":
    main()
