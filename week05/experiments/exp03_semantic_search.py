"""实验5-C：语义搜索最小闭环（索引 → 检索 → 拒答）。

观察点：
1. 查询与文档字面完全不同也能命中（语义匹配 vs 关键词匹配）
2. Top-K 分数降序，第一名是不是该是它？
3. 无关问题被阈值拦下（拒答是 RAG 系统的重要能力）
"""
from __future__ import annotations

from apps.semantic_search import DEMO_QUERIES, PolicySearch


def main() -> None:
    search = PolicySearch()
    n = search.build_index()
    print(f"索引：{n} 条制度，向量维度 {search.store.dim}\n")

    for q in DEMO_QUERIES:
        print(f"🔍 {q}")
        results = search.search(q, top_k=3)
        for r in results:
            print(f"   {r.score:.3f}  《{r.metadata.get('title')}》({r.metadata.get('category')})")
        top_score = results[0].score if results else 0.0
        verdict = "✅ 返回结果" if top_score >= 0.35 else "🚫 拒答"
        print(f"   → {verdict}（最高分 {top_score:.3f}，阈值 0.35）\n")


if __name__ == "__main__":
    main()
