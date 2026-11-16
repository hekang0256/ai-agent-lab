"""实验6-B：Top-K + Metadata Filter —— 语义检索与业务过滤一条 SQL 完成。

对照：
1. 全库检索「报销需要哪些材料」
2. 仅「财务」类目检索（排除其它类目的干扰项）
3. 仅「IT」类目检索（语义相近但业务上不相关 → 分数很低）
4. 无关问题 + min_score 阈值 → 拒答
"""
from __future__ import annotations

from apps.policy_search_pg import PgPolicySearch


def show(search: PgPolicySearch, query: str, *, category=None, top_k=5, min_score=0.0):
    scope = f"（仅 {category}）" if category else "（全库）"
    print(f"\n🔍 {query}{scope}")
    results = search.search(query, top_k=top_k, category=category, min_score=min_score)
    if not results:
        print("   （无结果）")
    for r in results:
        print(f"   {r.score:.3f}  [{r.metadata['category']}] 《{r.metadata['title']}》")
    return results


def main() -> None:
    search = PgPolicySearch()
    if search.store.count() == 0:
        search.build_index()

    q = "报销需要哪些材料"
    show(search, q)
    show(search, q, category="财务")
    show(search, q, category="IT")

    print("\n== 阈值拒答 ==")
    show(search, "今天中午吃什么", min_score=0.35)

    print("\n要点：")
    print("- Metadata Filter 不是检索后再过滤，而是 WHERE 与 ORDER BY 同一条 SQL 下推到数据库")
    print("- 业务上「类目不符」的结果即使语义相近也不该返回（IT 类目里找报销）")
    print("- min_score 在 SQL 里就是 embedding <=> q <= 1 - min_score，拒绝硬凑答案")


if __name__ == "__main__":
    main()
