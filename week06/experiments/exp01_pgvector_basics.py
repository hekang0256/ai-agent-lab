"""实验6-A：pgvector 基础 —— 扩展、向量写入、三种距离操作符。

直接用 SQL 观察 pgvector 的行为，理解：
- `<->`  L2 欧氏距离
- `<#>`  负内积（negative inner product）
- `<=>`  cosine distance（= 1 - 余弦相似度）
归一化向量下，三者的 Top-K 排序有什么异同？
"""
from __future__ import annotations

from apps.policy_search_pg import PgPolicySearch


def main() -> None:
    search = PgPolicySearch()
    store = search.store
    if store.count() == 0:
        print(f"库为空，先建立索引：{search.build_index()} 条")
    print(f"文档数：{store.count()}，类目：{store.categories()}")

    with store._connect() as conn:
        version = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()[0]
        print(f"pgvector 版本：{version}\n")

        query_text = "出差打车费用怎么报销"
        qv = search.client.embed([query_text]).embeddings[0]
        literal = "[" + ",".join(f"{x:.8g}" for x in qv) + "]"

        for op, name in (("<->", "L2 欧氏距离"), ("<#>", "负内积"), ("<=>", "cosine distance")):
            print(f"== {op} {name}：{query_text} ==")
            rows = conn.execute(
                f"SELECT id, title, embedding {op} %s::vector AS distance "
                f"FROM policy_docs ORDER BY embedding {op} %s::vector LIMIT 3",
                (literal, literal),
            ).fetchall()
            for row in rows:
                print(f"   {row[0]} 《{row[1]}》 distance={row[2]:.6f}")
            print()

    print("观察：")
    print("1. 三种操作符的 Top-1 都是《差旅报销制度》（P01）")
    print("2. 向量已归一化时，cosine 与内积的排序一致；L2 与 cosine 在归一化后也单调相关")
    print("3. cosine distance = 1 - 相似度：0 表示完全相同，2 表示完全相反")


if __name__ == "__main__":
    main()
