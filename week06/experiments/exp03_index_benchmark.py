"""实验6-C：索引对照 —— Seq Scan vs HNSW + EXPLAIN 执行计划 + 内存库一致性。

步骤：
1. 无索引：EXPLAIN（Seq Scan）+ 计时
2. 建 HNSW：EXPLAIN（小表仍可能 Seq Scan，优化器认为索引不划算）
3. SET enable_seqscan=off：强制展示 Index Scan 路径（教学用）
4. 与 week05 内存 VectorStore 做结果一致性校验
5. 输出 data/benchmark_results.json

注意：14 条数据看不出性能差异，实验目的是读懂执行计划；
索引的真正价值在十万级以上数据（暴力 O(N·D) vs ANN 近似对数级）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from apps.policy_search_pg import PgPolicySearch
from llm.vector_store import VectorStore

ROOT = Path(__file__).resolve().parents[1]
RUNS = 30


def time_search(store, qv, runs=RUNS) -> float:
    started = time.perf_counter()
    for _ in range(runs):
        store.search(qv, top_k=3)
    return (time.perf_counter() - started) / runs * 1000


def main() -> None:
    search = PgPolicySearch()
    store = search.store
    if store.count() == 0:
        search.build_index()

    query = "出差打车费用怎么报销"
    qv = search.client.embed([query]).embeddings[0]

    # 1. 无索引
    store.drop_index()
    plan_no_index = store.explain(qv, top_k=3)
    ms_no_index = time_search(store, qv)
    print("== 无索引执行计划 ==\n" + plan_no_index)
    print(f"平均耗时：{ms_no_index:.2f} ms/次（含连接开销）\n")

    # 2. HNSW 索引
    ddl = store.create_index("hnsw", m=16, ef_construction=64)
    print(f"已创建索引：\n{ddl}\n")
    plan_hnsw = store.explain(qv, top_k=3)
    ms_hnsw = time_search(store, qv)
    print("== HNSW 执行计划（小表可能仍选 Seq Scan） ==\n" + plan_hnsw)
    print(f"平均耗时：{ms_hnsw:.2f} ms/次\n")

    # 3. 强制索引路径（教学演示）
    plan_forced = store.explain(qv, top_k=3, force_index=True)
    print("== SET enable_seqscan=off 后（强制索引路径） ==\n" + plan_forced + "\n")

    # 4. HNSW 查询期参数
    store.set_ef_search(100)
    print("已设置 hnsw.ef_search=100（查询期参数：越大越准、越慢）\n")

    # 5. 与内存库一致性校验
    memory = VectorStore()
    import json as _json
    docs = _json.loads((ROOT / "data" / "docs" / "policies.json").read_text(encoding="utf-8"))
    texts = [f"{d['title']}\n{d['text']}" for d in docs]
    vectors = search.client.embed(texts).embeddings
    for d, text, v in zip(docs, texts, vectors):
        memory.add(d["id"], text, v, {"title": d["title"], "category": d["category"]})
    pg_top = [r.id for r in store.search(qv, top_k=3)]
    mem_top = [r.id for r in memory.search(qv, top_k=3)]
    consistent = pg_top == mem_top
    print(f"pg Top-3：{pg_top}")
    print(f"内存 Top-3：{mem_top}")
    print(f"结果一致：{consistent}")

    result = {
        "query": query,
        "runs": RUNS,
        "no_index": {"avg_ms": round(ms_no_index, 3), "plan": plan_no_index},
        "hnsw": {"avg_ms": round(ms_hnsw, 3), "plan": plan_hnsw, "forced_plan": plan_forced},
        "pg_top3": pg_top,
        "memory_top3": mem_top,
        "consistent_with_memory": consistent,
        "note": "14 条小数据看不出索引性能差异；实验重点是执行计划与结果一致性。",
    }
    out = ROOT / "data" / "benchmark_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {out}")


if __name__ == "__main__":
    main()
