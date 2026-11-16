"""PgVectorStore 测试（真实 PostgreSQL + pgvector，DB 不可用则跳过）。"""
import pytest

from llm.pg_store import PgVectorStore


def test_upsert_get_count(seeded_store):
    assert seeded_store.count() == 4
    doc = seeded_store.get("D1")
    assert doc["title"] == "差旅报销制度"
    assert doc["category"] == "财务"


def test_upsert_overwrites(seeded_store, embedding_client):
    new_text = "差旅报销制度（修订版）：电子发票同样有效。"
    vector = embedding_client.embed([new_text]).embeddings[0]
    seeded_store.upsert("D1", "差旅报销制度", "财务", new_text, vector)
    assert seeded_store.count() == 4  # upsert 不新增行
    assert "修订版" in seeded_store.get("D1")["text"]


def test_search_sorted_and_topk(seeded_store, embedding_client):
    qv = embedding_client.embed(["打车费用怎么报销"]).embeddings[0]
    results = seeded_store.search(qv, top_k=2)
    assert len(results) == 2
    # D2 只命中 expense 主题（方向更集中，0.9918）；
    # D1 还命中 travel 主题（出差），相似度被摊薄（0.9838）→ D2 排第一
    assert results[0].id == "D2"
    assert {r.id for r in results} == {"D1", "D2"}
    assert results[0].score >= results[1].score
    assert results[0].metadata["category"] == "财务"


def test_search_min_score_filter(seeded_store, embedding_client):
    qv = embedding_client.embed(["今天中午吃什么"]).embeddings[0]
    assert seeded_store.search(qv, top_k=4, min_score=0.35) == []
    # 不设阈值时能返回（分数低但存在）
    assert len(seeded_store.search(qv, top_k=4)) > 0


def test_search_category_filter(seeded_store, embedding_client):
    qv = embedding_client.embed(["报销需要哪些材料"]).embeddings[0]
    results = seeded_store.search(qv, top_k=4, category="财务")
    assert results
    assert all(r.metadata["category"] == "财务" for r in results)
    # IT 类目下没有报销制度：不应出现财务文档
    it_results = seeded_store.search(qv, top_k=4, category="IT")
    assert all(r.metadata["category"] == "IT" for r in it_results)


def test_delete_and_clear(seeded_store):
    assert seeded_store.delete("D1") is True
    assert seeded_store.delete("D1") is False
    assert seeded_store.count() == 3
    seeded_store.clear()
    assert seeded_store.count() == 0
    assert seeded_store.search([0.1] * 256) == []


def test_categories(seeded_store):
    assert set(seeded_store.categories()) == {"财务", "人事", "IT"}


def test_explain_returns_plan(seeded_store, embedding_client):
    qv = embedding_client.embed(["报销"]).embeddings[0]
    plan = seeded_store.explain(qv, top_k=3)
    assert "Seq Scan" in plan or "Index" in plan


def test_create_and_drop_index(seeded_store, embedding_client):
    ddl = seeded_store.create_index("hnsw", m=16, ef_construction=64)
    assert "USING hnsw" in ddl
    qv = embedding_client.embed(["报销"]).embeddings[0]
    # 强制索引路径时应出现索引扫描（教学演示）
    plan = seeded_store.explain(qv, top_k=3, force_index=True)
    assert "Index" in plan
    # 建索引后检索结果与建索引前一致（正确性不受索引影响）
    before = [r.id for r in seeded_store.search(qv, top_k=4)]
    seeded_store.drop_index()
    after = [r.id for r in seeded_store.search(qv, top_k=4)]
    assert before == after


def test_ivfflat_index(seeded_store):
    ddl = seeded_store.create_index("ivfflat", lists=4)
    assert "USING ivfflat" in ddl
    seeded_store.drop_index()


def test_set_ef_search(seeded_store):
    seeded_store.set_ef_search(100)  # 不抛异常即可


def test_invalid_table_name():
    with pytest.raises(ValueError):
        PgVectorStore(table="bad-table; DROP TABLE users;--")
