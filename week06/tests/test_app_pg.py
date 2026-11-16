"""应用层测试：PgPolicySearch + 与 week05 内存库的一致性对照。"""
import pytest

from apps.policy_search_pg import PgPolicySearch
from llm.embedding import MockEmbeddingClient
from llm.vector_store import VectorStore


@pytest.fixture()
def app(store):
    application = PgPolicySearch(MockEmbeddingClient(), store=store)
    application.build_index()
    return application


def test_build_index(app):
    assert app.store.count() == 14


def test_expense_query_top1(app):
    results = app.search("出差打车费用怎么报销", top_k=3)
    assert results[0].id == "P01"
    assert results[0].score > 0.9


def test_category_filter(app):
    results = app.search("报销需要哪些材料", top_k=5, category="财务")
    assert results
    assert all(r.metadata["category"] == "财务" for r in results)


def test_irrelevant_query_rejected(app):
    assert app.search("今天中午吃什么", top_k=3, min_score=0.35) == []
    assert "未找到相关制度" in app.answer("今天中午吃什么")


def test_answer_with_category_scope(app):
    text = app.answer("报销流程", category="IT", min_score=0.95)
    # IT 类目下没有高分报销制度 → 按类目范围提示拒答
    assert "IT" in text or "VPN" in text


def test_consistent_with_memory_store(app):
    """同一批向量，pg 与 week05 内存库的 Top-3 必须一致。"""
    import json
    from pathlib import Path
    docs = json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "docs" / "policies.json")
        .read_text(encoding="utf-8")
    )
    memory = VectorStore()
    texts = [f"{d['title']}\n{d['text']}" for d in docs]
    vectors = app.client.embed(texts).embeddings
    for d, text, v in zip(docs, texts, vectors):
        memory.add(d["id"], text, v, {"title": d["title"], "category": d["category"]})

    qv = app.client.embed(["年假能休几天"]).embeddings[0]
    pg_top = [r.id for r in app.store.search(qv, top_k=3)]
    mem_top = [r.id for r in memory.search(qv, top_k=3)]
    assert pg_top == mem_top
    assert pg_top[0] == "P03"
