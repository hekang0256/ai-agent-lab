"""VectorStore 测试：upsert / 检索 / 阈值过滤 / 持久化。"""
import pytest

from llm.vector_store import VectorStore


def make_store() -> VectorStore:
    store = VectorStore()
    store.add("a", "文本a", [1.0, 0.0], {"title": "A"})
    store.add("b", "文本b", [0.9, 0.1], {"title": "B"})
    store.add("c", "文本c", [0.0, 1.0], {"title": "C"})
    return store


def test_add_and_get():
    store = make_store()
    assert len(store) == 3
    assert store.get("a").metadata["title"] == "A"
    assert store.dim == 2


def test_upsert_overwrites_same_id():
    store = make_store()
    store.add("a", "新文本", [0.0, 1.0])
    assert len(store) == 3
    assert store.get("a").text == "新文本"


def test_dimension_mismatch_raises():
    store = make_store()
    with pytest.raises(ValueError):
        store.add("d", "维度错误", [1.0, 0.0, 0.0])


def test_search_sorted_and_top_k():
    store = make_store()
    results = store.search([1.0, 0.0], top_k=2)
    assert [r.id for r in results] == ["a", "b"]
    assert results[0].score >= results[1].score
    assert results[0].metadata["title"] == "A"


def test_search_min_score_filters():
    store = make_store()
    results = store.search([1.0, 0.0], top_k=5, min_score=0.999)  # b 的分数≈0.994，也被过滤
    assert [r.id for r in results] == ["a"]


def test_search_empty_store():
    assert VectorStore().search([1.0, 0.0]) == []


def test_delete():
    store = make_store()
    assert store.delete("a") is True
    assert store.delete("a") is False
    assert len(store) == 2


def test_clear():
    store = make_store()
    store.clear()
    assert len(store) == 0
    assert store.dim == 0


def test_save_load_roundtrip(tmp_path):
    store = make_store()
    path = tmp_path / "index.json"
    store.save(path)
    loaded = VectorStore.load(path)
    assert len(loaded) == 3
    assert loaded.dim == 2
    assert loaded.get("b").text == "文本b"
    # 加载后检索结果与原库一致
    original = [(r.id, round(r.score, 6)) for r in store.search([1.0, 0.05], top_k=3)]
    restored = [(r.id, round(r.score, 6)) for r in loaded.search([1.0, 0.05], top_k=3)]
    assert original == restored
