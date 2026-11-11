"""MockEmbeddingClient 行为测试：确定性、归一化、主题语义可分。"""
import pytest

from llm.embedding import MockEmbeddingClient, get_embedding_client
from llm.similarity import cosine_similarity, norm


def test_deterministic():
    client = MockEmbeddingClient()
    a = client.embed(["差旅报销流程"]).embeddings[0]
    b = client.embed(["差旅报销流程"]).embeddings[0]
    assert a == b


def test_dim_and_batch():
    client = MockEmbeddingClient(dim=256)
    resp = client.embed(["报销", "年假", "服务器"])
    assert resp.dim == 256
    assert len(resp.embeddings) == 3
    assert resp.total_tokens > 0


def test_vectors_are_normalized():
    client = MockEmbeddingClient()
    for v in client.embed(["报销打车费", "今天天气怎么样"]).embeddings:
        assert norm(v) == pytest.approx(1.0, abs=1e-6)


def test_different_texts_different_vectors():
    client = MockEmbeddingClient()
    a, b = client.embed(["报销打车费", "年假有几天"]).embeddings
    assert a != b


def test_same_topic_high_similarity():
    client = MockEmbeddingClient()
    a, b = client.embed(["出差打车费用怎么报销", "打车发票可以报销吗"]).embeddings
    assert cosine_similarity(a, b) > 0.8


def test_cross_topic_low_similarity():
    client = MockEmbeddingClient()
    a, b = client.embed(["出差打车费用怎么报销", "保密与竞业限制规定"]).embeddings
    assert cosine_similarity(a, b) < 0.35


def test_empty_input():
    resp = MockEmbeddingClient().embed([])
    assert resp.embeddings == []
    assert resp.dim == 0


def test_dim_too_small_rejected():
    with pytest.raises(ValueError):
        MockEmbeddingClient(dim=8)


def test_get_embedding_client_returns_mock_in_mock_mode():
    assert isinstance(get_embedding_client(), MockEmbeddingClient)
