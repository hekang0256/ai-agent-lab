"""向量与相似度数学基础测试。"""
import pytest

from llm.similarity import (
    cosine_matrix,
    cosine_similarity,
    dot,
    l2_distance,
    norm,
    normalize,
    top_k,
)


def test_dot():
    assert dot([1, 2, 3], [4, 5, 6]) == 32


def test_l2_distance():
    assert l2_distance([0, 0], [3, 4]) == 5.0
    assert l2_distance([1, 2], [1, 2]) == 0.0


def test_cosine_orthogonal():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_same_direction_ignores_length():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)


def test_cosine_opposite():
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_zero_vector_is_safe():
    assert cosine_similarity([0, 0], [1, 2]) == 0.0
    assert cosine_similarity([0, 0], [0, 0]) == 0.0


def test_dimension_mismatch_raises():
    for func in (dot, l2_distance, cosine_similarity):
        with pytest.raises(ValueError):
            func([1, 2], [1, 2, 3])


def test_normalize_unit_length():
    v = normalize([3, 4])
    assert norm(v) == pytest.approx(1.0)
    assert v == [0.6, 0.8]


def test_normalize_zero_vector():
    assert normalize([0, 0]) == [0, 0]


def test_cosine_matrix_symmetric():
    vectors = [[1, 0], [1, 1], [0, 1]]
    m = cosine_matrix(vectors)
    for i in range(3):
        assert m[i][i] == pytest.approx(1.0)
        for j in range(3):
            assert m[i][j] == pytest.approx(m[j][i])


def test_top_k_sorted_and_limited():
    vectors = [[1, 0], [0.9, 0.1], [0, 1], [-1, 0]]
    results = top_k([1, 0], vectors, k=2)
    assert len(results) == 2
    assert results[0][0] == 0  # 最相似的是自身方向
    assert results[0][1] >= results[1][1]


def test_top_k_min_score_filter():
    vectors = [[1, 0], [0, 1]]
    results = top_k([1, 0], vectors, k=5, min_score=0.5)
    assert len(results) == 1
