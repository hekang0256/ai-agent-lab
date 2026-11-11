"""向量与相似度：Embedding 的数学基础（纯 Python，不依赖 numpy）。

本周数据量小（几十~几百条），纯 Python 足够；
数据量大时可用 numpy 替换实现，函数签名不变。
"""
from __future__ import annotations

import math


def _check_same_dim(a: list[float], b: list[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致: {len(a)} vs {len(b)}")


def dot(a: list[float], b: list[float]) -> float:
    """点积：未归一化时同时受「方向」和「长度」影响。"""
    _check_same_dim(a, b)
    return sum(x * y for x, y in zip(a, b))


def l2_distance(a: list[float], b: list[float]) -> float:
    """欧氏距离：越小越相似。对向量长度敏感。"""
    _check_same_dim(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def normalize(v: list[float]) -> list[float]:
    """L2 归一化：把向量缩放到单位长度。零向量原样返回。"""
    n = norm(v)
    if n == 0:
        return list(v)
    return [x / n for x in v]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：只看方向，不看长度，范围 [-1, 1]。

    语义搜索最常用的度量。任一向量为零向量时返回 0.0（无方向可言）。
    """
    _check_same_dim(a, b)
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot(a, b) / (na * nb)


def cosine_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """两两余弦相似度矩阵（对角线恒为 1，除非零向量）。"""
    n = len(vectors)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = cosine_similarity(vectors[i], vectors[j])
            matrix[i][j] = s
            matrix[j][i] = s
    return matrix


def top_k(
    query: list[float],
    vectors: list[list[float]],
    k: int = 5,
    *,
    min_score: float = -1.0,
) -> list[tuple[int, float]]:
    """暴力 Top-K：返回 (下标, 余弦相似度)，按分数降序。

    暴力扫描是 O(N·D)。N 到十万级以上就该用 ANN 索引（HNSW 等），
    或直接交给 pgvector（week06）。
    """
    scored = [
        (i, cosine_similarity(query, v)) for i, v in enumerate(vectors)
    ]
    scored = [(i, s) for i, s in scored if s >= min_score]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
