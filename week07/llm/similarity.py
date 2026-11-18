"""纯 Python 向量计算，与 week05/06 保持一致。"""
from __future__ import annotations

import math


def dot(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator else 0.0


def normalize(a: list[float]) -> list[float]:
    length = norm(a)
    return [x / length for x in a] if length else list(a)
