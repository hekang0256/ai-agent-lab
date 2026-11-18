"""内存向量库：与 PgChunkStore 同接口，供离线实验和单元测试使用。"""
from __future__ import annotations

from dataclasses import dataclass, field

from .similarity import cosine_similarity


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


class MemoryVectorStore:
    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    def upsert_many(self, records: list[dict]) -> None:
        for record in records:
            vector = record["vector"]
            if self._records and len(vector) != len(next(iter(self._records.values())).vector):
                raise ValueError("向量维度与已有索引不一致")
            self._records[record["id"]] = VectorRecord(
                id=record["id"], text=record["text"], vector=list(vector),
                metadata=dict(record.get("metadata", {})),
            )

    def get(self, id: str) -> VectorRecord | None:
        return self._records.get(id)

    def search(
        self,
        query_vector: list[float], *, top_k: int = 5,
        min_score: float = -1.0, category: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        results = []
        for record in self._records.values():
            if category is not None and record.metadata.get("category") != category:
                continue
            score = cosine_similarity(query_vector, record.vector)
            if score >= min_score:
                results.append(SearchResult(record.id, record.text, score, dict(record.metadata)))
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]
