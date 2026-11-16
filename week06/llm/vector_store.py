"""极简向量库：内存存储 + 暴力余弦 Top-K + JSON 持久化。

定位：本周（Embedding）用它理解「向量入库 → 相似度检索」的最小闭环；
下周（week06）把同样的接口换成 pgvector，获得索引加速与 Metadata Filter。
因此本模块的接口设计刻意向数据库看齐：upsert / search / delete / save / load。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .similarity import cosine_similarity


@dataclass
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore:
    def __init__(self):
        self._records: dict[str, VectorRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def dim(self) -> int:
        return len(next(iter(self._records.values())).vector) if self._records else 0

    # ---------- 写入 ----------
    def add(self, id: str, text: str, vector: list[float], metadata: dict | None = None) -> None:
        """upsert：同 id 覆盖。维度必须与库中已有向量一致。"""
        if self._records and len(vector) != self.dim:
            raise ValueError(f"向量维度 {len(vector)} 与库维度 {self.dim} 不一致")
        self._records[id] = VectorRecord(id=id, text=text, vector=list(vector), metadata=metadata or {})

    def add_many(self, records: list[VectorRecord]) -> None:
        for r in records:
            self.add(r.id, r.text, r.vector, r.metadata)

    def get(self, id: str) -> VectorRecord | None:
        return self._records.get(id)

    def delete(self, id: str) -> bool:
        return self._records.pop(id, None) is not None

    def clear(self) -> None:
        self._records.clear()

    # ---------- 检索 ----------
    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = -1.0,
    ) -> list[SearchResult]:
        """余弦相似度 Top-K，降序；低于 min_score 的结果被过滤（用于拒答）。"""
        if not self._records:
            return []
        results = []
        for r in self._records.values():
            score = cosine_similarity(query_vector, r.vector)
            if score >= min_score:
                results.append(SearchResult(id=r.id, text=r.text, score=score, metadata=dict(r.metadata)))
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ---------- 持久化 ----------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dim": self.dim,
            "records": [
                {"id": r.id, "text": r.text, "vector": r.vector, "metadata": r.metadata}
                for r in self._records.values()
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls()
        for r in payload["records"]:
            store.add(r["id"], r["text"], r["vector"], r["metadata"])
        return store
