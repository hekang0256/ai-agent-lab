"""RAG Chunk 的 pgvector 存储：写入、Top-K、Metadata Filter 均参数化执行。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DSN = "postgresql://lab:lab123@localhost:5434/ai_agent_lab_rag"


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.8g}" for value in vector) + "]"


class PgChunkStore:
    def __init__(self, dsn: str | None = None, *, table: str = "rag_chunks"):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError(f"非法表名: {table}")
        self.dsn = dsn or os.getenv("RAG_DATABASE_URL", DEFAULT_DSN)
        self.table = table

    def _connect(self):
        return psycopg.connect(self.dsn)

    def init_schema(self, sql_path: str | Path) -> None:
        with self._connect() as conn:
            conn.execute(Path(sql_path).read_text(encoding="utf-8"))

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {self.table}")

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def get(self, id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT id, document_id, title, category, source, chunk_index, text FROM {self.table} WHERE id = %s",
                (id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0], "document_id": row[1], "title": row[2], "category": row[3],
            "source": row[4], "chunk_index": row[5], "text": row[6],
        }

    def upsert_many(self, records: list[dict]) -> None:
        sql = f"""
            INSERT INTO {self.table}
                (id, document_id, title, category, source, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                source = EXCLUDED.source,
                chunk_index = EXCLUDED.chunk_index,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
        with self._connect() as conn:
            with conn.cursor() as cursor:
                for record in records:
                    metadata = record["metadata"]
                    cursor.execute(sql, (
                        record["id"], metadata["document_id"], metadata["title"],
                        metadata["category"], metadata["source"], metadata["chunk_index"],
                        record["text"], _vector_literal(record["vector"]),
                    ))

    def search(
        self,
        query_vector: list[float], *, top_k: int = 3,
        min_score: float = -1.0, category: str | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        query = _vector_literal(query_vector)
        category_clause = "AND category = %s" if category is not None else ""
        params: list = [query]
        if category is not None:
            params.append(category)
        params.extend([query, 1.0 - min_score, query, top_k])
        sql = f"""
            SELECT id, text, 1 - (embedding <=> %s::vector) AS score,
                   document_id, title, category, source, chunk_index
            FROM {self.table}
            WHERE 1 = 1
              {category_clause}
              AND embedding <=> %s::vector <= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            SearchResult(id=row[0], text=row[1], score=float(row[2]), metadata={
                "document_id": row[3], "title": row[4], "category": row[5],
                "source": row[6], "chunk_index": row[7],
            })
            for row in rows
        ]
