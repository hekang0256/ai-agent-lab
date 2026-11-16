"""PgVectorStore：PostgreSQL + pgvector 实现的向量库（week06 主角）。

与 week05 内存 VectorStore 相同的检索语义（余弦 Top-K + min_score），
但存储、过滤、排序全部下推到数据库：

- `<=>`  cosine distance（= 1 - 余弦相似度），配合 vector_cosine_ops 索引
- Top-K  = ORDER BY embedding <=> %s LIMIT k（数据库内排序，不全量拉回内存）
- Filter = WHERE category = %s（元数据过滤与向量检索一条 SQL 完成）
- 索引   = HNSW / IVFFlat，EXPLAIN 可观测；ef_search 可调

工程说明：
1. 向量用文本 '[x,y,...]' + ::vector 传参，不依赖 pgvector-python 适配包；
2. 表名只允许 [a-z0-9_]，DDL 参数（m/lists 等）一律 int() 强转后拼接，杜绝注入；
3. 每次操作短连接（with connect 自动 commit/close），简单可靠；
   生产高并发应换连接池（psycopg_pool），接口不变。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DSN = "postgresql://lab:lab123@localhost:5433/ai_agent_lab_vector"
INDEX_NAME = "idx_policy_docs_embedding"


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def _vector_literal(vector: list[float]) -> str:
    """[0.1, 0.2] → '[0.1,0.2]'，配合 SQL 里的 %s::vector 使用。"""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"


class PgVectorStore:
    def __init__(self, dsn: str | None = None, *, table: str = "policy_docs"):
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", table):
            raise ValueError(f"非法表名: {table}")
        self.dsn = dsn or os.getenv("DATABASE_URL", DEFAULT_DSN)
        self.table = table

    def _connect(self):
        return psycopg.connect(self.dsn)

    # ----------  schema ----------
    def init_schema(self, sql_path: str | Path) -> None:
        """执行建库脚本（扩展 + 表）。容器首次启动已自动执行，通常无需调用。"""
        sql = Path(sql_path).read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.execute(sql)

    # ---------- 写入 ----------
    def upsert(self, id: str, title: str, category: str, text: str, vector: list[float]) -> None:
        sql = f"""
            INSERT INTO {self.table} (id, title, category, text, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
        with self._connect() as conn:
            conn.execute(sql, (id, title, category, text, _vector_literal(vector)))

    def upsert_many(self, records: list[dict]) -> None:
        """单事务批量 upsert。record 字段：id/title/category/text/vector。"""
        sql = f"""
            INSERT INTO {self.table} (id, title, category, text, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                text = EXCLUDED.text,
                embedding = EXCLUDED.embedding,
                updated_at = now()
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                for r in records:
                    cur.execute(sql, (r["id"], r["title"], r["category"], r["text"],
                                      _vector_literal(r["vector"])))

    def get(self, id: str) -> dict | None:
        sql = f"SELECT id, title, category, text FROM {self.table} WHERE id = %s"
        with self._connect() as conn:
            row = conn.execute(sql, (id,)).fetchone()
        if row is None:
            return None
        return {"id": row[0], "title": row[1], "category": row[2], "text": row[3]}

    def delete(self, id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(f"DELETE FROM {self.table} WHERE id = %s", (id,))
            return cur.rowcount > 0

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute(f"DELETE FROM {self.table}")

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]

    def categories(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT category FROM {self.table} ORDER BY category"
            ).fetchall()
        return [r[0] for r in rows]

    # ---------- 检索 ----------
    def _search_sql(self, *, filtered: bool, scored: bool) -> str:
        where = []
        if filtered:
            where.append("category = %s")
        if scored:
            where.append("embedding <=> %s::vector <= %s")  # distance <= 1 - min_score
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        return f"""
            SELECT id, title, category, text, 1 - (embedding <=> %s::vector) AS score
            FROM {self.table}
            {clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        min_score: float = -1.0,
        category: str | None = None,
    ) -> list[SearchResult]:
        q = _vector_literal(query_vector)
        sql = self._search_sql(filtered=category is not None, scored=True)
        params: list = [q]
        if category is not None:
            params.append(category)
        params += [q, 1.0 - min_score, q, top_k]
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            SearchResult(
                id=row[0], text=row[3], score=float(row[4]),
                metadata={"title": row[1], "category": row[2]},
            )
            for row in rows
        ]

    # ---------- 索引与执行计划 ----------
    def create_index(self, method: str = "hnsw", **options) -> str:
        """创建向量索引。hnsw(m, ef_construction) / ivfflat(lists)。"""
        method = method.lower()
        if method == "hnsw":
            m = int(options.get("m", 16))
            ef = int(options.get("ef_construction", 64))
            with_clause = f"m = {m}, ef_construction = {ef}"
        elif method == "ivfflat":
            lists = int(options.get("lists", 100))
            with_clause = f"lists = {lists}"
        else:
            raise ValueError(f"未知索引类型: {method}")
        sql = (
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON {self.table} "
            f"USING {method} (embedding vector_cosine_ops) WITH ({with_clause})"
        )
        with self._connect() as conn:
            conn.execute(sql)
        return sql

    def drop_index(self) -> None:
        with self._connect() as conn:
            conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")

    def set_ef_search(self, value: int) -> None:
        """HNSW 查询期参数：越大召回越准但越慢。对当前连接生效。"""
        with self._connect() as conn:
            conn.execute("SELECT set_config('hnsw.ef_search', %s, false)", (str(int(value)),))

    def explain(self, query_vector: list[float], *, top_k: int = 5,
                category: str | None = None, force_index: bool = False) -> str:
        """返回检索 SQL 的执行计划文本（EXPLAIN COSTS FALSE）。

        小表上优化器往往直接选 Seq Scan（索引不划算），
        force_index=True 时先 SET enable_seqscan=off，用于教学演示索引扫描路径。
        """
        q = _vector_literal(query_vector)
        sql = self._search_sql(filtered=category is not None, scored=False)
        params: list = [q]
        if category is not None:
            params.append(category)
        params += [q, top_k]
        with self._connect() as conn:
            if force_index:
                conn.execute("SELECT set_config('enable_seqscan', 'off', false)")
            rows = conn.execute("EXPLAIN (COSTS FALSE) " + sql, params).fetchall()
        return "\n".join(r[0] for r in rows)
