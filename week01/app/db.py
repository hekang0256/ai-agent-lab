"""数据库连接与查询辅助层（psycopg 3）。"""
import os

import psycopg
from psycopg.rows import dict_row

DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://lab:lab123@localhost:5432/ai_agent_lab",
)


def get_conn() -> psycopg.Connection:
    """返回一条到 PostgreSQL 的连接（行以 dict 形式返回）。"""
    return psycopg.connect(DSN, row_factory=dict_row)


def fetch_all(sql: str, params: tuple | None = None) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchall()


def fetch_one(sql: str, params: tuple | None = None) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            return cur.fetchone()


def execute(sql: str, params: tuple | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
