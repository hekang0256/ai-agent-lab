import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_USE_MOCK"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.embedding import MockEmbeddingClient
from llm.pg_store import PgVectorStore

SEED_DOCS = [
    {"id": "D1", "title": "差旅报销制度", "category": "财务",
     "text": "员工出差的机票、住宿、打车费用可凭发票报销。"},
    {"id": "D2", "title": "日常费用报销规定", "category": "财务",
     "text": "办公用品与聚餐费用报销需在三十天内提交发票。"},
    {"id": "D3", "title": "休假与考勤制度", "category": "人事",
     "text": "正式员工每年享有带薪年假，病假事假需提前请假。"},
    {"id": "D4", "title": "VPN 与服务器使用", "category": "IT",
     "text": "远程访问内网必须使用公司 VPN，服务器操作需申请权限。"},
]


@pytest.fixture()
def store():
    """每个测试一个干净的表；DB 不可用时整组跳过。"""
    s = PgVectorStore()
    try:
        with s._connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"pgvector 不可用：{e}（请先执行 docker compose up -d）")
    s.clear()
    yield s
    s.clear()


@pytest.fixture()
def seeded_store(store):
    client = MockEmbeddingClient()
    vectors = client.embed([d["text"] for d in SEED_DOCS]).embeddings
    store.upsert_many([
        {**d, "vector": v} for d, v in zip(SEED_DOCS, vectors)
    ])
    return store


@pytest.fixture()
def embedding_client():
    return MockEmbeddingClient()
