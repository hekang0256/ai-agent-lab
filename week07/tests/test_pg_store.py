"""真实 pgvector 集成测试；Docker 未启动时明确跳过，不伪装通过。"""
import pytest

from llm.document import SourceDocument
from llm.embedding import MockEmbeddingClient
from llm.pg_store import PgChunkStore
from llm.rag import RAGService


@pytest.fixture()
def pg_store():
    """使用独立测试表，绝不 clear 主表 rag_chunks。"""
    store = PgChunkStore(table="rag_chunks_test")
    try:
        with store._connect() as conn:
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
            ).fetchone()
            dim = int(row[0]) if row and row[0] > 0 else 256
            conn.execute("DROP TABLE IF EXISTS rag_chunks_test")
            conn.execute(f"""
                CREATE TABLE rag_chunks_test (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    title TEXT NOT NULL, category TEXT NOT NULL,
                    source TEXT NOT NULL, chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL, embedding vector({dim}) NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
    except Exception as error:
        pytest.skip(f"pgvector 不可用：{error}（请先 docker compose up -d）")
    yield store
    with store._connect() as conn:
        conn.execute("DROP TABLE IF EXISTS rag_chunks_test")


def _table_dim(store) -> int:
    """读取测试表 embedding 的实际维度，让 Mock 向量与表结构保持一致。"""
    with store._connect() as conn:
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = to_regclass(%s) AND attname = 'embedding'",
            (store.table,),
        ).fetchone()
    return int(row[0]) if row and row[0] > 0 else 256


def test_pg_rag_round_trip(pg_store):
    service = RAGService(pg_store, embedding_client=MockEmbeddingClient(dim=_table_dim(pg_store)))
    stats = service.index_documents([
        SourceDocument("D1", "差旅制度", "出差打车费用凭发票报销。", category="财务"),
        SourceDocument("D2", "VPN 制度", "远程访问内网必须使用 VPN。", category="IT"),
    ], clear=True)
    assert stats.chunks == 2
    assert pg_store.count() == 2
    result = service.answer("打车怎么报销")
    assert result.citations[0].document_id == "D1"
    assert pg_store.get("D1:chunk:000")["category"] == "财务"


def test_pg_metadata_filter_and_clear(pg_store):
    service = RAGService(pg_store, embedding_client=MockEmbeddingClient(dim=_table_dim(pg_store)))
    service.index_documents([
        SourceDocument("D1", "差旅制度", "出差打车费用凭发票报销。", category="财务"),
        SourceDocument("D2", "VPN 制度", "远程访问内网必须使用 VPN。", category="IT"),
    ])
    results = service.retrieve("VPN", category="IT")
    assert results and all(item.metadata["category"] == "IT" for item in results)
    pg_store.clear()
    assert pg_store.count() == 0


def test_invalid_table_name():
    with pytest.raises(ValueError):
        PgChunkStore(table="bad;DROP TABLE rag_chunks")
