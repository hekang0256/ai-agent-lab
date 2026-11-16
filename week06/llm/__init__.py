"""week06 LLM 工具箱：pgvector 向量检索 + week05 的向量/Embedding 基础。"""
from .embedding import (
    EmbeddingClient,
    EmbeddingError,
    EmbeddingResponse,
    MockEmbeddingClient,
    get_embedding_client,
)
from .pg_store import PgVectorStore, SearchResult
from .similarity import cosine_similarity, normalize
from .vector_store import VectorRecord, VectorStore

__all__ = [
    "EmbeddingClient", "EmbeddingError", "EmbeddingResponse",
    "MockEmbeddingClient", "get_embedding_client",
    "PgVectorStore", "SearchResult",
    "cosine_similarity", "normalize",
    "VectorRecord", "VectorStore",
]
