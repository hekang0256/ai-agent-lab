"""week05 LLM 工具箱：向量 / 相似度 / Embedding / 向量库。"""
from .embedding import (
    EmbeddingClient,
    EmbeddingError,
    EmbeddingResponse,
    MockEmbeddingClient,
    get_embedding_client,
)
from .similarity import (
    cosine_matrix,
    cosine_similarity,
    dot,
    l2_distance,
    norm,
    normalize,
    top_k,
)
from .vector_store import SearchResult, VectorRecord, VectorStore

__all__ = [
    "EmbeddingClient", "EmbeddingError", "EmbeddingResponse",
    "MockEmbeddingClient", "get_embedding_client",
    "cosine_matrix", "cosine_similarity", "dot", "l2_distance",
    "norm", "normalize", "top_k",
    "SearchResult", "VectorRecord", "VectorStore",
]
