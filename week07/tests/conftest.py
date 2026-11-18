import os
import sys
from pathlib import Path

import pytest

os.environ["LLM_USE_MOCK"] = "true"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.document import load_documents
from llm.embedding import MockEmbeddingClient
from llm.mock import MockLLMClient
from llm.rag import RAGService
from llm.vector_store import MemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def documents():
    return load_documents(ROOT / "data" / "docs" / "policies.md")


@pytest.fixture()
def llm_client():
    return MockLLMClient()


@pytest.fixture()
def service(documents, llm_client):
    rag = RAGService(
        MemoryVectorStore(), embedding_client=MockEmbeddingClient(),
        llm_client=llm_client, chunk_size=180, overlap=30,
    )
    rag.index_documents(documents)
    return rag
