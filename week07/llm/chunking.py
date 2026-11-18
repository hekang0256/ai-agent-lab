"""基础 Chunking：固定字符预算 + 尽量贴近句边界 + 可验证的 overlap。

不提前使用语义切块、Parent-Child Retrieval 等第 8 周内容。
start/end 是原文中的字符位置，chunk_id 在同文档同参数下保持稳定。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .document import SourceDocument

BOUNDARY_CHARS = "\n。！？!?；;"


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    chunk_index: int
    start: int
    end: int
    title: str
    category: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        return f"{self.title}\n{self.text}"


def _choose_end(text: str, start: int, chunk_size: int) -> int:
    target = min(start + chunk_size, len(text))
    if target == len(text):
        return target
    # 太靠前的句边界会产生很短的块，因此只接受后半段的边界。
    minimum = start + chunk_size // 2
    candidates = [text.rfind(mark, minimum, target + 1) for mark in BOUNDARY_CHARS]
    boundary = max(candidates)
    return boundary + 1 if boundary >= minimum else target


def chunk_document(document: SourceDocument, *, chunk_size: int = 180, overlap: int = 30) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须满足 0 <= overlap < chunk_size")
    text = document.text
    chunks = []
    start = 0
    while start < len(text):
        end = _choose_end(text, start, chunk_size)
        content = text[start:end].strip()
        if content:
            index = len(chunks)
            chunks.append(Chunk(
                id=f"{document.id}:chunk:{index:03d}", document_id=document.id,
                text=content, chunk_index=index, start=start, end=end,
                title=document.title, category=document.category, source=document.source,
                metadata={**document.metadata, "start": start, "end": end},
            ))
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        if next_start <= start:  # 防御性约束，避免异常输入导致死循环
            next_start = end
        start = next_start
    return chunks


def chunk_documents(documents: list[SourceDocument], *, chunk_size: int = 180, overlap: int = 30) -> list[Chunk]:
    return [chunk for doc in documents for chunk in chunk_document(doc, chunk_size=chunk_size, overlap=overlap)]
