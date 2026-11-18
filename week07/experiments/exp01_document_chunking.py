"""实验7-A：解析多制度 Markdown，观察 Chunk 边界、overlap 与元数据。"""
from __future__ import annotations

import json
from pathlib import Path

from llm.chunking import chunk_document, chunk_documents
from llm.document import SourceDocument, load_documents

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_documents(ROOT / "data" / "docs" / "policies.md")
    chunks = chunk_documents(documents, chunk_size=80, overlap=15)
    print(f"Markdown 解析：{len(documents)} 份制度；chunk_size=80 → {len(chunks)} 个 Chunk")
    for chunk in chunks[:5]:
        print(f"{chunk.id} [{chunk.start}:{chunk.end}] {chunk.title}：{chunk.text}")

    long_doc = SourceDocument(
        id="LONG", title="长文档切块演示", category="实验",
        text="第一条规则说明费用申请需要提前提交。" * 12 + "第二条规则说明审批后需要归档。" * 12,
    )
    long_chunks = chunk_document(long_doc, chunk_size=60, overlap=12)
    print(f"\n长文档：{len(long_doc.text)} 字符 → {len(long_chunks)} 块")
    for chunk in long_chunks:
        print(f"{chunk.id} len={len(chunk.text)} range=[{chunk.start}:{chunk.end}]")

    output = ROOT / "data" / "chunking_demo.json"
    output.write_text(json.dumps({
        "source_documents": len(documents), "policy_chunks": len(chunks),
        "long_document": {"characters": len(long_doc.text), "chunks": [
            {"id": c.id, "start": c.start, "end": c.end, "text": c.text} for c in long_chunks
        ]},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {output}")


if __name__ == "__main__":
    main()
