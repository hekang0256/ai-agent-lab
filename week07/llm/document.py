"""文档解析：JSON / Markdown / TXT / Word / PDF → 统一 SourceDocument。

Markdown/Word 的二级标题可写成 ``文档ID | 类目 | 标题``，这样一个文件也能
保留多份制度的边界与元数据。PDF 按页生成文档，保留 page 元数据。
扫描版 PDF 没有文本层，本周明确报错，OCR 留到第 9 周。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {".json", ".md", ".markdown", ".txt", ".docx", ".pdf"}


@dataclass
class SourceDocument:
    id: str
    title: str
    text: str
    category: str = "未分类"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = self.id.strip()
        self.title = self.title.strip()
        self.text = normalize_text(self.text)
        if not self.id:
            raise ValueError("文档 id 不能为空")
        if not self.title:
            raise ValueError(f"文档 {self.id} 的 title 不能为空")
        if not self.text:
            raise ValueError(f"文档 {self.id} 没有可索引文本")


def normalize_text(text: str) -> str:
    """统一换行和空白，但保留段落边界。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines = [line.strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _from_heading(heading: str, body: str, source: Path, fallback_id: str) -> SourceDocument:
    parts = [p.strip() for p in heading.split("|")]
    if len(parts) == 3:
        doc_id, category, title = parts
    else:
        doc_id, category, title = fallback_id, "未分类", heading.strip()
    return SourceDocument(
        id=doc_id or fallback_id,
        title=title or (doc_id or fallback_id),
        category=category or "未分类",
        text=body,
        source=str(source),
    )


def _parse_markdown_text(text: str, source: Path) -> list[SourceDocument]:
    """有 ## 标题则按标题拆分，否则整个文件是一份文档。"""
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", text))
    if not matches:
        return [SourceDocument(source.stem, source.stem, text, source=str(source))]
    documents = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        if body:
            documents.append(_from_heading(match.group(1), body, source, f"{source.stem}-{i + 1}"))
    if not documents:
        raise ValueError(f"Markdown 没有正文: {source}")
    return documents


def _parse_json(path: Path) -> list[SourceDocument]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("JSON 语料必须是文档对象数组")
    documents = []
    for item in payload:
        documents.append(SourceDocument(
            id=item["id"], title=item["title"], text=item["text"],
            category=item.get("category", "未分类"), source=str(path),
            metadata=item.get("metadata", {}),
        ))
    return documents


def _parse_docx(path: Path) -> list[SourceDocument]:
    from docx import Document

    word_doc = Document(path)
    lines = []
    for paragraph in word_doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # 既识别 Word 的 Heading 2，也接受纯文本形式的 ## 标题。
        is_heading_2 = paragraph.style is not None and paragraph.style.name in {"Heading 2", "标题 2"}
        lines.append(f"## {text}" if is_heading_2 and not text.startswith("## ") else text)
    return _parse_markdown_text("\n\n".join(lines), path)


def _parse_pdf(path: Path) -> list[SourceDocument]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    documents = []
    for page_number, page in enumerate(reader.pages, 1):
        text = normalize_text(page.extract_text() or "")
        if text:
            documents.append(SourceDocument(
                id=f"{path.stem}-p{page_number}", title=f"{path.stem}（第{page_number}页）",
                text=text, source=str(path), metadata={"page": page_number},
            ))
    if not documents:
        raise ValueError(f"PDF 未提取到文本（可能是扫描件，需要 OCR）: {path}")
    return documents


def parse_document(path: str | Path) -> list[SourceDocument]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"文档不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _parse_json(path)
    if suffix in {".md", ".markdown"}:
        return _parse_markdown_text(path.read_text(encoding="utf-8-sig"), path)
    if suffix == ".txt":
        return [SourceDocument(path.stem, path.stem, path.read_text(encoding="utf-8-sig"), source=str(path))]
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ValueError(f"不支持的文档格式 {suffix}；支持 {sorted(SUPPORTED_SUFFIXES)}")


def load_documents(path: str | Path) -> list[SourceDocument]:
    """加载一个文件，或递归加载目录中的全部支持格式。"""
    path = Path(path)
    if path.is_file():
        return parse_document(path)
    if not path.is_dir():
        raise FileNotFoundError(f"语料路径不存在: {path}")
    documents = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES):
        documents.extend(parse_document(file_path))
    if not documents:
        raise ValueError(f"目录中没有可解析文档: {path}")
    return documents
