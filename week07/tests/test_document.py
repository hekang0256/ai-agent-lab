import json

import pytest

from llm.document import SourceDocument, load_documents, normalize_text, parse_document


def test_parse_markdown_preserves_document_boundaries(documents):
    assert len(documents) == 14
    assert documents[0].id == "P01"
    assert documents[0].category == "财务"
    assert documents[0].title == "差旅报销制度"
    assert "发票" in documents[0].text
    assert len({doc.id for doc in documents}) == 14


def test_parse_plain_text_and_normalize(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("第一行\r\n\r\n\r\n第二行", encoding="utf-8")
    docs = parse_document(path)
    assert len(docs) == 1
    assert docs[0].text == "第一行\n\n第二行"
    assert docs[0].source.endswith("note.txt")


def test_parse_json_documents(tmp_path):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps([{"id": "A", "title": "制度A", "text": "正文", "category": "行政"}]), encoding="utf-8")
    docs = parse_document(path)
    assert docs[0].category == "行政"


def test_parse_word_heading_two(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "handbook.docx"
    word = docx.Document()
    word.add_heading("W01 | 人事 | 测试制度", level=2)
    word.add_paragraph("第一条正文。")
    word.save(path)
    docs = parse_document(path)
    assert len(docs) == 1
    assert docs[0].id == "W01"
    assert docs[0].title == "测试制度"
    assert docs[0].text == "第一条正文。"


def test_parse_pdf_by_page_with_metadata(tmp_path, monkeypatch):
    pypdf = pytest.importorskip("pypdf")

    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Reader:
        def __init__(self, path):
            self.pages = [Page("page one"), Page(""), Page("page three")]

    monkeypatch.setattr(pypdf, "PdfReader", Reader)
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"fake")
    docs = parse_document(path)
    assert [doc.metadata["page"] for doc in docs] == [1, 3]
    assert docs[0].id == "sample-p1"


def test_load_directory_ignores_unsupported_files(tmp_path):
    (tmp_path / "a.txt").write_text("文本", encoding="utf-8")
    (tmp_path / "ignore.exe").write_bytes(b"not a document")
    assert len(load_documents(tmp_path)) == 1


def test_empty_document_and_unknown_suffix_rejected(tmp_path):
    with pytest.raises(ValueError):
        SourceDocument("A", "空文档", "  ")
    path = tmp_path / "a.xyz"
    path.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_document(path)
