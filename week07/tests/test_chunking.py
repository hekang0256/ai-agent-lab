import pytest

from llm.chunking import chunk_document
from llm.document import SourceDocument


def long_document():
    return SourceDocument(
        id="LONG", title="长文档", category="测试", source="long.md",
        text="第一条规则需要提前申请。" * 10 + "第二条规则需要保留记录。" * 10,
        metadata={"page": 2},
    )


def test_chunk_has_stable_id_position_and_metadata():
    chunks = chunk_document(long_document(), chunk_size=60, overlap=12)
    assert len(chunks) > 2
    assert chunks[0].id == "LONG:chunk:000"
    assert chunks[0].start == 0
    assert chunks[-1].end == len(long_document().text)
    assert all(chunk.title == "长文档" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert chunks[1].start == chunks[0].end - 12


def test_chunk_is_deterministic_and_overlap_is_real():
    first = chunk_document(long_document(), chunk_size=55, overlap=10)
    second = chunk_document(long_document(), chunk_size=55, overlap=10)
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]
    assert first[0].text[-5:] in first[1].text


def test_short_document_is_one_chunk():
    doc = SourceDocument("S", "短文档", "只有一句话。")
    chunks = chunk_document(doc, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0].text == doc.text


@pytest.mark.parametrize("size,overlap", [(0, 0), (20, -1), (20, 20), (20, 30)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_document(SourceDocument("S", "短文档", "正文"), chunk_size=size, overlap=overlap)
