"""多模态消息与发票提取流程测试（Mock 离线）。"""
import base64

import pytest
from pydantic import ValidationError

from apps.invoice_ocr import Invoice, extract_invoice
from llm.client import ChatMessage, audio_part, image_part, text_part
from llm.mock import MockLLMClient


def test_image_part_from_local_file(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake-bytes")
    part = image_part(img)
    url = part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # base64 能还原原始字节
    assert base64.b64decode(url.split(",", 1)[1]) == img.read_bytes()


def test_image_part_from_jpg_extension(tmp_path):
    img = tmp_path / "b.jpg"
    img.write_bytes(b"jpg-bytes")
    assert image_part(img)["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_image_part_url_passthrough():
    part = image_part("https://example.com/x.png")
    assert part["image_url"]["url"] == "https://example.com/x.png"


def test_image_part_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        image_part(tmp_path / "not-exists.png")


def test_audio_part_format_inferred(tmp_path):
    audio = tmp_path / "note.mp3"
    audio.write_bytes(b"audio-bytes")
    part = audio_part(audio)
    assert part["type"] == "input_audio"
    assert part["input_audio"]["format"] == "mp3"
    assert base64.b64decode(part["input_audio"]["data"]) == b"audio-bytes"


def test_chat_message_text_extracts_text_parts():
    msg = ChatMessage(
        role="user",
        content=[text_part("第一段"), image_part(b"fake"), text_part("第二段")],
    )
    assert "第一段" in msg.text and "第二段" in msg.text
    # to_dict 保留 parts 结构
    assert isinstance(msg.to_dict()["content"], list)


def test_invoice_extraction_mock(tmp_path):
    img = tmp_path / "invoice_01.png"
    img.write_bytes(b"fake-png")  # Mock 不解码图片内容，只要求文件存在
    invoice = extract_invoice(MockLLMClient(), img)
    assert invoice.invoice_no == "INV-2026-0001"
    assert invoice.total == 13568.0
    assert invoice.tax == 768.0
    assert len(invoice.items) == 2


def test_invoice_extraction_second_sample(tmp_path):
    img = tmp_path / "invoice_02.png"
    img.write_bytes(b"fake-png")
    invoice = extract_invoice(MockLLMClient(), img)
    assert invoice.invoice_no == "INV-2026-0002"
    assert invoice.seller == "NORTHWIND TRADERS"
    assert invoice.total == 10176.0


def test_invoice_schema_rejects_bad_data():
    with pytest.raises(ValidationError):
        Invoice.model_validate({"invoice_no": "X"})  # 缺必填字段


def test_mock_multimodal_message_roundtrip(tmp_path):
    """多模态 user 消息走 Mock 全链路，不应因 content 是 list 而报错。"""
    img = tmp_path / "invoice_01.png"
    img.write_bytes(b"fake")
    client = MockLLMClient()
    msg = ChatMessage(
        role="user",
        content=[text_part("随便聊聊"), image_part(img)],
    )
    resp = client.chat([msg])
    assert "MOCK" in resp.content
    assert resp.prompt_tokens > 1024  # 图片 token 被计入
