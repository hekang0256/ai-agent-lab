"""实验4-A：多模态消息结构（图像 + 音频）。

要点：多模态不是新 API，而是 messages 里 content 从 string 变成 content parts 数组。
本实验打印每种 part 的结构（隐去过长的 base64），并用 Mock 跑通发票提取。
"""
from __future__ import annotations

from pathlib import Path

from llm.client import ChatMessage, audio_part, get_client, image_part, text_part
from llm.structured import StructuredClient

from apps.invoice_ocr import SYSTEM, Invoice

ROOT = Path(__file__).resolve().parents[1]


def show_message(msg: ChatMessage) -> None:
    """打印 content parts 结构，base64 只显示长度。"""
    print(f"role={msg.role}")
    if isinstance(msg.content, str):
        print(f"  text: {msg.content[:60]}")
        return
    for part in msg.content:
        ptype = part.get("type")
        if ptype == "text":
            print(f"  part[type=text]: {part['text'][:50]}")
        elif ptype == "image_url":
            url = part["image_url"]["url"]
            print(f"  part[type=image_url]: {url[:40]}... (长度 {len(url)} 字符)")
        elif ptype == "input_audio":
            data = part["input_audio"]
            print(f"  part[type=input_audio]: format={data['format']} base64={len(data['data'])} 字符")


def main() -> None:
    # 1. 本地图片 → data URI
    img_path = ROOT / "data" / "images" / "invoice_01.png"
    msg = ChatMessage(
        role="user",
        content=[text_part(f"请提取这张发票。文件名：{img_path.name}"), image_part(img_path)],
    )
    print("== 1. 本地图片消息 ==")
    show_message(msg)

    # 2. URL 图片：不下载、不编码，直接引用
    print("\n== 2. URL 图片消息 ==")
    url_msg = ChatMessage(
        role="user",
        content=[text_part("这张图里有什么？"), image_part("https://example.com/demo.png")],
    )
    show_message(url_msg)

    # 3. 音频 part（字段格式以具体厂商文档为准）
    print("\n== 3. 音频消息 ==")
    audio_msg = ChatMessage(
        role="user",
        content=[text_part("请转写这段语音"), audio_part(ROOT / "data" / "audio" / "note.wav")],
    )
    show_message(audio_msg)

    # 4. Mock 跑通完整提取流程
    print("\n== 4. 发票提取（Mock 流程） ==")
    client = get_client()
    invoice = StructuredClient(client).extract(
        schema=Invoice, system=SYSTEM,
        user=f"请提取这张发票的关键信息。图片文件名：{img_path.name}",
        images=[str(img_path)],
    )
    print(f"发票号={invoice.invoice_no} 销售方={invoice.seller} "
          f"税额={invoice.tax} 合计={invoice.total} {invoice.currency}")


if __name__ == "__main__":
    main()
