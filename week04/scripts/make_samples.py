"""生成 week04 多模态测试样本：2 张 PNG 发票 + 1 段 WAV 音频。

只用 Python 标准库（zlib/struct/wave），不依赖 Pillow：
- PNG 编码器：手写 IHDR/IDAT/IEND 三个 chunk
- 文字渲染：内置 5x7 位图字体（大写字母/数字/常用符号），按整数倍放大

运行：python scripts/make_samples.py
"""
from __future__ import annotations

import math
import struct
import wave
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "images"
AUDIO_DIR = ROOT / "data" / "audio"

# ---------- 5x7 位图字体（每字符 7 行，每行低 5 位有效，bit4 是最左列）----------
FONT: dict[str, list[int]] = {
    " ": [0x00] * 7,
    "A": [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "B": [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
    "C": [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
    "D": [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
    "E": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
    "F": [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
    "G": [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
    "H": [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
    "I": [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "J": [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
    "K": [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
    "L": [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
    "M": [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
    "N": [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
    "O": [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "P": [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
    "Q": [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
    "R": [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
    "S": [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
    "T": [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
    "U": [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
    "V": [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
    "W": [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
    "X": [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
    "Y": [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
    "Z": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
    "0": [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
    "1": [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
    "2": [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
    "3": [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
    "4": [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
    "5": [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
    "6": [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
    "7": [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
    "8": [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
    "9": [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
    ":": [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
    ".": [0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x06],
    ",": [0x00, 0x00, 0x00, 0x00, 0x06, 0x04, 0x08],
    "-": [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
    "/": [0x01, 0x02, 0x02, 0x04, 0x08, 0x08, 0x10],
    "#": [0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A],
    "%": [0x19, 0x1A, 0x02, 0x04, 0x08, 0x0B, 0x13],
    "+": [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00],
    "(": [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
    ")": [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
}


# ---------- 像素画布 ----------
def new_canvas(width: int, height: int) -> bytearray:
    return bytearray(b"\xff" * (width * height * 3))  # 白底


def fill_rect(buf: bytearray, w: int, h: int, x: int, y: int, rw: int, rh: int,
              color: tuple[int, int, int] = (20, 20, 20)) -> None:
    for yy in range(max(0, y), min(h, y + rh)):
        for xx in range(max(0, x), min(w, x + rw)):
            idx = (yy * w + xx) * 3
            buf[idx: idx + 3] = bytes(color)


def draw_text(buf: bytearray, w: int, h: int, x: int, y: int, text: str,
              *, scale: int = 3, color: tuple[int, int, int] = (20, 20, 20)) -> None:
    for i, ch in enumerate(text):
        glyph = FONT.get(ch.upper(), FONT[" "])
        for row, bits in enumerate(glyph):
            for col in range(5):
                if bits & (1 << (4 - col)):
                    fill_rect(
                        buf, w, h,
                        x + (i * 6 + col) * scale, y + row * scale,
                        scale, scale, color,
                    )


def write_png(path: Path, width: int, height: int, buf: bytearray) -> None:
    stride = width * 3
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # 每行的 filter type = 0 (None)
        raw += buf[y * stride:(y + 1) * stride]

    def chunk(ctype: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + ctype + data
            + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


# ---------- 发票 ----------
def make_invoice(path: Path, lines: list[str]) -> None:
    w, h = 900, 1000
    buf = new_canvas(w, h)
    draw_text(buf, w, h, 60, 50, "INVOICE", scale=8, color=(10, 40, 120))
    fill_rect(buf, w, h, 60, 140, w - 120, 4, (10, 40, 120))
    y = 200
    for line in lines:
        if line.startswith("TOTAL"):
            fill_rect(buf, w, h, 60, y - 16, w - 120, 2, (120, 120, 120))
            draw_text(buf, w, h, 60, y, line, scale=5)
            y += 64
        else:
            draw_text(buf, w, h, 60, y, line, scale=4)
            y += 48
    write_png(path, w, h, buf)


def make_wav(path: Path) -> None:
    """0.6 秒 440Hz 正弦波（演示音频消息构造用，无真实语音内容）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    rate, duration, freq = 16000, 0.6, 440
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * duration)):
            value = int(12000 * math.sin(2 * math.pi * freq * i / rate))
            frames += struct.pack("<h", value)
        f.writeframes(bytes(frames))


def main() -> None:
    make_invoice(IMG_DIR / "invoice_01.png", [
        "NO: INV-2026-0001",
        "DATE: 2026-04-12",
        "SELLER: ACME SUPPLY CO",
        "BUYER: EXAMPLE TECH LTD",
        "",
        "ITEM: CLOUD SERVER X2",
        "      AMOUNT CNY 10000.00",
        "ITEM: SSD 4TB X5",
        "      AMOUNT CNY 2800.00",
        "",
        "SUBTOTAL: CNY 12800.00",
        "TAX 6%: CNY 768.00",
        "TOTAL: CNY 13568.00",
    ])
    make_invoice(IMG_DIR / "invoice_02.png", [
        "NO: INV-2026-0002",
        "DATE: 2026-05-03",
        "SELLER: NORTHWIND TRADERS",
        "BUYER: EXAMPLE TECH LTD",
        "",
        "ITEM: OFFICE CHAIR X20",
        "      AMOUNT CNY 9600.00",
        "",
        "SUBTOTAL: CNY 9600.00",
        "TAX 6%: CNY 576.00",
        "TOTAL: CNY 10176.00",
    ])
    make_wav(AUDIO_DIR / "note.wav")
    print(f"已生成：{IMG_DIR / 'invoice_01.png'}")
    print(f"已生成：{IMG_DIR / 'invoice_02.png'}")
    print(f"已生成：{AUDIO_DIR / 'note.wav'}")


if __name__ == "__main__":
    main()
