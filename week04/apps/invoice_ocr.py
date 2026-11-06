r"""多模态应用：发票图片 → 结构化 JSON（Pydantic 强约束）。

用法：
  python -m apps.invoice_ocr                      # 批量处理 data/images/
  python -m apps.invoice_ocr data\images\invoice_01.png

真实 API 时把 .env 里的模型换成支持视觉的型号（如 qwen-vl-plus / gpt-4o-mini），
Mock 模式（LLM_USE_MOCK=true）离线返回与样例图片一致的数据。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from llm.client import get_client
from llm.structured import StructuredClient

ROOT = Path(__file__).resolve().parents[1]


class InvoiceItem(BaseModel):
    name: str
    quantity: int = 1
    amount: float = 0.0


class Invoice(BaseModel):
    invoice_no: str
    date: str                       # YYYY-MM-DD
    seller: str
    buyer: str
    items: list[InvoiceItem] = Field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    currency: str = "CNY"


SYSTEM = (
    "InvoiceExtraction：你是发票信息提取器。从发票图片中准确提取字段，"
    "金额为数字（不含货币符号），日期统一为 YYYY-MM-DD。"
)


def extract_invoice(client, image_path: str | Path) -> Invoice:
    sc = StructuredClient(client)
    return sc.extract(
        schema=Invoice,
        system=SYSTEM,
        user=f"请提取这张发票的关键信息。图片文件名：{Path(image_path).name}",
        images=[str(image_path)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="发票图片信息提取")
    parser.add_argument(
        "path", nargs="?", default=str(ROOT / "data" / "images"),
        help="图片文件或目录（默认 data/images/）",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.glob("*.png")) + sorted(p.glob("*.jpg")) + sorted(p.glob("*.jpeg"))
    else:
        files = [p]
    if not files:
        raise SystemExit(f"没有找到图片：{p}（可先运行 python scripts/make_samples.py）")

    client = get_client()
    results: dict[str, dict] = {}
    for f in files:
        invoice = extract_invoice(client, f)
        results[f.name] = invoice.model_dump()
        print(f"\n== {f.name} ==")
        print(invoice.model_dump_json(indent=2))

    out = ROOT / "data" / "invoice_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n共处理 {len(files)} 张，结果已写入 {out}")


if __name__ == "__main__":
    main()
