"""实验2：合同信息 JSON 提取工具 —— 企业 FDE 真实场景。

输入：合同文本（txt / md 文件，或直接粘贴）
输出：符合 Pydantic Schema 的结构化 JSON，含：
  - 合同编号、甲乙方、金额、签订/生效/到期日期
  - 付款条款、关键义务、潜在风险点

工程要点：
1. 用 Pydantic 定义强类型 schema（金额必须非负、日期必须 ISO 格式）
2. 用 StructuredClient 自动注入 JSON Schema + 校验 + 失败重试
3. 输出同时给「人类看的摘要」和「机器看的 JSON」
4. 失败要给出明确错误，不能默默吞掉

运行：
    python -m apps.contract_extractor data/contracts/contract_01.txt
    python -m apps.contract_extractor data/contracts/  # 批量处理整个目录
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from llm import get_client
from llm.structured import StructuredClient, ExtractionError


# ---------- Schema ----------
class ContractExtraction(BaseModel):
    """合同信息提取结果。字段设计对应企业 CRM/ERP 落地需要。"""

    contract_no: str = Field(description="合同编号")
    party_a: str = Field(description="甲方（买方/委托方）全称")
    party_b: str = Field(description="乙方（卖方/服务方）全称")
    amount_cny: float = Field(ge=0, description="合同总金额，单位为人民币元，数字")
    sign_date: date = Field(description="签订日期，ISO 格式 YYYY-MM-DD")
    effective_date: date | None = Field(default=None, description="生效日期")
    expiry_date: date | None = Field(default=None, description="到期日期")
    payment_terms: str = Field(description="付款条款原文或简要概括")
    key_obligations: list[str] = Field(default_factory=list, description="3-5 条关键义务")
    risks: list[str] = Field(default_factory=list, description="1-3 条潜在风险点")

    @field_validator("sign_date", "effective_date", "expiry_date", mode="before")
    @classmethod
    def parse_date(cls, v):
        if v is None or isinstance(v, date):
            return v
        if isinstance(v, str):
            return date.fromisoformat(v.strip())
        raise ValueError(f"非法日期: {v}")


# ---------- 提取器 ----------
SYSTEM_PROMPT = (
    "你是一位资深法务助理，擅长从中文合同中提取关键信息。\n"
    "要求：\n"
    "1) 严格按 JSON Schema 输出，不要任何额外解释；\n"
    "2) 金额统一为人民币数字（不带逗号）；\n"
    "3) 日期统一 YYYY-MM-DD；\n"
    "4) 合同中未提到的字段填 null 或空列表，不要编造。"
)


def extract_from_text(text: str) -> ContractExtraction:
    llm = get_client()
    client = StructuredClient(llm, max_retries=2)
    return client.extract(
        schema=ContractExtraction,
        system=SYSTEM_PROMPT,
        user=f"请从以下合同文本中提取信息：\n\n{text}",
    )


def extract_from_file(path: Path) -> ContractExtraction:
    text = path.read_text(encoding="utf-8")
    return extract_from_text(text)


def print_summary(c: ContractExtraction, source: str = "") -> None:
    """人类可读的摘要输出。"""
    print(f"\n{'=' * 60}")
    if source:
        print(f"来源: {source}")
    print(f"合同编号: {c.contract_no}")
    print(f"甲方:     {c.party_a}")
    print(f"乙方:     {c.party_b}")
    print(f"金额:     ¥{c.amount_cny:,.2f}")
    print(f"签订:     {c.sign_date}")
    print(f"生效:     {c.effective_date or '(未填写)'}")
    print(f"到期:     {c.expiry_date or '(未填写)'}")
    print(f"付款条款: {c.payment_terms}")
    if c.key_obligations:
        print("关键义务:")
        for o in c.key_obligations:
            print(f"  - {o}")
    if c.risks:
        print("风险点:")
        for r in c.risks:
            print(f"  ⚠️  {r}")


# ---------- 主流程 ----------
def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("用法: python -m apps.contract_extractor <合同文件或目录>")
        return 1

    target = Path(argv[1])
    files = sorted(target.glob("*.txt")) if target.is_dir() else [target]
    if not files:
        print(f"未找到 .txt 文件: {target}")
        return 1

    results: list[dict] = []
    for f in files:
        try:
            contract = extract_from_file(f)
            print_summary(contract, source=str(f))
            results.append({"source": str(f), "ok": True, "data": json.loads(contract.model_dump_json())})
        except ExtractionError as e:
            print(f"❌ {f.name}: 提取失败 - {e}", file=sys.stderr)
            results.append({"source": str(f), "ok": False, "error": str(e)})

    # 把所有结果写到 results.json，便于后续程序消费
    out = Path("data") / "extraction_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n✅ 结构化结果已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
