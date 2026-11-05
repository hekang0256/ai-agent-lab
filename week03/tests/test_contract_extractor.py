"""合同提取器端到端测试（用 MockLLM）。"""
import os

os.environ["LLM_USE_MOCK"] = "true"

from datetime import date
from pathlib import Path

import pytest

from apps.contract_extractor import (
    ContractExtraction,
    extract_from_text,
    extract_from_file,
)


def test_schema_validates_correct_data():
    c = ContractExtraction(
        contract_no="T-001",
        party_a="A公司",
        party_b="B公司",
        amount_cny=100.0,
        sign_date="2026-01-01",  # str 自动转 date
        payment_terms="一次性",
    )
    assert c.sign_date == date(2026, 1, 1)
    assert c.effective_date is None
    assert c.key_obligations == []


def test_schema_rejects_negative_amount():
    with pytest.raises(Exception):
        ContractExtraction(
            contract_no="T", party_a="A", party_b="B",
            amount_cny=-100, sign_date="2026-01-01", payment_terms="x",
        )


def test_extract_from_text_with_mock():
    """Mock 对'合同'关键词返回固定 JSON，应该能解析并通过校验。"""
    text = "这是一份技术服务合同，合同编号HT-2026-001……"
    result = extract_from_text(text)
    assert isinstance(result, ContractExtraction)
    assert result.contract_no == "HT-2026-001"
    assert result.amount_cny == 1200000.0
    assert result.sign_date == date(2026, 3, 15)
    assert len(result.key_obligations) >= 1
    assert len(result.risks) >= 1


def test_extract_from_real_contract_file():
    """用 data/contracts/contract_01.txt 真实文件做端到端。"""
    contract_file = Path(__file__).parent.parent / "data" / "contracts" / "contract_01.txt"
    assert contract_file.exists(), f"测试文件不存在: {contract_file}"
    result = extract_from_file(contract_file)
    assert isinstance(result, ContractExtraction)
