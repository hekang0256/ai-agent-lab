"""日报生成器测试（用 MockLLM）。"""
import os

import pytest

# 强制走 Mock，保证 CI 稳定
os.environ["LLM_USE_MOCK"] = "true"

from apps.daily_report import (
    ReportInput,
    build_messages,
    generate_report,
    validate_report,
    polish_markdown,
    REQUIRED_SECTIONS,
)


@pytest.fixture
def sample_input() -> ReportInput:
    return ReportInput(
        project="ai-agent-lab",
        date="2026-11-05",
        author="tester",
        progress=["完成 A", "完成 B"],
        blockers=["外部 API 慢"],
        plans=["做 C"],
        supports=[],
    )


def test_build_messages_has_few_shot(sample_input):
    msgs = build_messages(sample_input)
    # system + few-shot user + few-shot assistant + 真实 user
    assert len(msgs) == 4
    assert msgs[0].role == "system"
    assert msgs[1].role == "user" and "CRM" in msgs[1].content
    assert msgs[2].role == "assistant" and "项目日报" in msgs[2].content
    assert msgs[3].role == "user" and "ai-agent-lab" in msgs[3].content


def test_validate_report_pass():
    good = "# 项目日报\n## 一、今日进展\n- x\n## 二、风险与阻塞\n- x\n## 三、明日计划\n- x\n## 四、需要的支持\n- x"
    assert validate_report(good) == []


def test_validate_report_missing_sections():
    bad = "# 项目日报\n## 一、今日进展\n- x"
    missing = validate_report(bad)
    assert "明日计划" in missing
    assert "需要的支持" in missing
    assert "今日进展" not in missing


def test_polish_markdown_normalizes_title(sample_input):
    md = "# 老标题\n## 一、今日进展\n- x"
    result = polish_markdown(md, sample_input)
    assert result.startswith(f"# 项目日报 · {sample_input.project} · {sample_input.date}")
    assert f"作者：{sample_input.author}" in result


def test_generate_report_end_to_end(sample_input, capsys):
    md = generate_report(sample_input)
    # 标题规范
    assert sample_input.project in md
    assert sample_input.date in md
    assert sample_input.author in md
    # Mock 输出可能缺章节，但 polish 不应抛错
    assert isinstance(md, str) and len(md) > 0
