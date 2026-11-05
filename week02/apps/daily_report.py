"""实验3：AI 生成结构化项目日报。

输入：零散的进展/问题/计划条目（JSON 或命令行）
输出：格式统一的项目日报（Markdown），含：今日进展 / 风险与阻塞 / 明日计划 / 需要的支持

工程要点：
1. 用 Prompt 五要素（角色/任务/上下文/约束/输出格式）约束输出
2. 用 Few-shot 给一份「样例日报」让模型模仿结构
3. 输出后再用 Python 做后处理校验（必须包含 4 个小节）

运行：
    python -m apps.daily_report                       # 用内置示例数据
    python -m apps.daily_report data/report_input.json
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from llm import get_client
from llm.client import ChatMessage


# ---------- 数据模型 ----------
@dataclass
class ReportInput:
    project: str
    date: str
    author: str
    progress: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)


# ---------- Prompt ----------
SYSTEM_PROMPT = (
    "你是一名经验丰富的项目经理，擅长把零散信息整理成结构清晰的项目日报。\n"
    "约束：\n"
    "1) 严格按给定 Markdown 模板输出，不要加额外解释；\n"
    "2) 每节最多 3 条，每条不超过 30 字；\n"
    "3) 没有内容的章节写「无」。"
)

FEW_SHOT_USER = """项目：CRM 重构
日期：2026-10-10
进展：完成客户列表页开发；联调通过 5 个接口
阻塞：测试环境数据库连接不稳定
计划：开发客户详情页
支持：需要 DBA 协助排查测试库"""

FEW_SHOT_ASSISTANT = """# 项目日报 · CRM 重构 · 2026-10-10

## 一、今日进展
- 完成客户列表页开发
- 联调通过 5 个接口

## 二、风险与阻塞
- 测试环境数据库连接不稳定

## 三、明日计划
- 开发客户详情页

## 四、需要的支持
- 需要 DBA 协助排查测试库"""


def build_messages(data: ReportInput) -> list[ChatMessage]:
    user = (
        f"项目：{data.project}\n"
        f"日期：{data.date}\n"
        f"进展：{'；'.join(data.progress) or '无'}\n"
        f"阻塞：{'；'.join(data.blockers) or '无'}\n"
        f"计划：{'；'.join(data.plans) or '无'}\n"
        f"支持：{'；'.join(data.supports) or '无'}"
    )
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=FEW_SHOT_USER),
        ChatMessage(role="assistant", content=FEW_SHOT_ASSISTANT),
        ChatMessage(role="user", content=user),
    ]


# ---------- 输出校验 ----------
REQUIRED_SECTIONS = ["今日进展", "风险与阻塞", "明日计划", "需要的支持"]


def validate_report(markdown: str) -> list[str]:
    """返回缺失的章节列表，空列表 = 校验通过。"""
    return [s for s in REQUIRED_SECTIONS if s not in markdown]


def polish_markdown(md: str, data: ReportInput) -> str:
    """统一标题，把作者信息补进去。"""
    expected_title = f"# 项目日报 · {data.project} · {data.date}"
    lines = md.strip().splitlines()
    if lines and lines[0].startswith("#"):
        lines[0] = expected_title
    else:
        lines.insert(0, expected_title)
    # 在标题下加作者行
    if len(lines) >= 2 and "作者" not in lines[1]:
        lines.insert(1, f"作者：{data.author}")
    return "\n".join(lines)


# ---------- 主流程 ----------
def generate_report(data: ReportInput) -> str:
    client = get_client()
    resp = client.chat(build_messages(data), temperature=0.3)
    md = resp.content.strip()
    md = polish_markdown(md, data)
    missing = validate_report(md)
    if missing:
        # 不直接失败，打印告警；真实生产可触发重试或人工 review
        print(f"[WARN] 报告缺少章节: {missing}", file=sys.stderr)
    return md


def load_input(path: str | None) -> ReportInput:
    if path is None:
        return ReportInput(
            project="ai-agent-lab",
            date="2026-11-05",
            author="kekang",
            progress=["完成 LLM 客户端封装（含超时/重试）", "跑通 Token/Prompt/Few-shot 三个实验", "写完 5 个 pytest"],
            blockers=["外部 API 偶发超时，已通过指数退避重试解决"],
            plans=["接入 Structured Output", "实现 Tool Calling"],
            supports=["希望评审一下 Prompt 模板是否可复用"],
        )
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReportInput(**raw)


if __name__ == "__main__":
    data = load_input(sys.argv[1] if len(sys.argv) > 1 else None)
    print(generate_report(data))
