"""实验1：AI 文本处理工具箱 —— 摘要 / 翻译 / 润色 / 关键词 / 情感 五合一。

把多个常见文本处理任务收敛到一个 CLI，每个任务都是「Prompt 模板 + LLM 调用」，
体会「同一个模型，不同 prompt 解决不同问题」。

运行：
    python -m experiments.exp04_text_toolkit summarize "长文本..."
    python -m experiments.exp04_text_toolkit translate "你好世界"
    python -m experiments.exp04_text_toolkit polish "这个报告写的不好"
    python -m experiments.exp04_text_toolkit keywords "一段文本"
    python -m experiments.exp04_text_toolkit sentiment "今天很开心"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from llm import get_client
from llm.client import ChatMessage


@dataclass
class TaskSpec:
    name: str
    system: str
    user_template: str
    temperature: float = 0.3


TASKS: dict[str, TaskSpec] = {
    "summarize": TaskSpec(
        name="摘要",
        system="你是专业摘要员。输出不超过 80 字的中文摘要，只输出摘要本身。",
        user_template="请摘要以下文本：\n\n{text}",
    ),
    "translate": TaskSpec(
        name="中→英翻译",
        system="你是专业笔译。把用户输入的中文翻译成地道英文，只输出译文。",
        user_template="{text}",
    ),
    "polish": TaskSpec(
        name="润色",
        system="你是资深中文编辑。润色用户文本，使其更专业、简洁、通顺，保持原意，只输出润色后的文本。",
        user_template="{text}",
    ),
    "keywords": TaskSpec(
        name="关键词提取",
        system="你是 SEO 专家。从用户文本中提取 3-5 个关键词，用英文逗号分隔，不要其他内容。",
        user_template="{text}",
    ),
    "sentiment": TaskSpec(
        name="情感分析",
        system="你是情感分类器。只能输出 positive / negative / neutral 之一。",
        user_template="{text}",
        temperature=0.0,
    ),
}


def run_task(task_key: str, text: str) -> str:
    spec = TASKS[task_key]
    client = get_client()
    resp = client.chat(
        [
            ChatMessage(role="system", content=spec.system),
            ChatMessage(role="user", content=spec.user_template.format(text=text)),
        ],
        temperature=spec.temperature,
    )
    return resp.content.strip()


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in TASKS:
        print("用法: python -m experiments.exp04_text_toolkit <task> <text>")
        print("可用任务:", ", ".join(TASKS))
        return 1
    task, text = argv[1], " ".join(argv[2:])
    result = run_task(task, text)
    print(f"[{TASKS[task].name}] {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
