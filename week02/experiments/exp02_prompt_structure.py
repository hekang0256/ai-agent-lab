"""实验0-B：Prompt 五要素对比 —— 角色 / 任务 / 上下文 / 约束 / 输出格式。

运行：python -m experiments.exp02_prompt_structure
"""
from __future__ import annotations

from dataclasses import dataclass

from llm import get_client
from llm.client import ChatMessage


@dataclass
class Prompt:
    role: str
    task: str
    context: str
    constraints: str
    output_format: str

    def to_messages(self) -> list[ChatMessage]:
        system = (
            f"# 角色\n{self.role}\n\n"
            f"# 约束\n{self.constraints}\n\n"
            f"# 输出格式\n{self.output_format}"
        )
        user = f"# 任务\n{self.task}\n\n# 上下文\n{self.context}"
        return [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]


def demo_bad_vs_good_prompt() -> None:
    client = get_client()

    bad = [ChatMessage(role="user", content="写个总结")]

    good = Prompt(
        role="你是一位有10年经验的项目经理，擅长向高管汇报。",
        task="把下面的项目进展总结成一段 100 字以内的汇报。",
        context=(
            "本周完成了 LLM 客户端封装（含超时/重试），跑了 4 个实验，"
            "写了 5 个 pytest，发现外部 API 偶发超时，已通过重试解决。"
        ),
        constraints="1) 不要技术细节；2) 突出进展和风险；3) 语气专业克制。",
        output_format="一段中文，100 字以内，不要使用列表。",
    ).to_messages()

    print("=" * 60)
    print("Prompt 对比：Bad vs Good")
    print("=" * 60)
    for label, msgs in [("Bad Prompt", bad), ("Good Prompt", good)]:
        resp = client.chat(msgs)
        print(f"\n[{label}]")
        print(f"  System: {msgs[0].content if msgs[0].role=='system' else '(无)'}")
        print(f"  回复: {resp.content}")
        print(f"  Token: {resp.cost_hint} | 延迟: {resp.latency_ms:.0f}ms")


if __name__ == "__main__":
    demo_bad_vs_good_prompt()
