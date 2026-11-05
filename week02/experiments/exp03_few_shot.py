"""实验0-C：Zero-shot vs Few-shot —— 用「示例」教模型输出格式。

运行：python -m experiments.exp03_few_shot
"""
from __future__ import annotations

from llm import get_client
from llm.client import ChatMessage


SYSTEM = "你是一个情感分类器，只能输出 positive / negative / neutral 之一。"

ZERO_SHOT = [ChatMessage(role="system", content=SYSTEM)]

FEW_SHOT = [
    ChatMessage(role="system", content=SYSTEM),
    # 三个示例，覆盖三类
    ChatMessage(role="user", content="这家餐厅的菜太难吃了，再也不会来"),
    ChatMessage(role="assistant", content="negative"),
    ChatMessage(role="user", content="服务很贴心，环境也不错，推荐"),
    ChatMessage(role="assistant", content="positive"),
    ChatMessage(role="user", content="今天星期二"),
    ChatMessage(role="assistant", content="neutral"),
]

TEST_INPUTS = [
    "物流太慢了，等了一个星期",
    "客服回复很及时，问题解决了",
    "这本书有 300 页",
]


def run(mode: str, history: list[ChatMessage], client) -> None:
    print(f"\n=== {mode} ===")
    for text in TEST_INPUTS:
        msgs = history + [ChatMessage(role="user", content=text)]
        resp = client.chat(msgs)
        print(f"  输入: {text!r:30s} → {resp.content.strip()!r}")


if __name__ == "__main__":
    client = get_client()
    print("Few-shot 的价值：把「输出格式」用示例喂给模型，比纯文字描述更稳定")
    run("Zero-shot", ZERO_SHOT, client)
    run("Few-shot (3 examples)", FEW_SHOT, client)
    print("\n注意：MockLLM 对'情感'关键词总是回 positive，")
    print("      接真实 API 时可以明显看到 Few-shot 让输出更稳定。")
