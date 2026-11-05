"""实验3-A：JSON Mode —— 让模型保证输出合法 JSON（但不保证 schema）。

运行：python -m experiments.exp01_json_mode
"""
from __future__ import annotations

import json

from llm import get_client
from llm.client import ChatMessage
from llm.structured import extract_json


def demo() -> None:
    client = get_client()
    messages = [
        ChatMessage(role="system", content="你是数据提取助手，只输出 JSON。"),
        ChatMessage(
            role="user",
            content="从下面文本提取姓名和年龄，输出 JSON：张三今年 28 岁，是个工程师。",
        ),
    ]

    print("=" * 60)
    print("1. 不用 json_object 模式（模型可能包裹 markdown 代码块）")
    print("=" * 60)
    resp = client.chat(messages)
    print(f"原始输出:\n{resp.content}\n")
    try:
        data = extract_json(resp.content)
        print(f"✅ extract_json 成功: {data}")
    except Exception as e:
        print(f"❌ 解析失败: {e}")

    print("\n" + "=" * 60)
    print("2. 用 json_object 模式（API 层保证合法 JSON）")
    print("=" * 60)
    resp = client.chat(messages, response_format={"type": "json_object"})
    print(f"原始输出:\n{resp.content}\n")
    try:
        data = json.loads(resp.content)
        print(f"✅ json.loads 直接成功: {data}")
    except json.JSONDecodeError as e:
        print(f"❌ 仍然失败: {e}")


if __name__ == "__main__":
    demo()
    print("\n关键收获：json_object 模式只保证「合法 JSON」，不保证字段名/类型符合预期。")
    print("          要保证 schema 需要 exp02 的 Pydantic 校验。")
