"""实验4-C：长期记忆（写入 → 持久化 → 跨会话回忆 → 遗忘）。

短期记忆（对话历史）随会话结束消失；长期记忆落盘，新会话仍可回忆。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from llm.context import ContextManager
from llm.memory import MemoryStore, parse_memory_command
from llm.mock import MockLLMClient
from llm.session import ChatSession


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "memory.json"

        # 1. 指令解析
        print("== 1. 记忆指令解析 ==")
        for text in ["记住我叫何康", "记住我喜欢简洁的回答", "记住我的城市是上海", "忘掉我的城市"]:
            print(f"  {text!r} → {parse_memory_command(text)}")

        # 2. 写入并持久化
        store = MemoryStore.load(path)
        store.remember("name", "何康")
        store.remember("likes", "简洁的回答")
        store.remember("城市", "上海")
        print(f"\n== 2. 已写入 {path.name} ==\n{path.read_text(encoding='utf-8')}")

        # 3. 模拟新进程：重新 load，记忆仍在
        store2 = MemoryStore.load(path)
        print(f"\n== 3. 跨会话恢复 ==\n{store2.profile_text()}")

        # 4. 接入会话：画像注入 system，模型可以回答「我叫什么」
        session = ChatSession(
            MockLLMClient(), memory=store2,
            context=ContextManager(strategy="sliding"),
        )
        print("\n== 4. 会话中的记忆 ==")
        print(f"你：我叫什么？\n助手：{session.ask('我叫什么？')}")
        print(f"你：忘掉我的名字\n助手：{session.ask('忘掉我的名字')}")
        print(f"你：我叫什么？\n助手：{session.ask('我叫什么？')}")


if __name__ == "__main__":
    main()
