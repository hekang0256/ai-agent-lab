"""实验7-B：拆开 Retrieval 与 Generation，检查真正发给模型的 Context。"""
from __future__ import annotations

import json
from pathlib import Path

from apps.policy_qa import DOCS_PATH, create_service, ensure_index
from llm.tokenizer import estimate_messages_tokens

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    service = create_service("memory")
    ensure_index(service, DOCS_PATH)
    query = "差旅报销需要哪些材料，多久到账？"
    results = service.retrieve(query, top_k=2)
    messages = service.build_messages(query, results)
    print(f"检索到 {len(results)} 块，Prompt 估算 {estimate_messages_tokens(messages)} tokens")
    print("\n== System ==\n" + messages[0].content)
    print("\n== User（问题 + Context）==\n" + messages[1].content)
    output = ROOT / "data" / "context_prompt_demo.json"
    output.write_text(json.dumps({
        "query": query, "estimated_tokens": estimate_messages_tokens(messages),
        "messages": [message.to_dict() for message in messages],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nPrompt 已写入 {output}")


if __name__ == "__main__":
    main()
