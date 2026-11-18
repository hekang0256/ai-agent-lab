"""实验7-C：完整 RAG 闭环，观察有证据回答、引用映射与无证据拒答。"""
from __future__ import annotations

from apps.policy_qa import create_service, ensure_index, print_response


def main() -> None:
    service = create_service("memory")
    stats = ensure_index(service)
    print(f"索引：{stats.documents} 份文档 → {stats.chunks} 个 Chunk")
    for query, category in [
        ("出差打车费用怎么报销？", None),
        ("报销发票抬头怎么开？", "财务"),
        ("报销发票抬头怎么开？", "IT"),
        ("今天中午吃什么？", None),
    ]:
        print_response(service.answer(query, category=category))
    print("\n要点：无检索结果时不会调用 LLM；引用来自检索结果映射，而非模型自行提供 URL。")


if __name__ == "__main__":
    main()
