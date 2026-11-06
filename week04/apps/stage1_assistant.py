"""第1阶段综合验收：企业 AI 助手 CLI。

集成 week01~week04 的能力：多轮对话 + 长期记忆 + 上下文管理 + 工具调用 + 成本统计。

用法：
  python -m apps.stage1_assistant --demo            # 离线演示（Mock）
  python -m apps.stage1_assistant                   # 交互模式
  python -m apps.stage1_assistant --strategy sliding

交互命令：/reset 清空对话  /cost 成本报告  /memory 查看长期记忆  /quit 退出
"""
from __future__ import annotations

import argparse

from pathlib import Path

from llm import ChatSession, ContextManager, MemoryStore, get_client
from llm.tools import ToolRegistry, tool

ROOT = Path(__file__).resolve().parents[1]

_FAKE_CUSTOMERS = {
    1001: {"id": 1001, "name": "星辰电商集团", "level": "VIP", "last_order": "SO-2026-0312"},
    1002: {"id": 1002, "name": "甲方示例科技有限公司", "level": "普通", "last_order": "SO-2026-0288"},
}


@tool(description="查询客户信息")
def get_customer(customer_id: int) -> dict:
    """按客户 ID 查询客户等级与最近订单。"""
    return _FAKE_CUSTOMERS.get(customer_id, {"error": f"客户 {customer_id} 不存在"})


@tool(description="按关键词查询合同")
def query_contract(keyword: str) -> dict:
    """返回模拟合同库中的合同要点。"""
    return {
        "contract_no": "HT-2026-001",
        "keyword": keyword,
        "payment_terms": "签约后30日内支付50%，验收后支付50%",
        "risks": ["违约金条款偏严", "未明确知识产权归属"],
    }


def build_session(strategy: str = "summary") -> ChatSession:
    registry = ToolRegistry()
    registry.register(get_customer)
    registry.register(query_contract)
    memory = MemoryStore.load(ROOT / "data" / "memory_store.json")
    context = ContextManager(
        strategy=strategy, max_tokens=2048, reserve_output=512, keep_recent=4,
    )
    return ChatSession(
        get_client(), memory=memory, context=context, registry=registry,
    )


DEMO_SCRIPT = [
    "记住我叫何康",
    "记住我喜欢简洁的回答",
    "我叫什么？",
    "查一下客户 1001 的信息",
    "合同 HT-2026-001 的付款条件是什么？",
]


def show_stats(session: ChatSession) -> None:
    stats = session.stats()
    print("\n----- 会话统计 -----")
    print(f"轮数={stats['turns']} 历史消息={stats['history_messages']} "
          f"tokens={stats['total_tokens']} 成本=${stats['total_cost']:.6f}")
    ctx = stats["last_context"]
    if ctx:
        print(f"上下文策略={ctx.strategy} {ctx.before_tokens}→{ctx.after_tokens} tokens "
              f"(丢弃 {ctx.dropped_messages}，摘要 {ctx.summarized_messages})")
    print(session.tracker.report())


def main() -> None:
    parser = argparse.ArgumentParser(description="第1阶段综合验收：企业 AI 助手")
    parser.add_argument("--strategy", default="summary", choices=["summary", "sliding", "none"])
    parser.add_argument("--demo", action="store_true", help="运行脚本化演示（非交互）")
    args = parser.parse_args()

    session = build_session(args.strategy)

    if args.demo:
        for q in DEMO_SCRIPT:
            print(f"\n用户：{q}")
            print(f"助手：{session.ask(q)}")
        show_stats(session)
        return

    print("企业 AI 助手（输入 /quit 退出，/reset 清空对话，/cost 成本，/memory 记忆）")
    while True:
        try:
            text = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            session.reset()
            print("已清空对话历史（长期记忆保留）。")
            continue
        if text == "/cost":
            show_stats(session)
            continue
        if text == "/memory":
            print(session.memory.facts(user_id=session.user_id) if session.memory else "未启用记忆")
            continue
        print(f"助手：{session.ask(text)}")


if __name__ == "__main__":
    main()
