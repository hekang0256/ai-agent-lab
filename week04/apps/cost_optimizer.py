"""模型选型与成本优化实战：对一批企业任务做路由，量化「按需选模型」省多少钱。

对比口径（同一把尺子）：
- 基线：所有任务都用旗舰模型（默认 gpt-4o）
- 优化：ModelRouter 按任务约束选最便宜的可用模型
两者都按 tasks.json 里的预估 token 用量计算，最后给出节省比例。
另外用 Mock 实际跑一遍每个任务，验证路由结果可执行并把真实 token 记账。

用法：
  python -m apps.cost_optimizer                 # cost 优先
  python -m apps.cost_optimizer --prefer quality
  python -m apps.cost_optimizer --budget 0.5    # 每 1k 次调用预算 0.5 USD
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.client import ChatMessage, get_client
from llm.pricing import CostTracker, estimate_cost
from llm.router import ModelRouter, RouterError, TaskRequirement

ROOT = Path(__file__).resolve().parents[1]


def load_tasks() -> list[dict]:
    return json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="模型路由成本优化仿真")
    parser.add_argument("--prefer", default="cost", choices=["cost", "quality", "latency"])
    parser.add_argument("--baseline", default="gpt-4o", help="基线旗舰模型")
    parser.add_argument("--budget", type=float, default=None, help="每 1k 次调用的预算上限（USD）")
    args = parser.parse_args()

    tasks = load_tasks()
    router = ModelRouter(prefer=args.prefer)
    client = get_client()
    tracker = CostTracker()

    lines = [
        f"# 模型路由成本报告（prefer={args.prefer}，基线={args.baseline}）",
        "",
        "| 任务 | 复杂度 | 选中模型 | 单次成本(USD) | 基线成本(USD) |",
        "|---|---|---|---:|---:|",
    ]
    routed_total, baseline_total = 0.0, 0.0
    reasons: list[str] = []

    for t in tasks:
        req = TaskRequirement(
            needs_vision=t.get("needs_vision", False),
            needs_tools=t.get("needs_tools", False),
            min_context=t.get("min_context", 8000),
            complexity=t.get("complexity", "medium"),
            expected_prompt_tokens=t.get("expected_prompt_tokens", 1000),
            expected_completion_tokens=t.get("expected_completion_tokens", 300),
            max_cost_per_1k=args.budget,
        )
        base_cost = estimate_cost(
            args.baseline, req.expected_prompt_tokens, req.expected_completion_tokens,
        )
        try:
            decision = router.route(req)
        except RouterError as e:
            lines.append(f"| {t['description']} | {req.complexity} | ❌ 无法路由 | - | {base_cost:.6f} |")
            reasons.append(f"- **{t['id']} {t['description']}**：{e}")
            baseline_total += base_cost
            continue

        # 仿真执行一次：验证选中的模型能跑通流程，token 真实记账
        resp = client.chat([ChatMessage(role="user", content=t["input"])], model=decision.model)
        tracker.add_response(resp, task=t["id"])

        routed_total += decision.estimated_cost
        baseline_total += base_cost
        lines.append(
            f"| {t['description']} | {req.complexity} | {decision.model} "
            f"| {decision.estimated_cost:.6f} | {base_cost:.6f} |"
        )
        reasons.append(f"- **{t['id']} {t['description']}** → `{decision.model}`：{decision.reason}")

    saving = (1 - routed_total / baseline_total) * 100 if baseline_total else 0.0
    summary = [
        "",
        "## 结论",
        "",
        f"- 路由方案预估成本：${routed_total:.6f} / 批",
        f"- 旗舰基线预估成本：${baseline_total:.6f} / 批",
        f"- **节省 {saving:.1f}%**（同口径预估）",
        "",
        "## 决策理由",
        "",
        *reasons,
        "",
        "## 仿真执行实际记账（Mock）",
        "",
        "```text",
        tracker.report(),
        "```",
    ]

    report = "\n".join(lines + summary)
    out = ROOT / "data" / "cost_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    main()
