"""实验4：模型对比实验 —— 准确性 / 速度 / 成本 / 结构化稳定性。

对同一个任务，用不同变量各跑 N 次，量化对比：
- 变量 A：模型（deepseek-chat / kimi / qwen / ...）
- 变量 B：温度（0.0 / 0.7 / 1.2）
- 变量 C：Prompt 风格（简短 vs 五要素）

输出：一张 Markdown 对比表 + JSON 原始数据。

注意：本脚本的"模型"通过 LLM_MODEL 环境变量或 --model 参数切换，
      Mock 模式下也能跑（用来验证对比框架本身），但数字没有业务意义。

运行：
    python -m apps.model_compare                       # 用 Mock 跑框架演示
    python -m apps.model_compare --real --runs 5      # 真实 API，每组合跑 5 次
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from llm import get_client
from llm.client import ChatMessage
from llm.structured import extract_json, ExtractionError


# ---------- 测试任务：抽一个「结构化提取」任务作为基准 ----------
TASK_PROMPT = "从下面文本中提取姓名、年龄、城市，输出 JSON：\n\n李四，35 岁，在北京做产品经理。"
EXPECTED_KEYS = {"name", "age", "city"}


@dataclass
class RunResult:
    variant: str
    run_idx: int
    ok: bool
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    output: str
    error: str = ""


def build_messages(style: str) -> list[ChatMessage]:
    if style == "short":
        return [ChatMessage(role="user", content=TASK_PROMPT)]
    # full 风格：五要素齐全
    return [
        ChatMessage(
            role="system",
            content=(
                "# 角色\n你是信息提取专家。\n"
                "# 约束\n1) 只输出 JSON；2) 不要任何解释。\n"
                "# 输出格式\n{\"name\": str, \"age\": int, \"city\": str}"
            ),
        ),
        ChatMessage(role="user", content=TASK_PROMPT),
    ]


def run_one(variant: str, run_idx: int, *, model: str | None, temperature: float, style: str) -> RunResult:
    client = get_client()
    messages = build_messages(style)
    started = time.perf_counter()
    try:
        resp = client.chat(messages, model=model, temperature=temperature,
                           response_format={"type": "json_object"})
        latency = (time.perf_counter() - started) * 1000
        # 校验输出
        try:
            data = extract_json(resp.content)
            if isinstance(data, dict) and EXPECTED_KEYS.issubset(data.keys()):
                ok, err = True, ""
            else:
                ok, err = False, f"缺字段: {EXPECTED_KEYS - set(data.keys())}"
        except ExtractionError as e:
            ok, err = False, str(e)
        return RunResult(
            variant=variant, run_idx=run_idx, ok=ok, latency_ms=latency,
            prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
            output=resp.content, error=err,
        )
    except Exception as e:
        return RunResult(
            variant=variant, run_idx=run_idx, ok=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            prompt_tokens=0, completion_tokens=0,
            output="", error=f"{type(e).__name__}: {e}",
        )


def summarize(results: list[RunResult]) -> dict:
    ok_count = sum(r.ok for r in results)
    latencies = [r.latency_ms for r in results]
    return {
        "runs": len(results),
        "success_rate": ok_count / len(results) if results else 0,
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0,
        "avg_total_tokens": statistics.mean(r.prompt_tokens + r.completion_tokens for r in results) if results else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3, help="每个变体跑几次")
    parser.add_argument("--real", action="store_true", help="用真实 API（默认 Mock）")
    parser.add_argument("--models", default="default", help="逗号分隔的模型列表")
    args = parser.parse_args()

    if not args.real:
        import os
        os.environ["LLM_USE_MOCK"] = "true"

    models = args.models.split(",")
    variants = []
    for m in models:
        for temp in (0.0, 0.7):
            for style in ("short", "full"):
                variants.append({
                    "model": None if m == "default" else m,
                    "temperature": temp,
                    "style": style,
                    "name": f"{m}/T{temp}/{style}",
                })

    print(f"共 {len(variants)} 个变体 × {args.runs} 次 = {len(variants) * args.runs} 次调用\n")

    all_results: list[RunResult] = []
    table_rows = []
    for v in variants:
        print(f"运行变体: {v['name']}")
        results = [run_one(v["name"], i, model=v["model"], temperature=v["temperature"], style=v["style"])
                   for i in range(args.runs)]
        all_results.extend(results)
        s = summarize(results)
        table_rows.append((v["name"], s))
        print(f"  成功率={s['success_rate']:.0%}  P50延迟={s['p50_latency_ms']:.0f}ms  平均token={s['avg_total_tokens']:.0f}")

    # 输出 Markdown 表
    md_lines = ["| 变体 | 成功率 | P50延迟(ms) | 平均Token |",
                "|---|---|---|---|"]
    for name, s in table_rows:
        md_lines.append(f"| {name} | {s['success_rate']:.0%} | {s['p50_latency_ms']:.0f} | {s['avg_total_tokens']:.0f} |")
    md = "\n".join(md_lines)
    print("\n" + "=" * 60)
    print("对比表")
    print("=" * 60)
    print(md)

    # 落盘
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "compare_results.json").write_text(
        json.dumps([asdict(r) for r in all_results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "compare_summary.md").write_text(md, encoding="utf-8")
    print(f"\n✅ 结果已保存: data/compare_results.json, data/compare_summary.md")


if __name__ == "__main__":
    main()
