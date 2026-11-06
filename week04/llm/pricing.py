"""模型价格表 + 成本核算 + 成本追踪。

⚠️ 价格为**教学示例价**（USD / 1M tokens），不代表任何厂商的实时报价。
真实项目请把价格表放到配置文件中并定期更新，本模块的代码逻辑不变。

成本公式：cost = prompt_tokens / 1e6 * input_per_m + completion_tokens / 1e6 * output_per_m
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import ChatResponse


@dataclass(frozen=True)
class ModelPrice:
    name: str
    input_per_m: float        # USD / 1M input tokens
    output_per_m: float       # USD / 1M output tokens
    context_window: int       # 上下文窗口（tokens）
    supports_vision: bool = False
    supports_tools: bool = True
    tier: str = "standard"    # economy / standard / flagship
    avg_latency_ms: int = 800  # 经验延迟（教学示例，用于成本-速度权衡演示）


# 教学示例价格表：覆盖「便宜文本模型 → 旗舰多模态模型」的完整光谱
MODEL_PRICES: dict[str, ModelPrice] = {
    "deepseek-chat": ModelPrice(
        "deepseek-chat", 0.27, 1.10, 65536, tier="economy", avg_latency_ms=900,
    ),
    "deepseek-reasoner": ModelPrice(
        "deepseek-reasoner", 0.55, 2.19, 65536, tier="standard", avg_latency_ms=2500,
    ),
    "qwen-plus": ModelPrice(
        "qwen-plus", 0.40, 1.20, 131072, tier="standard", avg_latency_ms=800,
    ),
    "qwen-vl-plus": ModelPrice(
        "qwen-vl-plus", 0.40, 1.20, 131072, supports_vision=True, tier="standard",
        avg_latency_ms=1500,
    ),
    "gpt-4o-mini": ModelPrice(
        "gpt-4o-mini", 0.15, 0.60, 128000, supports_vision=True, tier="economy",
        avg_latency_ms=700,
    ),
    "gpt-4o": ModelPrice(
        "gpt-4o", 2.50, 10.00, 128000, supports_vision=True, tier="flagship",
        avg_latency_ms=1200,
    ),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """单次调用成本（USD）。未知模型按 0 计并交给调用方决定如何处理。"""
    price = MODEL_PRICES.get(model)
    if price is None:
        return 0.0
    return (
        prompt_tokens / 1_000_000 * price.input_per_m
        + completion_tokens / 1_000_000 * price.output_per_m
    )


@dataclass
class UsageRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    latency_ms: float = 0.0
    task: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CostTracker:
    """累计多次调用的 token 与成本，生成报告。所有 LLM 调用都应经过它。"""
    records: list[UsageRecord] = field(default_factory=list)

    def add(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        latency_ms: float = 0.0,
        task: str = "",
    ) -> UsageRecord:
        record = UsageRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=estimate_cost(model, prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            task=task,
        )
        self.records.append(record)
        return record

    def add_response(self, resp: "ChatResponse", *, task: str = "") -> UsageRecord:
        """直接从 ChatResponse 记账（model 取响应里的实际模型名）。"""
        return self.add(
            resp.model, resp.prompt_tokens, resp.completion_tokens,
            latency_ms=resp.latency_ms, task=task,
        )

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.records)

    def by_model(self) -> dict[str, dict]:
        agg: dict[str, dict] = {}
        for r in self.records:
            slot = agg.setdefault(r.model, {"calls": 0, "tokens": 0, "cost": 0.0})
            slot["calls"] += 1
            slot["tokens"] += r.total_tokens
            slot["cost"] += r.cost
        return agg

    def report(self) -> str:
        lines = [
            f"{'模型':<20} {'调用':>4} {'tokens':>10} {'成本(USD)':>12}",
            "-" * 52,
        ]
        for model, slot in sorted(self.by_model().items()):
            lines.append(
                f"{model:<20} {slot['calls']:>4} {slot['tokens']:>10} {slot['cost']:>12.6f}"
            )
        lines.append("-" * 52)
        lines.append(f"合计：{len(self.records)} 次调用，{self.total_tokens} tokens，${self.total_cost:.6f}")
        return "\n".join(lines)
