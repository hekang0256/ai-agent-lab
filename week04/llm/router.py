"""模型路由（Model Router）：按任务需求与预算，选「够用且最便宜」的模型。

核心原则：**不是所有任务都值得用旗舰模型**。
- 硬约束过滤：视觉能力 / 工具调用 / 上下文窗口 / 预算，不满足直接出局
- 软偏好排序：cost（默认， cheapest first）/ quality / latency
- 复杂度门槛：high 复杂度任务不允许 economy 模型（质量风险）

每次路由返回 RouteDecision，包含候选列表与理由 —— 决策必须可解释、可审计。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pricing import MODEL_PRICES, ModelPrice, estimate_cost


class RouterError(Exception):
    """没有满足约束的模型。"""


@dataclass(frozen=True)
class TaskRequirement:
    needs_vision: bool = False
    needs_tools: bool = False
    min_context: int = 8000
    complexity: str = "medium"          # low / medium / high
    expected_prompt_tokens: int = 1000  # 用于成本估算
    expected_completion_tokens: int = 300
    max_cost_per_1k: float | None = None  # 预算：每 1000 次调用的成本上限（USD）


@dataclass
class RouteDecision:
    model: str
    reason: str
    estimated_cost: float                       # 单次调用的预估成本（USD）
    candidates: list[tuple[str, float]] = field(default_factory=list)  # (模型, 预估成本) 按序


class ModelRouter:
    def __init__(
        self,
        prices: dict[str, ModelPrice] | None = None,
        *,
        prefer: str = "cost",
    ):
        if prefer not in ("cost", "quality", "latency"):
            raise ValueError(f"未知偏好: {prefer}")
        self.prices = prices or MODEL_PRICES
        self.prefer = prefer

    # ---------- 硬约束 ----------
    def candidates(self, req: TaskRequirement) -> list[ModelPrice]:
        result = []
        for p in self.prices.values():
            if req.needs_vision and not p.supports_vision:
                continue
            if req.needs_tools and not p.supports_tools:
                continue
            if p.context_window < req.min_context:
                continue
            if req.complexity == "high" and p.tier == "economy":
                continue
            result.append(p)
        return result

    # ---------- 路由 ----------
    def route(self, req: TaskRequirement) -> RouteDecision:
        candidates = self.candidates(req)
        if not candidates:
            raise RouterError(f"没有模型满足硬约束: {req}")

        # 预算过滤（按预估用量算 1000 次调用的成本）
        if req.max_cost_per_1k is not None:
            within = [
                p for p in candidates
                if self._cost(p, req) * 1000 <= req.max_cost_per_1k
            ]
            if not within:
                raise RouterError(
                    f"所有候选模型都超出预算 ${req.max_cost_per_1k}/1k 次调用；"
                    f"请降低任务用量、放宽预算或调整复杂度要求"
                )
            candidates = within

        tier_rank = {"flagship": 0, "standard": 1, "economy": 2}
        if self.prefer == "quality":
            candidates.sort(key=lambda p: (tier_rank.get(p.tier, 3), self._cost(p, req)))
        elif self.prefer == "latency":
            candidates.sort(key=lambda p: (p.avg_latency_ms, self._cost(p, req)))
        else:
            candidates.sort(key=lambda p: self._cost(p, req))

        chosen = candidates[0]
        cost = self._cost(chosen, req)
        reason = (
            f"偏好={self.prefer}；硬约束过滤后 {len(candidates)} 个候选；"
            f"选择 {chosen.name}（tier={chosen.tier}，预估 ${cost:.6f}/次，"
            f"约 ${cost * 1000:.2f}/1k 次）"
        )
        return RouteDecision(
            model=chosen.name,
            reason=reason,
            estimated_cost=cost,
            candidates=[(p.name, self._cost(p, req)) for p in candidates],
        )

    @staticmethod
    def _cost(price: ModelPrice, req: TaskRequirement) -> float:
        return estimate_cost(price.name, req.expected_prompt_tokens, req.expected_completion_tokens)
