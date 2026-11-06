"""成本核算与模型路由测试。"""
import pytest

from llm.client import ChatResponse
from llm.pricing import CostTracker, estimate_cost
from llm.router import ModelRouter, RouterError, TaskRequirement


# ---------- 成本 ----------
def test_estimate_cost_formula():
    # deepseek-chat：0.27 + 1.10 = 1.37 USD / (1M in + 1M out)
    assert estimate_cost("deepseek-chat", 1_000_000, 1_000_000) == pytest.approx(1.37)


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost("no-such-model", 1000, 1000) == 0.0


def test_cost_tracker_accumulates():
    tracker = CostTracker()
    tracker.add("deepseek-chat", 1000, 500)
    tracker.add("gpt-4o", 1000, 500)
    assert len(tracker.records) == 2
    assert tracker.total_tokens == 3000
    assert tracker.total_cost == pytest.approx(
        estimate_cost("deepseek-chat", 1000, 500) + estimate_cost("gpt-4o", 1000, 500)
    )
    assert set(tracker.by_model()) == {"deepseek-chat", "gpt-4o"}
    assert "合计" in tracker.report()


def test_cost_tracker_add_response():
    tracker = CostTracker()
    resp = ChatResponse(content="x", model="qwen-plus", prompt_tokens=2000,
                        completion_tokens=1000, total_tokens=3000, latency_ms=12.0)
    record = tracker.add_response(resp, task="t1")
    assert record.task == "t1"
    assert record.cost == pytest.approx(estimate_cost("qwen-plus", 2000, 1000))


# ---------- 路由：硬约束 ----------
def test_vision_constraint_filters_models():
    router = ModelRouter()
    req = TaskRequirement(needs_vision=True)
    names = {p.name for p in router.candidates(req)}
    assert names == {"qwen-vl-plus", "gpt-4o-mini", "gpt-4o"}


def test_context_window_constraint():
    router = ModelRouter()
    names = {p.name for p in router.candidates(TaskRequirement(min_context=100000))}
    assert "deepseek-chat" not in names  # 65536 窗口不够
    assert "qwen-plus" in names


def test_high_complexity_excludes_economy():
    router = ModelRouter()
    names = {p.name for p in router.candidates(TaskRequirement(complexity="high"))}
    assert "gpt-4o-mini" not in names
    assert "deepseek-chat" not in names


# ---------- 路由：偏好 ----------
def test_cost_prefers_cheapest():
    router = ModelRouter(prefer="cost")
    req = TaskRequirement(complexity="low", expected_prompt_tokens=800,
                          expected_completion_tokens=100)
    assert router.route(req).model == "gpt-4o-mini"


def test_cost_high_complexity_picks_cheapest_capable():
    router = ModelRouter(prefer="cost")
    req = TaskRequirement(complexity="high", expected_prompt_tokens=12000,
                          expected_completion_tokens=1500)
    assert router.route(req).model == "qwen-plus"


def test_quality_prefers_flagship():
    router = ModelRouter(prefer="quality")
    req = TaskRequirement(complexity="high")
    assert router.route(req).model == "gpt-4o"


def test_latency_prefers_fastest():
    router = ModelRouter(prefer="latency")
    req = TaskRequirement(complexity="low")
    assert router.route(req).model == "gpt-4o-mini"  # avg_latency 700 最小


def test_vision_high_complexity_routes_to_vl():
    router = ModelRouter(prefer="cost")
    req = TaskRequirement(complexity="high", needs_vision=True)
    assert router.route(req).model == "qwen-vl-plus"


def test_decision_is_explainable():
    router = ModelRouter()
    d = router.route(TaskRequirement(complexity="low"))
    assert d.reason and d.candidates and d.estimated_cost > 0


# ---------- 路由：预算 ----------
def test_budget_filters_expensive_models():
    router = ModelRouter()
    req = TaskRequirement(complexity="low", max_cost_per_1k=0.5)
    d = router.route(req)
    assert d.estimated_cost * 1000 <= 0.5


def test_budget_too_tight_raises():
    router = ModelRouter()
    with pytest.raises(RouterError):
        router.route(TaskRequirement(max_cost_per_1k=1e-9))


def test_no_candidate_raises():
    router = ModelRouter()
    with pytest.raises(RouterError):
        router.route(TaskRequirement(min_context=10_000_000))
