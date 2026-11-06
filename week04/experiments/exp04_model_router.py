"""实验4-D：模型选型（硬约束 / 软偏好 / 预算）。

观察同一批任务在不同 prefer 下的选择差异，以及预算过紧时的失败行为。
"""
from __future__ import annotations

from llm.router import ModelRouter, RouterError, TaskRequirement

TASKS = [
    ("工单分类", TaskRequirement(complexity="low", expected_prompt_tokens=800,
                                 expected_completion_tokens=100)),
    ("长文摘要", TaskRequirement(complexity="medium", expected_prompt_tokens=6000,
                                 expected_completion_tokens=600)),
    ("合同风险审查", TaskRequirement(complexity="high", expected_prompt_tokens=12000,
                                     expected_completion_tokens=1500)),
    ("发票识别", TaskRequirement(complexity="medium", needs_vision=True,
                                 expected_prompt_tokens=2500, expected_completion_tokens=400)),
    ("超长文档问答", TaskRequirement(complexity="medium", min_context=100000,
                                     expected_prompt_tokens=60000, expected_completion_tokens=800)),
]


def main() -> None:
    for prefer in ("cost", "quality", "latency"):
        print(f"\n===== prefer={prefer} =====")
        router = ModelRouter(prefer=prefer)
        for name, req in TASKS:
            d = router.route(req)
            candidates = ", ".join(f"{m}(${c:.4f})" for m, c in d.candidates)
            print(f"\n[{name}] → {d.model}")
            print(f"  理由：{d.reason}")
            print(f"  候选：{candidates}")

    print("\n===== 预算约束演示 =====")
    router = ModelRouter()
    req = TaskRequirement(complexity="low", max_cost_per_1k=0.5)
    d = router.route(req)
    print(f"预算 $0.5/1k：{d.model}（{d.reason}）")
    try:
        router.route(TaskRequirement(complexity="high", needs_vision=True, max_cost_per_1k=0.000001))
    except RouterError as e:
        print(f"预算过紧：{e}")


if __name__ == "__main__":
    main()
