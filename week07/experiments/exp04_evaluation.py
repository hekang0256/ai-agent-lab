"""实验7-D：50 条问题集验收——召回、答案关键词、引用与拒答。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from apps.policy_qa import DOCS_PATH, create_service
from llm.document import load_documents
from llm.evaluation import evaluate, load_question_set, save_report

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="week07 50 条问题集评估")
    parser.add_argument("--store", choices=["memory", "pg"], default="memory")
    parser.add_argument("--output", type=Path, help="自定义报告路径，避免覆盖其它模型结果")
    parser.add_argument("--resume", action="store_true", help="从 output 对应的 checkpoint 续跑")
    parser.add_argument("--delay", type=float, default=float(os.getenv("EVAL_DELAY_SECONDS", "0")))
    args = parser.parse_args()
    service = create_service(args.store)
    service.index_documents(load_documents(DOCS_PATH))
    questions = load_question_set(ROOT / "data" / "eval_questions.json")
    filename = "evaluation_results_pg.json" if args.store == "pg" else "evaluation_results.json"
    output = args.output or ROOT / "data" / filename
    checkpoint = output.with_suffix(".checkpoint.json")
    report = evaluate(
        service, questions, checkpoint_path=checkpoint,
        resume=args.resume, pause_seconds=args.delay,
    )
    llm_model = getattr(getattr(service.llm_client, "config", None), "model", None) or getattr(service.llm_client, "model", "unknown")
    report = {
        "configuration": {
            "store": args.store,
            "embedding_model": getattr(service.embedding_client, "model", "unknown"),
            "llm_model": llm_model,
            "top_k": service.top_k,
            "min_score": service.min_score,
        },
        **report,
    }
    save_report(report, output)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    failures = [case for case in report["cases"] if not case["passed"]]
    print(f"\n失败 {len(failures)} 条：")
    for case in failures:
        print(f"- {case['id']} {case['query']} → 检索 {case['retrieved_document_ids']}，引用 {case['cited_document_ids']}")
    print(f"\n完整结果已写入 {output}")


if __name__ == "__main__":
    main()
