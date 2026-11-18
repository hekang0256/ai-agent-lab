"""第2阶段验收：50 条问题集的基础 RAG 评估。

answer_keyword_hit 是可离线复现的关键词代理指标，不等于真实语义评分；
真实项目还应增加人工标注或可信的 LLM-as-a-judge，并保留抽样复核。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .rag import INSUFFICIENT_ANSWER, RAGService


def load_question_set(path: str | Path) -> list[dict]:
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(questions) < 50:
        raise ValueError(f"第2阶段验收要求至少 50 条问题，当前只有 {len(questions)} 条")
    ids = [item["id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("问题集 id 必须唯一")
    return questions


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _case_from_response(item: dict, response) -> dict:
    expected_docs = set(item.get("expected_document_ids", []))
    retrieved_docs = {r.metadata.get("document_id") for r in response.retrieved}
    cited_docs = {c.document_id for c in response.citations}
    should_answer = bool(expected_docs)
    retrieval_hit = not should_answer or bool(expected_docs & retrieved_docs)
    keywords = item.get("expected_answer_keywords", [])
    answer_keyword_hit = all(keyword in response.answer for keyword in keywords) if should_answer else INSUFFICIENT_ANSWER in response.answer
    citation_hit = (bool(cited_docs) and cited_docs <= expected_docs and not response.invalid_citation_indexes) if should_answer else response.refused
    refusal_correct = response.refused if not should_answer else not response.refused
    passed = (retrieval_hit and answer_keyword_hit and citation_hit and not response.refused) if should_answer else (response.refused and INSUFFICIENT_ANSWER in response.answer)
    return {
        "id": item["id"], "query": item["query"], "should_answer": should_answer,
        "expected_document_ids": sorted(expected_docs),
        "retrieved_document_ids": sorted(retrieved_docs), "cited_document_ids": sorted(cited_docs),
        "retrieval_hit": retrieval_hit, "answer_keyword_hit": answer_keyword_hit,
        "citation_hit": citation_hit, "refusal_correct": refusal_correct, "passed": passed,
        "answer": response.answer, "citation_status": response.citation_status,
    }


def _metrics(cases: list[dict]) -> dict:
    answerable = [case for case in cases if case["should_answer"]]
    unanswerable = [case for case in cases if not case["should_answer"]]
    return {
        "total": len(cases),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "retrieval_recall": _ratio(sum(c["retrieval_hit"] for c in answerable), len(answerable)),
        "answer_keyword_accuracy": _ratio(
            sum(c["answer_keyword_hit"] and c["refusal_correct"] for c in answerable), len(answerable)
        ),
        "citation_accuracy": _ratio(sum(c["citation_hit"] for c in answerable), len(answerable)),
        "refusal_accuracy": _ratio(sum(c["passed"] for c in unanswerable), len(unanswerable)),
        "overall_pass_rate": _ratio(sum(c["passed"] for c in cases), len(cases)),
    }


def evaluate(
    service: RAGService, questions: list[dict], *, top_k: int = 3,
    min_score: float = 0.35, checkpoint_path: str | Path | None = None,
    resume: bool = False, pause_seconds: float = 0.0,
) -> dict[str, Any]:
    """逐条评估；checkpoint 用于真实 API 中断后续跑，避免重复计费。"""
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed: dict[str, dict] = {}
    if checkpoint and resume and checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        completed = {case["id"]: case for case in payload.get("cases", [])}
        print(f"从 checkpoint 恢复 {len(completed)} 条：{checkpoint}")
    cases = []
    for item in questions:
        if item["id"] in completed:
            cases.append(completed[item["id"]])
            continue
        try:
            response = service.answer(item["query"], top_k=top_k, min_score=min_score)
        except Exception as error:
            if checkpoint:
                save_report({
                    "complete": False, "failed_question_id": item["id"],
                    "error": f"{type(error).__name__}: {error}", "cases": cases,
                }, checkpoint)
            raise RuntimeError(
                f"评估在 {item['id']} 中断，已完成 {len(cases)}/{len(questions)}；"
                f"修复后可使用 --resume 继续"
            ) from error
        cases.append(_case_from_response(item, response))
        if checkpoint:
            save_report({"complete": len(cases) == len(questions), "cases": cases}, checkpoint)
        print(f"评估进度 {len(cases)}/{len(questions)}：{item['id']} {'PASS' if cases[-1]['passed'] else 'FAIL'}")
        if pause_seconds > 0 and len(cases) < len(questions):
            time.sleep(pause_seconds)
    return {"metrics": _metrics(cases), "cases": cases}


def save_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
