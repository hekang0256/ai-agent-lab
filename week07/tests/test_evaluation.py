from pathlib import Path

from llm.evaluation import evaluate, load_question_set

ROOT = Path(__file__).resolve().parents[1]


def test_question_set_has_stage_acceptance_size():
    questions = load_question_set(ROOT / "data" / "eval_questions.json")
    assert len(questions) == 50
    assert sum(bool(q["expected_document_ids"]) for q in questions) == 42
    assert sum(not q["expected_document_ids"] for q in questions) == 8


def test_offline_evaluation_report(service):
    questions = load_question_set(ROOT / "data" / "eval_questions.json")
    report = evaluate(service, questions)
    metrics = report["metrics"]
    assert metrics["total"] == 50
    assert len(report["cases"]) == 50
    assert metrics["retrieval_recall"] >= 0.8
    assert metrics["citation_accuracy"] >= 0.8
    assert metrics["refusal_accuracy"] == 1.0
