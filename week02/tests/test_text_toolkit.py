"""文本工具箱测试（用 MockLLM）。"""
import os

os.environ["LLM_USE_MOCK"] = "true"

import pytest

from experiments.exp04_text_toolkit import TASKS, run_task


def test_all_five_tasks_registered():
    assert set(TASKS.keys()) == {"summarize", "translate", "polish", "keywords", "sentiment"}


def test_summarize():
    result = run_task("summarize", "今天讨论了大模型应用开发的关键路径，包括 Prompt、RAG 和 Agent。")
    assert result  # 非空
    assert "摘要" in result  # Mock 的特征


def test_sentiment():
    result = run_task("sentiment", "今天很开心")
    assert result == "positive"


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        run_task("nonexistent", "hi")
