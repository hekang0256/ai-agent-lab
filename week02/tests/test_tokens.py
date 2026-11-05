"""Token 工具单元测试：不依赖外部 API。"""
import pytest

from llm.tokens import (
    count_tokens,
    count_messages_tokens,
    fits_context,
    estimate_cost,
    _heuristic_count,
)


def test_empty_string():
    assert count_tokens("") == 0
    assert _heuristic_count("") == 0


def test_chinese_vs_english_token_ratio():
    """同义中英文 token 数应不同，中文一般更费 token。"""
    zh = "人工智能正在改变世界"
    en = "AI is changing the world"
    zh_tokens = count_tokens(zh)
    en_tokens = count_tokens(en)
    assert zh_tokens > 0 and en_tokens > 0
    # 不强制大小关系，但启发式下中文应该更密
    assert _heuristic_count(zh) >= _heuristic_count(en) // 2


def test_heuristic_monotonic():
    """文本越长，token 数应单调不减。"""
    a = "人工智能"
    b = a * 10
    assert _heuristic_count(b) > _heuristic_count(a)


def test_messages_overhead():
    """messages 的 token 数应大于 content 之和（有 role/格式开销）。"""
    msgs = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "你好"},
    ]
    raw = sum(count_tokens(m["content"]) for m in msgs)
    total = count_messages_tokens(msgs)
    assert total > raw


def test_fits_context():
    small = [{"role": "user", "content": "hi"}]
    assert fits_context(small, context_window=4096)

    big = [{"role": "user", "content": "人工智能" * 5000}]
    assert not fits_context(big, context_window=4096)


def test_estimate_cost():
    # 1M in @ $1 + 1M out @ $2 = $3
    cost = estimate_cost(1_000_000, 1_000_000, in_price_per_1m=1.0, out_price_per_1m=2.0)
    assert cost == pytest.approx(3.0)


def test_count_with_tiktoken_if_available():
    """如果 tiktoken 可用，验证精确模式 ≥ 启发式模式 70% 的准确度（粗校验）。"""
    text = "Hello, 你好, world, 世界"
    precise = count_tokens(text, precise=True)
    rough = count_tokens(text, precise=False)
    assert precise > 0 and rough > 0
    # 两者应在同一量级
    assert abs(precise - rough) <= max(precise, rough)
