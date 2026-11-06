"""Token 估算的单元测试：不追求精确，追求稳定与合理。"""
from llm.client import ChatMessage, image_part, text_part
from llm.tokenizer import (
    estimate_content_tokens,
    estimate_messages_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
)


def test_empty_text_is_zero():
    assert estimate_text_tokens("") == 0


def test_english_about_four_chars_per_token():
    assert estimate_text_tokens("hello world") == 3  # 11 字符 / 4 向上取整


def test_chinese_about_one_token_per_char():
    assert estimate_text_tokens("你好世界") == 4


def test_mixed_text_is_sum_of_two_rules():
    # 4 个中文字 + "abcd"（4 字符 = 1 token）
    assert estimate_text_tokens("你好世界abcd") == 5


def test_message_overhead_is_counted():
    msgs = [
        ChatMessage(role="system", content="你是助手"),
        ChatMessage(role="user", content="你好"),
    ]
    total = estimate_messages_tokens(msgs)
    content_only = sum(estimate_text_tokens(m.content) for m in msgs)
    # 总估算 = 内容 + 每条消息开销 + 输出引导，必然大于纯内容
    assert total > content_only
    assert total == content_only + 2 * 4 + 2


def test_image_part_tokens_by_detail():
    low = [text_part("看图"), {"type": "image_url", "image_url": {"url": "data:...", "detail": "low"}}]
    auto = [text_part("看图"), image_part(b"\x89PNG fake", detail="auto")]
    assert estimate_content_tokens(low) == estimate_text_tokens("看图") + 85
    assert estimate_content_tokens(auto) == estimate_text_tokens("看图") + 1024


def test_audio_part_has_positive_estimate():
    part = {"type": "input_audio", "input_audio": {"data": "A" * 4000, "format": "wav"}}
    assert estimate_content_tokens([part]) >= 32


def test_tool_calls_add_tokens():
    plain = ChatMessage(role="assistant", content="")
    with_tools = ChatMessage(
        role="assistant", content="",
        tool_calls=[{"id": "1", "function": {"name": "f", "arguments": "{}"}}],
    )
    assert estimate_message_tokens(with_tools) > estimate_message_tokens(plain)
