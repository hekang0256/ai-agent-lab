"""Context Engineering：长上下文管理。

核心思想：**本地全量保留历史，发给模型的消息按预算裁剪**。
上下文窗口是稀缺资源，ContextManager 负责决定「这一轮到底给模型看哪些内容」。

三种策略：
- none     不裁剪（对照组，长对话必然超窗口）
- sliding  滑动窗口：system + 最近若干轮，旧的直接丢弃 → 会遗忘
- summary  摘要压缩：较早的对话压缩成一条摘要消息 + 最近对话原文 → 保留更多长期信息

裁剪以「轮」为最小单位（user 及其后的 assistant/tool 消息为同一轮），
避免把 tool_calls 和对应的 role=tool 结果拆散（OpenAI 会因此报错）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .client import ChatMessage
from .tokenizer import estimate_messages_tokens

Summarizer = Callable[[list[ChatMessage]], str]


def heuristic_summary(messages: list[ChatMessage]) -> str:
    """不依赖 LLM 的兜底摘要：抽取每轮 user 发言的开头作为要点。"""
    lines: list[str] = []
    for m in messages:
        if m.role == "user":
            t = m.text.strip().replace("\n", " ")
            if t:
                lines.append(f"- {t[:60]}")
    if not lines:
        return "（较早对话无用户发言）"
    return "较早对话的要点：\n" + "\n".join(lines[:15])


@dataclass
class ContextStats:
    """一次上下文构建的统计，用于实验对比与可观测性。"""
    strategy: str
    before_tokens: int
    after_tokens: int
    dropped_messages: int = 0      # 既没保留也没进摘要的消息数
    summarized_messages: int = 0   # 被压缩进摘要的消息数

    @property
    def saved_tokens(self) -> int:
        return self.before_tokens - self.after_tokens


class ContextManager:
    def __init__(
        self,
        *,
        max_tokens: int = 4096,
        reserve_output: int = 512,
        strategy: str = "sliding",
        keep_recent: int = 6,
        summarizer: Summarizer | None = None,
    ):
        """
        max_tokens:      模型上下文窗口大小
        reserve_output:  给模型输出预留的 token（输入预算 = max_tokens - reserve_output）
        strategy:        none / sliding / summary
        keep_recent:     summary 策略下，保留原文的最近消息条数（按整轮对齐，可能略多）
        summarizer:      摘要函数，缺省用启发式摘要；生产环境注入 LLM 摘要
        """
        if strategy not in ("none", "sliding", "summary"):
            raise ValueError(f"未知策略: {strategy}")
        self.max_tokens = max_tokens
        self.reserve_output = reserve_output
        self.strategy = strategy
        self.keep_recent = keep_recent
        self.summarizer = summarizer

    @property
    def budget(self) -> int:
        return self.max_tokens - self.reserve_output

    # ---------- 对外入口 ----------
    def build_messages(
        self,
        history: list[ChatMessage],
        *,
        system: str | None = None,
    ) -> tuple[list[ChatMessage], ContextStats]:
        """根据完整 history 构造本轮要发给模型的消息列表（不修改 history 本身）。"""
        sys_msgs = [ChatMessage(role="system", content=system)] if system else []
        all_msgs = sys_msgs + list(history)
        before = estimate_messages_tokens(all_msgs)

        if self.strategy == "none" or before <= self.budget:
            return all_msgs, ContextStats(self.strategy, before, before)

        if self.strategy == "sliding":
            kept = self._fit_sliding(history, self.budget - estimate_messages_tokens(sys_msgs))
            msgs = sys_msgs + kept
            return msgs, ContextStats(
                self.strategy, before, estimate_messages_tokens(msgs),
                dropped_messages=len(history) - len(kept),
            )

        # summary：旧消息 → 摘要；最近消息 → 原文；仍超预算则对最近消息再滑动
        old, recent = self._split_old_recent(history)
        if not old:
            kept = self._fit_sliding(history, self.budget - estimate_messages_tokens(sys_msgs))
            msgs = sys_msgs + kept
            return msgs, ContextStats(
                self.strategy, before, estimate_messages_tokens(msgs),
                dropped_messages=len(history) - len(kept),
            )
        summarize = self.summarizer or heuristic_summary
        # 摘要本身也占预算：最多用掉预算的 40%，超出按行截断，防止「摘要比原文还长」
        summary_text = self._truncate_to_tokens(
            summarize(old), max(64, int(self.budget * 0.4)),
        )
        summary_msg = ChatMessage(
            role="system",
            content=f"# 较早对话摘要\n{summary_text}",
        )
        fixed_tokens = estimate_messages_tokens(sys_msgs + [summary_msg])
        kept_recent = self._fit_sliding(recent, self.budget - fixed_tokens)
        msgs = sys_msgs + [summary_msg] + kept_recent
        return msgs, ContextStats(
            self.strategy, before, estimate_messages_tokens(msgs),
            dropped_messages=len(recent) - len(kept_recent),
            summarized_messages=len(old),
        )

    # ---------- 内部 ----------
    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """按行截断文本，保证估算 token 不超过 max_tokens。"""
        from .tokenizer import estimate_text_tokens

        lines = text.splitlines()
        kept: list[str] = []
        used = 0
        for line in lines:
            cost = estimate_text_tokens(line) + 1
            if kept and used + cost > max_tokens:
                kept.append("- …（摘要已按预算截断）")
                break
            kept.append(line)
            used += cost
        return "\n".join(kept)

    @staticmethod
    def _group_turns(messages: list[ChatMessage]) -> list[list[ChatMessage]]:
        """按轮分组：user 消息开启新一轮，其后的 assistant/tool 归入同一轮。"""
        turns: list[list[ChatMessage]] = []
        for m in messages:
            if m.role == "user" or not turns:
                turns.append([m])
            else:
                turns[-1].append(m)
        return turns

    def _fit_sliding(self, history: list[ChatMessage], avail_tokens: int) -> list[ChatMessage]:
        """从最新一轮开始往回装，装不下就停；至少保留最新一轮（保证对话能继续）。"""
        turns = self._group_turns(history)
        kept_turns: list[list[ChatMessage]] = []
        used = 0
        for turn in reversed(turns):
            t = estimate_messages_tokens(turn)
            if kept_turns and used + t > avail_tokens:
                break
            kept_turns.append(turn)
            used += t
            if used > avail_tokens:
                break  # 最新一轮本身就超预算：保留它，交由 max_tokens 限制输出
        kept_turns.reverse()
        return [m for turn in kept_turns for m in turn]

    def _split_old_recent(
        self, history: list[ChatMessage]
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """把历史切成「较早（待摘要）」和「最近（保留原文）」，按整轮对齐。"""
        turns = self._group_turns(history)
        recent_turns: list[list[ChatMessage]] = []
        count = 0
        for turn in reversed(turns):
            if recent_turns and count >= self.keep_recent:
                break
            recent_turns.append(turn)
            count += len(turn)
        recent_turns.reverse()
        old_count = len(turns) - len(recent_turns)
        old = [m for turn in turns[:old_count] for m in turn]
        recent = [m for turn in recent_turns for m in turn]
        return old, recent
