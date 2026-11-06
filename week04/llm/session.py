"""ChatSession：第1阶段能力的集大成者。

把前五周（week01~week04）的积木组装成一个可用的多轮对话助手：

    用户输入
      ↓
    记忆指令？（记住/忘掉 → MemoryStore，直接确认）
      ↓
    组装 system = 基础人设 + 用户画像（长期记忆注入）
      ↓
    ContextManager 按预算裁剪历史（sliding / summary）
      ↓
    LLM（可带 tools schema）
      ↓ 模型要求调工具？
    ToolRegistry.dispatch → role=tool 回喂 → 再次调用（最多 max_tool_rounds 轮）
      ↓
    最终回答写入 history，CostTracker 记录每一次调用成本

注意：history 永远全量保留在本地；裁剪只影响发给模型的内容。
"""
from __future__ import annotations

import json

from .client import ChatMessage, get_client, image_part, text_part
from .context import ContextManager, ContextStats
from .memory import MemoryStore, parse_memory_command
from .pricing import CostTracker
from .tools import ToolRegistry

DEFAULT_SYSTEM = "你是企业 AI 助手，回答简洁、专业；可以使用提供的工具查询业务数据。"


class ChatSession:
    def __init__(
        self,
        llm=None,
        *,
        system: str = DEFAULT_SYSTEM,
        memory: MemoryStore | None = None,
        user_id: str = "default",
        context: ContextManager | None = None,
        registry: ToolRegistry | None = None,
        tracker: CostTracker | None = None,
        max_tool_rounds: int = 3,
    ):
        self.llm = llm or get_client()
        self.base_system = system
        self.memory = memory
        self.user_id = user_id
        self.context = context or ContextManager(
            strategy="sliding", max_tokens=8192, reserve_output=1024,
        )
        self.registry = registry
        self.tracker = tracker or CostTracker()
        self.max_tool_rounds = max_tool_rounds
        self.history: list[ChatMessage] = []
        self.last_context_stats: ContextStats | None = None

        # summary 策略默认用 LLM 做摘要（也可注入启发式摘要以便完全离线）
        if self.context.strategy == "summary" and self.context.summarizer is None:
            self.context.summarizer = self._llm_summarize

    # ---------- 对外入口 ----------
    def ask(self, text: str, *, images: list[str] | None = None) -> str:
        """一轮对话。text 可以是普通问题，也可以是「记住…/忘掉…」记忆指令。"""
        if self.memory:
            cmd = parse_memory_command(text)
            if cmd:
                reply = self._handle_memory(cmd)
                self.history.append(ChatMessage(role="user", content=text))
                self.history.append(ChatMessage(role="assistant", content=reply))
                return reply

        if images:
            content = [text_part(text)] + [image_part(src) for src in images]
            self.history.append(ChatMessage(role="user", content=content))
        else:
            self.history.append(ChatMessage(role="user", content=text))

        reply = self._run_loop()
        self.history.append(ChatMessage(role="assistant", content=reply))
        return reply

    def reset(self) -> None:
        """清空短期记忆（对话历史）；长期记忆保留。"""
        self.history.clear()
        self.last_context_stats = None

    def stats(self) -> dict:
        return {
            "turns": sum(1 for m in self.history if m.role == "user"),
            "history_messages": len(self.history),
            "total_tokens": self.tracker.total_tokens,
            "total_cost": self.tracker.total_cost,
            "last_context": self.last_context_stats,
        }

    # ---------- 内部 ----------
    def _system_prompt(self) -> str:
        system = self.base_system
        if self.memory:
            profile = self.memory.profile_text(user_id=self.user_id)
            if profile:
                system = f"{system}\n\n{profile}"
        return system

    def _run_loop(self) -> str:
        messages, stats = self.context.build_messages(self.history, system=self._system_prompt())
        self.last_context_stats = stats
        tools = self.registry.to_openai_schema() if self.registry and len(self.registry) else None

        resp = self.llm.chat(messages, tools=tools)
        self.tracker.add_response(resp, task="chat")

        rounds = 0
        while resp.has_tool_calls and self.registry and rounds < self.max_tool_rounds:
            rounds += 1
            assistant_msg = ChatMessage(
                role="assistant",
                content=resp.content or "",
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in resp.tool_calls
                ],
            )
            self.history.append(assistant_msg)
            messages.append(assistant_msg)
            for tc in resp.tool_calls:
                result = self.registry.dispatch(tc.id, tc.name, tc.arguments)
                tool_msg = ChatMessage(
                    role="tool", content=result.to_message_content(),
                    tool_call_id=tc.id, name=tc.name,
                )
                self.history.append(tool_msg)
                messages.append(tool_msg)
            resp = self.llm.chat(messages, tools=tools)
            self.tracker.add_response(resp, task="chat")
        return resp.content

    def _handle_memory(self, cmd) -> str:
        display = {"name": "名字", "likes": "喜好"}.get(cmd.key, cmd.key)
        if cmd.action == "remember":
            self.memory.remember(cmd.key, cmd.value, user_id=self.user_id)
            return f"好的，我记住了：你的{display}是{cmd.value}。"
        existed = self.memory.forget(cmd.key, user_id=self.user_id)
        if existed:
            return f"好的，已经忘掉你的{display}。"
        return f"我的长期记忆里本来就没有你的{display}。"

    def _llm_summarize(self, messages: list[ChatMessage]) -> str:
        """用 LLM 把旧对话压缩成摘要（记账到 tracker，成本可观测）。"""
        transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
        resp = self.llm.chat(
            [
                ChatMessage(
                    role="system",
                    content="ContextSummary：你是对话摘要器，用不超过 120 字概括对话要点，"
                            "保留关键实体（人名/编号/金额/偏好）。",
                ),
                ChatMessage(role="user", content=transcript),
            ],
            temperature=0.0,
        )
        self.tracker.add_response(resp, task="summary")
        return resp.content
