"""MockLLMClient（week03 增强版）：支持 JSON 模式、工具调用、流式输出。

路由规则覆盖本周实验场景：
- 合同提取 → 返回固定合同 JSON
- 工具调用 → 根据用户关键词触发 get_customer / query_contract / send_email
- 流式输出 → 把固定回复切成 chunk yield
"""
from __future__ import annotations

import json
import time
from typing import Any, Iterator

from .client import (
    ChatMessage,
    ChatResponse,
    LLMError,
    StreamChunk,
    ToolCall,
)


class MockLLMClient:
    def __init__(self, model: str = "mock-llm-v3"):
        self.model = model
        self.calls: list[list[ChatMessage]] = []

    # ---------- 非流式 ----------
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        extra: dict | None = None,
    ) -> ChatResponse:
        self.calls.append(messages)
        started = time.perf_counter()
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = next((m.content for m in messages if m.role == "system"), "")

        # 工具结果回喂优先：如果最后一条是 role=tool，说明模型已拿到工具结果，
        # 应该给最终回答而不是再次触发工具调用（避免无限循环）
        if messages[-1].role == "tool":
            tool_payload = messages[-1].content
            return ChatResponse(
                content=f"根据工具返回：{tool_payload[:200]}",
                model=model or self.model,
                prompt_tokens=80, completion_tokens=30, total_tokens=110,
                latency_ms=(time.perf_counter() - started) * 1000,
                finish_reason="stop",
                raw={"mock": True},
            )

        # 工具调用：如果传了 tools 且用户问题匹配，返回 tool_calls
        if tools:
            tc = self._maybe_call_tool(tools, last_user)
            if tc:
                return ChatResponse(
                    content="",
                    model=model or self.model,
                    prompt_tokens=50, completion_tokens=20, total_tokens=70,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    finish_reason="tool_calls",
                    tool_calls=[tc],
                    raw={"mock": True},
                )

        content = self._route(system, last_user, response_format)
        return ChatResponse(
            content=content,
            model=model or self.model,
            prompt_tokens=60, completion_tokens=len(content) // 2, total_tokens=60 + len(content) // 2,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason="stop",
            raw={"mock": True},
        )

    # ---------- 流式 ----------
    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> Iterator[StreamChunk]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = next((m.content for m in messages if m.role == "system"), "")
        full = self._route(system, last_user, None)
        # 每 8 字符切一片
        for i in range(0, len(full), 8):
            yield StreamChunk(delta_content=full[i : i + 8])
            time.sleep(0.001)  # 模拟网络延迟
        yield StreamChunk(delta_content="", finish_reason="stop")

    # ---------- 内部路由 ----------
    def _route(self, system: str, user: str, response_format: dict | None) -> str:
        # ===== 合同提取场景：根据输入文本特征返回对应合同数据 =====
        if "ContractExtraction" in system or "法务" in system:
            return self._mock_contract(user)

        # ===== Person 提取场景（实验3-B）=====
        if "Person" in system or ("姓名" in system and "年龄" in system):
            return self._mock_person(user)

        # ===== name/age/city 提取场景（实验4 模型对比）=====
        if '"name"' in system and '"age"' in system and '"city"' in system:
            return self._mock_name_age_city(user)
        if "姓名" in user and "年龄" in user and "城市" in user:
            return self._mock_name_age_city(user)

        # 兜底：仅含"合同"二字但 system 不像合同提取 → 仍走合同
        if "合同" in user or "contract" in user.lower():
            return self._mock_contract(user)

        # 通用 JSON 模式
        if response_format and response_format.get("type") == "json_object":
            return '{"ok": true, "echo": "mock"}'

        # 默认
        if not user:
            raise LLMError("MockLLMClient 收到空 user 消息")
        return f"[MOCK 回复] 你说：{user[:80]}"

    def _mock_contract(self, user: str) -> str:
        """根据合同文本特征返回对应提取结果。"""
        # 合同02：星辰电商 / 云启软件 / DEV-2026-078
        if "星辰" in user or "云启" in user or "DEV-2026" in user:
            return json.dumps({
                "contract_no": "DEV-2026-078",
                "party_a": "星辰电商集团",
                "party_b": "云启软件工作室",
                "amount_cny": 850000.0,
                "sign_date": "2026-05-10",
                "effective_date": "2026-05-20",
                "expiry_date": "2026-12-31",
                "payment_terms": "分三期支付：签约30%、中期40%、验收30%",
                "key_obligations": ["完成需求规格说明书编写", "每两周进行一次版本迭代", "提供一年免费维护"],
                "risks": ["未明确需求变更的计价方式"],
            }, ensure_ascii=False, indent=2)
        # 默认：合同01（HT-2026-001）
        return json.dumps({
            "contract_no": "HT-2026-001",
            "party_a": "甲方示例科技有限公司",
            "party_b": "乙方示例咨询公司",
            "amount_cny": 1200000.0,
            "sign_date": "2026-03-15",
            "effective_date": "2026-04-01",
            "expiry_date": "2027-03-31",
            "payment_terms": "签约后30日内支付50%，验收后支付50%",
            "key_obligations": ["按月交付项目报告", "提供7x24小时技术支持"],
            "risks": ["违约金条款偏严", "未明确知识产权归属"],
        }, ensure_ascii=False, indent=2)

    def _mock_person(self, user: str) -> str:
        """实验3-B：从文本中提取 Person。"""
        if "张三" in user:
            return json.dumps({"name": "张三", "age": 28, "job": "工程师"}, ensure_ascii=False)
        if "李四" in user:
            return json.dumps({"name": "李四", "age": 35, "job": "产品经理"}, ensure_ascii=False)
        # 默认给一个合法 Person
        return json.dumps({"name": "未知", "age": 0, "job": None}, ensure_ascii=False)

    def _mock_name_age_city(self, user: str) -> str:
        """实验4：从文本中提取 name/age/city。"""
        if "李四" in user:
            return json.dumps({"name": "李四", "age": 35, "city": "北京"}, ensure_ascii=False)
        if "张三" in user:
            return json.dumps({"name": "张三", "age": 28, "city": "上海"}, ensure_ascii=False)
        return json.dumps({"name": "未知", "age": 0, "city": "未知"}, ensure_ascii=False)

    def _maybe_call_tool(self, tools: list[dict], user: str) -> ToolCall | None:
        """根据用户关键词触发对应工具。"""
        tool_names = {t["function"]["name"] for t in tools}
        if "get_customer" in tool_names and "客户" in user:
            # 简单提取数字
            digits = "".join(ch for ch in user if ch.isdigit())
            cid = int(digits) if digits else 1001
            return ToolCall(
                id="call_mock_001",
                name="get_customer",
                arguments={"customer_id": cid},
            )
        if "query_contract" in tool_names and "合同" in user:
            return ToolCall(
                id="call_mock_002",
                name="query_contract",
                arguments={"keyword": user[:30]},
            )
        if "send_email" in tool_names and "邮件" in user:
            return ToolCall(
                id="call_mock_003",
                name="send_email",
                arguments={"to": "boss@example.com", "subject": "测试", "body": user[:50]},
            )
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> "MockLLMClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None
