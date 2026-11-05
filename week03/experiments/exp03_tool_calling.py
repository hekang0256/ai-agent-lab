"""实验3-C：Tool Calling 完整链路。

演示「用户提问 → 模型决定调工具 → 执行工具 → 把结果回喂 → 模型给最终回答」闭环。

运行：python -m experiments.exp03_tool_calling
"""
from __future__ import annotations

from llm import get_client, ToolRegistry, tool
from llm.client import ChatMessage


# ---------- 定义工具 ----------
@tool(description="根据客户ID查询客户信息")
def get_customer(customer_id: int) -> dict:
    """模拟 CRM 查询。"""
    fake_db = {
        1001: {"id": 1001, "name": "张三", "tier": "VIP", "balance": 50000},
        1002: {"id": 1002, "name": "李四", "tier": "普通", "balance": 3000},
    }
    return fake_db.get(customer_id, {"error": f"客户 {customer_id} 不存在"})


@tool(description="发送邮件", requires_approval=True)
def send_email(to: str, subject: str, body: str) -> dict:
    """真实场景这是高风险操作，应该走人工审批。"""
    return {"sent": True, "to": to, "subject": subject}


# ---------- 注册表 ----------
def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(get_customer)
    reg.register(send_email)
    return reg


# ---------- 完整 Agent Loop（简化版）----------
def run_with_tools(user_input: str, max_iterations: int = 3) -> str:
    client = get_client()
    registry = build_registry()

    messages = [
        ChatMessage(role="system", content="你是一个 CRM 助手，可以用工具查询客户、发送邮件。"),
        ChatMessage(role="user", content=user_input),
    ]

    for step in range(max_iterations):
        print(f"\n--- Step {step + 1} ---")
        resp = client.chat(messages, tools=registry.to_openai_schema())
        print(f"模型 finish_reason={resp.finish_reason}  has_tool_calls={resp.has_tool_calls}")

        if not resp.has_tool_calls:
            # 模型给了最终回答
            return resp.content

        # 把 assistant 的 tool_calls 加入历史
        messages.append(ChatMessage(
            role="assistant",
            content=resp.content or "",
            tool_calls=[{"id": tc.id, "type": "function",
                         "function": {"name": tc.name, "arguments": __import__("json").dumps(tc.arguments, ensure_ascii=False)}}
                        for tc in resp.tool_calls],
        ))

        # 执行每个工具调用
        for tc in resp.tool_calls:
            print(f"  调用工具: {tc.name}({tc.arguments})")
            result = registry.dispatch(tc.id, tc.name, tc.arguments)
            print(f"  工具结果: ok={result.ok}  result={result.result}  error={result.error}")
            messages.append(ChatMessage(
                role="tool",
                content=result.to_message_content(),
                tool_call_id=tc.id,
                name=tc.name,
            ))

    return "达到最大迭代次数仍未完成"


if __name__ == "__main__":
    print("=" * 60)
    print("场景1：查询客户")
    print("=" * 60)
    answer = run_with_tools("帮我查一下客户 1001 的信息")
    print(f"\n最终回答: {answer}")

    print("\n" + "=" * 60)
    print("场景2：模型判断不需要工具")
    print("=" * 60)
    answer = run_with_tools("今天天气怎么样？")
    print(f"\n最终回答: {answer}")
