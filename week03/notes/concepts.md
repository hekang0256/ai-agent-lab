# Week03 学习笔记 · 结构化输出 / Tool Calling / Streaming

## 1. 为什么 Structured Output 是企业 AI 的命脉？

企业系统（CRM/ERP/OA）只吃结构化数据。
如果 LLM 输出的是"一段自然语言"，就没办法自动写入数据库、触发工作流、做后续计算。

**Structured Output = LLM 与传统软件系统的"接口契约"**。

## 2. 三种 Structured Output 策略对比

```
L1: 文字描述 + 后处理 json.loads
    ├─ 优点：通用、不依赖 API 特性
    └─ 缺点：模型可能包裹 markdown、可能缺字段

L2: JSON Mode (response_format={"type":"json_object"})
    ├─ 优点：API 层保证合法 JSON
    └─ 缺点：不保证字段名/类型/必填

L3: JSON Schema + Pydantic + 失败重试
    ├─ 优点：严格符合 schema，生产可用
    └─ 缺点：要写 Pydantic 模型，多花 token
```

**经验法则**：PoC 用 L2，生产用 L3。

## 3. Pydantic 在 LLM 应用中的三个角色

1. **Schema 定义**：`model_json_schema()` 自动生成 JSON Schema 注入 Prompt
2. **数据校验**：`model_validate()` 校验 LLM 输出
3. **错误反馈**：`ValidationError` 信息友好，可直接回喂让模型自我修正

## 4. Tool Calling 深度理解

### 4.1 模型到底做了什么？

模型**没有**执行函数。模型做的是：
- 看到 `tools` 列表（每个工具有 name/description/parameters schema）
- 根据用户问题，决定"要不要调工具、调哪个、传什么参数"
- 返回结构化的 `tool_calls` 字段

**执行权在 Python 代码**，这是安全的关键。

### 4.2 完整消息序列示例

```json
[
  {"role": "system", "content": "你是 CRM 助手"},
  {"role": "user", "content": "查一下客户 1001"},

  // 第 1 次 LLM 返回
  {"role": "assistant", "tool_calls": [
    {"id": "call_001", "type": "function",
     "function": {"name": "get_customer", "arguments": "{\"customer_id\": 1001}"}}
  ]},

  // Python 执行后回喂
  {"role": "tool", "tool_call_id": "call_001", "name": "get_customer",
   "content": "{\"id\":1001,\"name\":\"张三\"}"},

  // 第 2 次 LLM 返回（最终回答）
  {"role": "assistant", "content": "客户 1001 是张三..."}
]
```

### 4.3 工具设计的工程要点

| 要点 | 说明 |
|---|---|
| **描述要精确** | description 直接影响模型选择准确率 |
| **参数要少而明确** | 参数越多，模型出错概率越大 |
| **失败不 crash** | 业务异常也返回结构化 error，让模型看到 |
| **幂等性** | 同一参数多次调用结果应一致（读操作天然幂等，写操作要设计） |
| **高风险要审批** | 删数据、发邮件、转账类工具必须走 Human Approval |

## 5. Streaming 与 SSE 协议

### 5.1 HTTP 层的本质

非流式：
```
Client ──POST──→ Server
       ←──完整 JSON（5秒后）──
```

流式：
```
Client ──POST──→ Server
       ←──data: chunk1──
       ←──data: chunk2──
       ←──data: chunk3──
       ←──data: [DONE]──
```

### 5.2 Python 客户端关键代码

```python
with client.stream("POST", url, json=payload) as resp:
    for line in resp.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]": break
            chunk = json.loads(data)
            yield chunk
```

**易错点**：不能用 `resp.json()`，因为响应还没结束。

## 6. 模型选型的成本/延迟权衡

| 模型档位 | 输入价格 | 输出价格 | 延迟 | 适用 |
|---|---|---|---|---|
| 旗舰（GPT-4o / Claude Opus） | 高 | 高 | 中 | 复杂推理 |
| 中端（DeepSeek-V3 / Qwen-Plus） | 中 | 中 | 低 | 日常任务 |
| 轻量（GPT-4o-mini / Qwen-Turbo） | 低 | 低 | 极低 | 简单分类、提取 |
| Reasoning（o1 / R1） | 高 | 极高 | 高 | 数学、代码、复杂逻辑 |

**经验**：80% 的企业任务用中端模型就够，把旗舰留给真正复杂的场景。

## 7. 本周踩坑总结

1. Mock 路由顺序导致工具无限调用 → role=tool 优先判断
2. 模型输出带 markdown 代码块 → extract_json 多策略解析
3. 中文日期"2026年3月15日" → Pydantic `@field_validator(mode="before")` 预处理
4. `tool` 装饰器忘记导出 → `__init__.py` 的 `__all__` 要同步更新
5. 流式响应用 `httpx.stream()` 上下文管理器，不能用 `.post()`

## 8. 与后续阶段的连接

- **week11 (Tool Calling 专题)**：本周的 `Tool` / `ToolRegistry` 直接复用，扩展 10+ 业务工具
- **week12-13 (MCP)**：把 ToolRegistry 包装成 MCP Server
- **week14-16 (Agent)**：本周的"调用→回喂→回答"就是 Agent Loop 的雏形
- **week21 (Security)**：`requires_approval` hook 在这里已经埋好
