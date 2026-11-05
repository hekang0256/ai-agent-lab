# Week03 ｜ 结构化输出 + Tool Calling + Streaming

> 《AI → 智能体》24周系统学习路线 · 第1阶段第2周（总第3周）
> 实验目标：让 LLM 稳定输出结构化数据；让 LLM 调用真实 Python 函数；实现流式响应；交付合同提取器 + 模型对比框架。
> 验收标准：能用 Pydantic 定义 schema 并让 LLM 严格遵守；能写出完整的 Tool Calling 闭环；理解流式与非流式的差异。

## 一、本周做了什么

| 实验 | 内容 | 结果 |
|------|------|------|
| 实验3-A | JSON Mode（`response_format={"type":"json_object"}`） | ✅ |
| 实验3-B | JSON Schema + Pydantic 强约束 + 失败重试 | ✅ |
| 实验3-C | Tool Calling 完整链路（注册→调用→回喂→回答） | ✅ |
| 实验3-D | SSE 流式响应 + TTFT 测量 | ✅ |
| **实验2** | **合同信息 JSON 提取工具**（企业真实场景） | ✅ |
| **实验4** | **模型对比实验框架**（准确性/速度/成本/稳定性） | ✅ |
| 测试 | pytest 22 条 | ✅ 22 passed |

## 二、架构图

```text
┌────────────────────────────────────────────────────────────────┐
│  apps/contract_extractor.py        apps/model_compare.py        │
│  experiments/exp01 ~ exp04                                      │
└────────────────────────────┬───────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼─────────┐  ┌──────▼──────────┐
│ StructuredClient│  │  ToolRegistry    │  │  StreamChunk    │
│  (Pydantic 校验)│  │  (Tool 装饰器)   │  │  (SSE 解析)     │
│  · extract()    │  │  · dispatch()    │  │  · chat_stream()│
│  · 失败重试     │  │  · 参数校验      │  │  · TTFT 测量    │
└───────┬────────┘  └────────┬─────────┘  └──────┬──────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │   LLMClient (httpx)  │
                  │   OpenAI 兼容协议     │
                  │   超时/重试/Token统计 │
                  └──────────┬───────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                                         │
┌───────▼─────────┐                      ┌────────▼────────┐
│ 真实 LLM API    │                      │  MockLLMClient  │
│ (DeepSeek/Kimi/ │                      │  · 模拟工具调用  │
│  Qwen/OpenAI)   │                      │  · 模拟流式输出  │
└─────────────────┘                      │  · 模拟 JSON    │
                                         └─────────────────┘
```

## 三、目录结构

```text
week03/
├── llm/
│   ├── client.py           # 增强版：支持 tools / tool_choice / stream
│   ├── structured.py       # StructuredClient + extract_json（处理 markdown 包裹）
│   ├── tools.py            # @tool 装饰器 + ToolRegistry
│   └── mock.py             # 支持工具调用/流式的 Mock
├── experiments/
│   ├── exp01_json_mode.py
│   ├── exp02_json_schema.py
│   ├── exp03_tool_calling.py
│   └── exp04_streaming.py
├── apps/
│   ├── contract_extractor.py   # 实验2：合同提取器
│   └── model_compare.py        # 实验4：模型对比框架
├── data/
│   ├── contracts/              # 测试合同文本
│   │   ├── contract_01.txt
│   │   └── contract_02.txt
│   ├── extraction_results.json # 提取结果
│   ├── compare_results.json    # 对比实验原始数据
│   └── compare_summary.md      # 对比表
├── tests/                      # 22 个 pytest
├── .env.example
├── requirements.txt
└── README.md
```

## 四、如何运行

```bash
# 1. 准备环境
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置 API（同 week02）
#    设置 LLM_USE_MOCK=true 走 Mock；否则填 .env

# 3. 跑实验
python -m experiments.exp01_json_mode
python -m experiments.exp02_json_schema
python -m experiments.exp03_tool_calling
python -m experiments.exp04_streaming

# 4. 跑应用
python -m apps.contract_extractor data\contracts\contract_01.txt   # 单文件
python -m apps.contract_extractor data\contracts\                  # 批量
python -m apps.model_compare --runs 3                              # 对比实验（Mock）
python -m apps.model_compare --real --runs 5 --models deepseek-chat # 真实 API

# 5. 跑测试
pytest
```

## 五、本周学习要点

### 5.1 结构化输出的三个层级

| 层级 | 手段 | 保证强度 | 适用场景 |
|---|---|---|---|
| **L1 文字描述** | Prompt 里写"输出 JSON" | ❌ 不保证 | 快速试验 |
| **L2 JSON Mode** | `response_format={"type":"json_object"}` | ✅ 合法 JSON<br>❌ 不保证 schema | 一般场景 |
| **L3 JSON Schema** | Pydantic 注入 Schema + 校验 + 重试 | ✅ 严格 schema | **企业生产** |

**核心经验**：L3 = JSON Mode + Pydantic 校验 + 失败回喂重试。`StructuredClient` 把这套封装成一行调用。

### 5.2 失败重试的关键：把错误回喂

```python
# 第一次：模型输出不合规
messages = [system, user]
resp = llm.chat(messages)
# 校验失败，把 ValidationError 回喂
messages.append(assistant(resp.content))
messages.append(user(f"上次输出错误：{e}，请重新输出"))
# 第二次：模型自我修正
resp = llm.chat(messages)
```

实验数据显示：Pydantic 错误信息对模型非常友好，二次成功率 > 90%。

### 5.3 Tool Calling 的本质

**不是模型真的"执行"了工具，而是模型返回「我想调这个函数 + 参数」的结构化请求，由 Python 代码执行后再把结果回喂。**

完整闭环：
```
用户 → LLM(带 tools schema)
       ↓ 模型判断要调工具
       返回 {name, arguments}
       ↓
       Python 执行函数
       ↓
       把结果作为 role=tool 消息回喂
       ↓
       LLM 基于工具结果生成最终回答
       ↓
用户 ← 自然语言回答
```

**关键设计**：
- `@tool` 装饰器自动从函数签名生成 JSON Schema（类型 hints → Pydantic 模型）
- `ToolRegistry.dispatch()` 统一入口，参数校验失败/业务异常都返回结构化错误，不 crash
- `requires_approval=True` 标记高风险工具（如发邮件），为后续 Security 周留 hook

### 5.4 Streaming 的价值

| 指标 | 非流式 | 流式 |
|---|---|---|
| TTFT（首字延迟） | = 总耗时 | ≈ 100-500ms |
| 用户体验 | 等待→突然看到全部 | 逐字看到 |
| 客户端超时风险 | 高（长文本） | 低 |
| 实现复杂度 | 简单 | 中等（SSE 解析） |

**SSE 协议要点**：
```
data: {"choices":[{"delta":{"content":"你好"}}]}

data: {"choices":[{"delta":{"content":"，"}}]}

data: [DONE]
```

### 5.5 模型对比实验方法论

**变量**：模型 / 温度 / Prompt 风格
**指标**：
- 成功率（输出符合 schema）
- P50 延迟
- 平均 token 数（→成本）
- 稳定性（多次运行的一致性）

**结论（待真实 API 填充）**：
- 温度越低，结构化输出越稳定
- 五要素 Prompt 比简短 Prompt 成功率高
- 不同模型在"严格 schema"任务上差异巨大

## 六、本周问题清单

1. **Mock 路由顺序 bug**：工具结果回喂后还触发新的工具调用 → 把 `role=tool` 检查放到 tools 检查之前，避免无限循环。
2. **模型输出包裹 markdown 代码块**：`extract_json` 增加正则匹配 ```` ```json ... ``` ```` 和兜底 `{...}` 截取。
3. **日期字段格式**：模型可能返回 "2026年3月15日"，用 Pydantic 的 `@field_validator` 在 `mode="before"` 预处理。
4. **Tool 参数校验**：用 `get_type_hints` + `create_model` 动态生成 Pydantic 模型，省去手写 Schema。
5. **流式响应的 HTTP 层**：必须用 `httpx.Client.stream()` 上下文管理器，不能用 `.post()` 然后 `.json()`。

## 七、自测题

1. JSON Mode 和 JSON Schema 的区别是什么？为什么生产环境必须用后者？
2. Tool Calling 中，模型真的执行了函数吗？如果不是，那它做了什么？
3. 为什么 Tool 执行失败不应该 raise，而是返回结构化错误给模型？
4. 如果用户输入让模型陷入"无限调工具"的循环，应该怎么防护？
5. Streaming 的 TTFT 和总耗时分别反映什么？哪个对用户体验影响更大？
6. 对比实验中为什么要固定 temperature 和 Prompt 风格？不固定会怎样？
7. 如果让你给「合同提取」再加一个"风险等级评分"字段（1-5 整数），Schema 怎么写？怎么校验？
8. `requires_approval=True` 的工具在真正执行前应该走什么流程？

## 八、验收自查

- [x] 能用 Pydantic 定义复杂 schema（含嵌套、可选字段、日期、枚举）
- [x] 能用 StructuredClient 让 LLM 稳定输出符合 schema 的 JSON
- [x] 能写出 Tool Calling 完整闭环（注册→调用→执行→回喂→回答）
- [x] 能理解 Streaming 与 TTFT 的意义，并用 httpx 实现 SSE 解析
- [x] 能设计并执行模型对比实验（多变量、多指标）
- [x] 22 个 pytest 全部通过

## 九、下周预告（week04）

- 多模态模型（图像/音频输入）
- Context Engineering 进阶（长上下文管理、记忆）
- 模型选型与成本优化实战
- 第1阶段综合验收
