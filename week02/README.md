# Week02 ｜ LLM 基础认知 + API 调用 + Prompt 工程

> 《AI → 智能体》24周系统学习路线 · 第1阶段第1周（总第2周）
> 实验目标：建立 LLM 统一调用层；理解 Token/Context Window/Messages；掌握 Prompt 五要素与 Few-shot；交付文本工具箱 + 项目日报生成器。
> 验收标准：能独立调用 OpenAI 兼容 API、用 Prompt 解决具体任务、判断 token 是否超窗。

## 一、本周做了什么

| 实验 | 内容 | 结果 |
|------|------|------|
| 实验0-A | Token / Context Window / Messages 直观理解 | ✅ |
| 实验0-B | Prompt 五要素（角色/任务/上下文/约束/输出格式）对比 | ✅ |
| 实验0-C | Zero-shot vs Few-shot 情感分类对比 | ✅ |
| **实验1** | **AI 文本处理工具箱**（摘要/翻译/润色/关键词/情感） | ✅ |
| **实验3** | **AI 结构化项目日报生成器**（Few-shot + 后处理校验） | ✅ |
| 测试 | pytest 24 条 | ✅ 24 passed |

## 二、架构图（V1 LLM 调用层雏形）

```text
┌─────────────────────────────────────────────────────────┐
│  apps/daily_report.py       experiments/exp04_text_toolkit │
│  （业务层：具体任务 Prompt）                                │
└────────────────────────┬────────────────────────────────┘
                         │ ChatMessage / ChatResponse
┌────────────────────────▼────────────────────────────────┐
│  llm.client.LLMClient  (OpenAI 兼容, httpx)              │
│  · 超时 / 指数退避重试 / Token 统计 / 延迟统计            │
│  · get_client() 工厂：根据 LLM_USE_MOCK 切换真实 / Mock   │
└────┬───────────────────────────────────┬────────────────┘
     │ HTTPS /chat/completions            │ 离线
┌────▼──────────────┐            ┌────────▼─────────────┐
│ DeepSeek / Kimi / │            │  MockLLMClient       │
│ Qwen / OpenAI /   │            │  (规则路由, 不联网)   │
│ vLLM / Ollama     │            │  用于 CI & 离线演示   │
└───────────────────┘            └──────────────────────┘
```

关键设计：**只依赖 OpenAI 兼容协议，不绑定厂商 SDK**。后续 RAG / Agent / Multi-Agent 都通过这一层访问模型。

## 三、目录结构

```text
week02/
├── llm/                       # V1 LLM 统一调用层（贯穿24周复用）
│   ├── client.py              # OpenAI 兼容客户端（httpx + 重试 + Token 统计）
│   ├── mock.py                # MockLLMClient：离线/测试专用
│   └── tokens.py              # tiktoken + 启发式 token 估算
├── experiments/
│   ├── exp01_token_context.py # Token/Context Window 实验
│   ├── exp02_prompt_structure.py
│   ├── exp03_few_shot.py
│   └── exp04_text_toolkit.py  # 实验1：五合一文本工具
├── apps/
│   └── daily_report.py        # 实验3：项目日报生成器
├── data/
│   └── report_input.json      # 日报输入示例
├── tests/                     # 24 个 pytest
├── .env.example               # 配置模板（DeepSeek/Kimi/通义/Ollama 示例）
├── requirements.txt
└── README.md
```

## 四、如何运行

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 配置 API（三选一）
# A. 用真实 API：复制 .env.example 为 .env 并填入 LLM_API_KEY
# B. 用本地 Ollama：.env 中 LLM_BASE_URL=http://localhost:11434/v1
# C. 不联网演示：设置环境变量 LLM_USE_MOCK=true

# 3. 跑实验
python -m experiments.exp01_token_context
python -m experiments.exp02_prompt_structure
python -m experiments.exp03_few_shot
python -m experiments.exp04_text_toolkit summarize "一段长文本"
python -m apps.daily_report data\report_input.json

# 4. 跑测试（默认走 Mock，不消耗 token）
pytest
```

## 五、本周学习要点

### 5.1 Token 是什么？为什么必须自己算？

实验 0-A 实测数据（cl100k_base 编码）：

| 文本 | 字符数 | Token 数 | Token/字符 |
|---|---|---|---|
| `Hello, how are you?` | 19 | 6 | 0.32 |
| `你好，你怎么样？` | 8 | 9 | 1.13 |
| `人工智能正在改变世界` | 10 | 11 | 1.10 |
| `AI is changing the world` | 24 | 5 | 0.21 |

**结论**：中文 token 化效率远低于英文，同义内容中文 token 数是英文的 2 倍左右。8000 字中文需要 ≥32K 上下文窗口。

不算 token 的后果：被截断、被多扣费、被 API 报错。

### 5.2 Messages 元信息也占 token

每条 message 额外 4 token（role + 格式），整个对话收尾 2 token。
10 轮对话 ≈ 100+ token 的"隐藏开销"，长对话必须考虑。

### 5.3 Prompt 五要素

| 要素 | 位置 | 作用 |
|---|---|---|
| 角色 Role | system | 设定专业视角 |
| 任务 Task | user | 要做什么 |
| 上下文 Context | user | 输入数据 |
| 约束 Constraints | system | 不能做什么 |
| 输出格式 Format | system | 输出长什么样 |

### 5.4 Few-shot 解决什么问题？

Zero-shot 文字描述输出格式 → 模型可能不遵守；
Few-shot 给 2-3 个示例 → 模型"模仿"输出格式，稳定性显著提升。

代价：每次调用都多了示例的 token 开销。**3 个示例通常足够**。

### 5.5 为什么需要 Mock？

- CI/CD 不能依赖外部 API（不稳定、扣费）
- 没网也能开发
- 后续 Agent 阶段可以注入故障测试鲁棒性

**Mock 必须和真实客户端接口一致**，否则失去意义。

## 六、本周问题清单

1. **PowerShell 中文乱码**：设置 `$env:PYTHONIOENCODING='utf-8'` 后解决。
2. **Mock 路由规则不全**：初版只看 user 消息，但不同任务模板把关键词放在 system 或 user 不同位置 → 改为同时检查两个。
3. **日报输出章节缺失**：真实 LLM 可能偷懒只写部分章节 → 增加 `validate_report` 后处理校验，打印 warning 而不是直接失败（生产环境可触发重试）。
4. **tiktoken 在中文上的"感觉"**：发现 8000 字中文超出 8K 窗口，立刻理解了为什么企业 RAG 必须做 chunking。

## 七、自测题

1. 为什么中文 token 化效率比英文低？对 RAG 的 chunking 策略有什么影响？
2. `messages` 里 system/user/assistant 三种 role 的区别是什么？assistant 消息能用来干什么（提示：Few-shot）？
3. 为什么把「约束」放在 system 而不是 user？
4. 100 字的 Few-shot 示例 ≈ 多少 token？如果每次调用都带 3 个示例，调用 10000 次会多花多少钱？
5. 如果用户输入超出 Context Window，有哪些应对策略？（至少列 3 种）
6. 为什么 Mock 必须和真实客户端接口一致？Mock 能不能模拟"超时"和"限流"？
7. 温度（temperature）参数对输出有什么影响？情感分类任务应该用多少？
8. 如果让你给「合同信息提取」写 Prompt，五要素各写什么？

## 八、验收自查

- [x] 能解释 Token / Context Window / Messages 三者的关系
- [x] 能用 Python 调用 OpenAI 兼容 API（httpx 实现，含重试）
- [x] 能写出五要素齐全的 Prompt
- [x] 能用 Few-shot 让模型输出固定格式
- [x] 知道为什么需要 Mock，且能写出接口一致的 Mock
- [x] 24 个 pytest 全部通过

## 九、下周预告（week03）

- Structured Output（JSON Schema + Pydantic 校验）
- Function / Tool Calling（让模型调用真实函数）
- Streaming（SSE 流式响应）
- 实验2：合同信息 JSON 提取工具
- 实验4：模型对比实验（不同 prompt/温度/模型的准确性、速度、成本）
