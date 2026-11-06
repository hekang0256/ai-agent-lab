# Week04 ｜ 多模态 + Context Engineering + 模型选型与成本（第1阶段收官）

> 《AI → 智能体》24周系统学习路线 · 第1阶段第3周（总第4周）
> 实验目标：让 LLM 处理图像/音频输入；掌握长上下文管理与记忆分层；建立「按需选模型」的成本意识；把 week01~04 组装成可用的企业助手。
> 验收标准：能构造多模态消息并完成发票提取；能解释三种上下文策略的取舍并量化对比；能实现跨会话记忆；能用路由器按约束选模型并算出节省比例；61 个 pytest 全过。

## 一、本周做了什么

| 实验 | 内容 | 结果 |
|------|------|------|
| 实验4-A | 多模态消息（图像 base64/URL、音频 input_audio） | ✅ |
| 实验4-B | 上下文策略对比：none / sliding / summary（含遗忘演示） | ✅ |
| 实验4-C | 长期记忆：写入 → 持久化 → 跨会话回忆 → 遗忘 | ✅ |
| 实验4-D | 模型路由：硬约束 / 软偏好（cost/quality/latency）/ 预算 | ✅ |
| **应用1** | **发票图片提取器**（多模态 + Pydantic 强约束） | ✅ |
| **应用2** | **成本优化仿真**：8 个企业任务路由，比旗舰基线**节省 91.1%** | ✅ |
| **应用3** | **第1阶段综合验收**：企业助手 CLI（对话+记忆+工具+成本） | ✅ |
| 测试 | pytest 61 条 | ✅ 61 passed |

## 二、架构图

```text
┌──────────────────────────────────────────────────────────────────┐
│ apps/invoice_ocr   apps/cost_optimizer   apps/stage1_assistant   │
│ experiments/exp01 ~ exp04                                        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
┌───────▼────────┐  ┌────────▼─────────┐  ┌────────▼─────────┐
│  ChatSession   │  │  ModelRouter     │  │ StructuredClient │
│  (多轮对话编排) │  │  (硬约束+软偏好) │  │ (多模态+Pydantic)│
│  · 工具闭环     │  │  · 预算过滤      │  │  · 失败回喂重试   │
│  · 记忆指令     │  │  · 决策可解释    │  └────────┬─────────┘
└───┬───┬───┬────┘  └────────┬─────────┘           │
    │   │   │                │                     │
┌───▼┐ ┌▼────────────┐ ┌─────▼──────┐              │
│Ctx │ │ MemoryStore │ │ CostTracker│              │
│Mgr │ │ (JSON 持久化)│ │ (成本记账) │              │
│裁剪│ │ 画像注入system│ │ by_model  │              │
└───┬┘ └─────────────┘ └────────────┘              │
    │        ┌──────────────────────────┐          │
    └───────▶│  LLMClient (httpx)       │◀─────────┘
             │  content parts 多模态     │
             │  tools / stream / 重试    │
             └──────────┬───────────────┘
              ┌─────────┴─────────┐
              │                   │
     ┌────────▼────────┐  ┌───────▼──────────┐
     │ 真实 LLM API     │  │  MockLLMClient   │
     │ (视觉模型需换    │  │  · token 真实估算 │
     │  qwen-vl/gpt-4o)│  │  · 发票/摘要/记忆 │
     └─────────────────┘  └──────────────────┘

支撑模块：tokenizer.py（离线 token 估算，所有裁剪与成本都建立在它之上）
```

## 三、目录结构

```text
week04/
├── llm/
│   ├── client.py           # 多模态增强：content parts / image_part / audio_part
│   ├── tokenizer.py        # 离线 token 估算（CJK≈1/字，英文≈4字符/token）
│   ├── context.py          # ContextManager：none/sliding/summary + 摘要预算截断
│   ├── memory.py           # MemoryStore（JSON 持久化）+ 记忆指令解析
│   ├── pricing.py          # 价格表（教学示例价）+ estimate_cost + CostTracker
│   ├── router.py           # ModelRouter：硬约束过滤 + 软偏好排序 + 预算
│   ├── structured.py       # StructuredClient：支持 images/audio 输入
│   ├── tools.py            # Tool 装饰器 + ToolRegistry（同 week03）
│   ├── session.py          # ChatSession：对话+记忆+工具+成本 的编排核心
│   └── mock.py             # Mock：发票/摘要/记忆问答/工具，token 真实估算
├── experiments/
│   ├── exp01_multimodal.py
│   ├── exp02_context_strategies.py
│   ├── exp03_memory.py
│   └── exp04_model_router.py
├── apps/
│   ├── invoice_ocr.py        # 应用1：发票图片 → JSON
│   ├── cost_optimizer.py     # 应用2：路由成本仿真（节省 91.1%）
│   └── stage1_assistant.py   # 应用3：第1阶段综合验收 CLI
├── scripts/
│   └── make_samples.py       # 纯标准库生成 PNG 发票（手写 5x7 位图字体）+ WAV
├── data/
│   ├── images/               # 2 张发票样例（可直接用于真实视觉模型）
│   ├── audio/note.wav
│   ├── tasks.json            # 8 个企业任务（成本仿真输入）
│   ├── invoice_results.json  # 提取结果
│   └── cost_report.md        # 成本报告
├── tests/                    # 61 个 pytest
├── .env.example
├── requirements.txt
└── README.md
```

## 四、如何运行

```bash
# 1. 准备环境（依赖与 week03 相同，无新增第三方包）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 离线模式：设置 LLM_USE_MOCK=true；真实模式：复制 .env.example 为 .env 填 key
#    图像提取需把 LLM_MODEL 换成视觉模型（qwen-vl-plus / gpt-4o-mini 等）

# 3. 生成多模态样本（首次）
python scripts\make_samples.py

# 4. 跑实验
python -m experiments.exp01_multimodal
python -m experiments.exp02_context_strategies
python -m experiments.exp03_memory
python -m experiments.exp04_model_router

# 5. 跑应用
python -m apps.invoice_ocr                          # 批量发票提取
python -m apps.cost_optimizer                       # 成本仿真（--prefer quality / --budget 0.5）
python -m apps.stage1_assistant --demo              # 综合验收演示
python -m apps.stage1_assistant                     # 交互模式（/cost /memory /reset /quit）

# 6. 跑测试
pytest
```

## 五、本周学习要点

### 5.1 多模态不是新 API，而是消息结构变了

```python
# 纯文本：content 是 string
{"role": "user", "content": "你好"}
# 多模态：content 是 parts 数组，文本/图像/音频可任意组合
{"role": "user", "content": [
    {"type": "text", "text": "提取这张发票"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
]}
```

**工程取舍**：
- **base64 data URI**：不依赖图床、随请求直达，但 payload 膨胀约 33%，且**千万不要整个打进日志**（本仓库 `show_message` 只打印长度）
- **URL 引用**：payload 小，但要求模型服务能访问该 URL（内网图片/带鉴权的图片常常不可达）
- 图像 token 不菲（本仓库按 low=85 / high≈1024 估算），批量图片任务要先算成本
- 音频字段各家差异大（input_audio / audio_url / 独立 ASR），以厂商文档为准

### 5.2 上下文管理：裁剪的是「发给模型的内容」，不是历史本身

本周实验数据（14 轮对话，预算 1000 tokens）：

| 策略 | 输入 tokens | 遗忘关键信息？ | 说明 |
|---|---:|---|---|
| none | 1424 | — | 超窗口，真实 API 直接报 `context_length_exceeded` |
| sliding | 922 | ❌ 遗忘第1轮的项目代号 | 简单可靠，适合短平快对话 |
| summary | 609 | ✅ 代号保留在摘要中 | 多一次摘要调用（有成本），适合长对话 |

**三个关键设计**：
1. **按轮裁剪**：user 与其后的 assistant/tool 消息是一个整体，拆开 tool_calls 与 tool 结果会让 API 报错
2. **摘要也占预算**：摘要文本最多用预算的 40%，超出按行截断——这是本周自查修掉的真实 bug（20 轮摘要直接把 250 的预算撑爆到 372）
3. **本地全量保留 history**：裁剪只影响发送内容，用户随时能回看完整对话

### 5.3 记忆分层：短期靠 history，长期靠注入 system

| 层 | 载体 | 生命周期 | 风险 |
|---|---|---|---|
| 短期记忆 | 会话消息历史 | 会话结束即消 | 超窗口被裁剪（交给 ContextManager） |
| 长期记忆 | MemoryStore（JSON 键值） | 持久化、跨会话 | 隐私与遗忘权（必须支持「忘掉…」） |

长期记忆**以画像文本注入 system**，而不是塞进对话历史：不会被滑动窗口裁掉，也不随对话变长线性膨胀。
下周的 RAG 解决的是另一类问题：记忆是「关于用户的事实」，RAG 是「关于世界的知识」。

### 5.4 模型选型：够用且最便宜，而不是一律旗舰

路由 = **硬约束过滤**（视觉/工具/窗口/预算）→ **软偏好排序**（cost/quality/latency）。
本周 8 个企业任务的仿真结果：路由方案 $0.0272 vs 全旗舰 $0.3053，**节省 91.1%**。

但必须记住两条红线：
- **便宜 ≠ 够用**：`complexity=high` 的任务直接排除 economy 模型，质量风险不能用省钱来换
- **成本可观测**：所有调用（包括摘要这种「隐藏调用」）都进 CostTracker，否则月底账单会教你做人
- 路由只保证「约束满足」，选型的**质量验证**需要评估集——那是 week13 Evaluation 的主题

### 5.5 第1阶段能力地图（week01 → week04）

```text
week01  工程底座：Python / API / Docker / FastAPI / SQL / Git / pytest
week02  LLM 认知：Token / Context / Prompt 五要素 / API 调用
week03  可靠交互：Structured Output / Tool Calling / Streaming
week04  生产思维：多模态 / 上下文与记忆 / 成本与选型
        └─ 交付物：ChatSession = 对话 + 记忆 + 工具 + 上下文 + 成本 的最小可用助手
```

## 六、本周问题清单

1. **摘要自身超预算**：旧消息太多时，启发式摘要（15 行要点）比预算还长 → 给摘要设 40% 预算上限并按行截断，对应测试 `test_summary_respects_budget_when_possible`。
2. **Mock 固定 token 让成本实验失真**：week03 的 Mock 永远返回 `prompt_tokens=60`，无法反映裁剪效果 → week04 Mock 改用 tokenizer 真实估算，成本/上下文实验才有意义。
3. **裁剪拆散 tool 轮次**：如果按单条消息滑动，可能留下 role=tool 却丢了 assistant 的 tool_calls → 裁剪单位改为「轮」。
4. **多模态消息处处假设 content 是 str**：新增 `ChatMessage.text` 属性统一取文本，Mock/tokenizer/session 全部走它，避免 `str | list` 类型分支散落各处。
5. **tiktoken 离线不可用**：首次运行要联网下载词表 → 自写启发式估算（偏保守），生产可替换为官方 tokenizer，接口不变。
6. **docstring 里的 Windows 路径触发 DeprecationWarning**：`data\images\...` 的 `\i` 是非法转义 → 模块 docstring 改用 `r"""`。

## 七、自测题

1. 图像用 base64 内嵌和 URL 引用各有什么代价？内网图片为什么通常只能走 base64？
2. 为什么上下文裁剪要以「轮」为单位？拆散 tool_calls 和 tool 结果会发生什么？
3. sliding 和 summary 各适合什么场景？摘要策略引入了哪些新成本和新风险？
4. 长期记忆为什么注入 system 而不是追加到对话历史？「忘掉我的城市」为什么是必须实现的功能？
5. 成本公式是什么？一个 10k 输入 / 2k 输出的任务，用 gpt-4o（$2.5/$10 per 1M）单次多少钱？
6. 路由器为什么把「复杂度门槛」放在硬约束里？能不能为了省钱给合同审查也选 economy 模型？
7. ChatSession 里摘要调用也要记账，为什么？如果漏记会怎样？
8. 真实项目里价格表应该硬编码在代码里吗？应该怎么管理？

## 八、验收自查

- [x] 能构造图像/音频 content parts，本地文件自动转 data URI
- [x] 能用 StructuredClient 完成「图片 → Pydantic JSON」的提取闭环
- [x] 能实现三种上下文策略并量化对比（token / 遗忘）
- [x] 能实现跨会话长期记忆，并支持显式「记住/忘掉」
- [x] 能实现带预算约束、决策可解释的模型路由器
- [x] 能把对话+记忆+工具+上下文+成本组装成 ChatSession
- [x] 61 个 pytest 全部通过

## 九、下周预告（week05 · 进入第2阶段）

- 文档解析（PDF/Word/Markdown）与 Chunking 策略
- Embedding 与向量检索（本周记忆管「用户事实」，RAG 管「世界知识」）
- RAG 最小闭环：检索 → 增强 → 生成 → 引用
