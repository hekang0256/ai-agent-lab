# Week07 ｜ 基础 RAG：从文档到带引用的答案

> 《AI → 智能体》24周系统学习路线 · 第2阶段第3周（总第7周）
> 实验目标：不依赖重型 RAG 框架，从零串起 **文档解析 → Chunking → Embedding → Retrieval → Context → Generation → Citation**。
> 实测结果：Docker + pgvector 下 **30 passed（含独立测试表集成测试）**；Mock 为 **47/50**；真实 Kimi + 百炼 Embedding 严格自动口径为 **44/50**，Top-3 召回与无关问题拒答均为 100%。6 个失败已逐条人工复核。

## 一、本周交付

| 实验 | 内容 | 结果 |
|---|---|---|
| 实验7-A | JSON / Markdown / TXT / Word / PDF 统一解析；固定字符 + 句边界 + overlap 切块 | ✅ |
| 实验7-B | 查看检索结果如何变成 Context 与最终 Prompt，估算 Token | ✅ |
| 实验7-C | 完整问答、引用映射、类目过滤、无证据拒答 | ✅ |
| 实验7-D | 50 条问题集：42 条可回答 + 8 条应拒答 | ✅ Mock 47/50；真实模型 44/50 |
| **应用** | **企业制度 RAG 问答**，支持内存库离线复现、pgvector 持久化 | ✅ |
| 测试 | 文档、切块、RAG、评估、pgvector 集成 | ✅ 30 passed |

本周刻意只做基础 RAG；BM25、Hybrid Search、Rerank、Query Rewrite、Multi Query、Parent-Child Retrieval 留到第 8 周。

## 二、架构图

```text
JSON / Markdown / TXT / Word / PDF
                │
                ▼
       llm/document.py                  统一 SourceDocument
       文档边界、标题、类目、来源、页码元数据
                │
                ▼
       llm/chunking.py                  固定预算 + 句边界 + overlap
       document_id / chunk_id / start / end
                │
                ▼
       EmbeddingClient / MockEmbeddingClient
                │
                ▼
   ┌───────────────────────────────┐
   │ MemoryVectorStore（离线/测试） │  或  │ PgChunkStore（主项目） │
   └───────────────┬───────────────┘      └──────────┬──────────┘
                   └────────────── Retrieval ─────────┘
                                   │ Top-K + min_score + category
                                   ▼
                         带编号的 Context [1] [2] ...
                                   │
                                   ▼
                         LLMClient / MockLLMClient
                         仅用资料、受约束生成、逐句引用
                                   │
                                   ▼
                         Citation 白名单映射与校验
                         只接受本次检索结果中的编号
                                   │
                     ┌─────────────┴──────────────┐
                     ▼                            ▼
                有证据：回答 + 引用          无证据：直接拒答，不调 LLM
```

## 三、目录结构

```text
week07/
├── llm/
│   ├── document.py       # 多格式解析，统一 SourceDocument
│   ├── chunking.py       # 基础切块，稳定 Chunk ID 与原文位置
│   ├── embedding.py      # 真实 OpenAI 兼容 Embedding API + 离线 Mock
│   ├── vector_store.py   # 内存向量库，接口与数据库一致
│   ├── pg_store.py       # Chunk 级 pgvector 存储与检索
│   ├── client.py         # 真实 Chat Completions 客户端
│   ├── mock.py           # 抽取式离线生成器，不是通用 LLM
│   ├── rag.py            # 本周核心：RAGService / Citation / RAGResponse
│   └── evaluation.py     # 50 条问题集评估
├── apps/policy_qa.py     # 企业制度问答 CLI
├── experiments/exp01~exp04
├── data/
│   ├── docs/policies.md        # 14 份企业制度，二级标题保留文档边界
│   ├── eval_questions.json     # 50 条验收问题
│   ├── chunking_demo.json      # 切块实验结果
│   ├── context_prompt_demo.json# 实际 Prompt
│   ├── rag_demo.json           # 问答与引用结果
│   ├── evaluation_results.json # 内存库 Mock 评估
│   ├── evaluation_results_pg.json # pgvector Mock 评估
│   ├── evaluation_results_real.json # 真实 Kimi + 百炼 + pgvector 原始评估
│   └── evaluation_real_manual_review.md # 真实失败案例人工复核
├── scripts/                 # 真实连接、RAG 冒烟与重建索引
├── db/init.sql
├── docker-compose.yml    # 独立端口 5434，不覆盖 week06 数据
└── tests/
```

## 四、运行

### 4.1 离线复现（无需 API Key、无需 Docker）

```powershell
cd D:\ai-agent-lab\week07
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:LLM_USE_MOCK="true"

# 完整问答：自动把 policies.md 建入内存库
.venv\Scripts\python.exe -m apps.policy_qa --store memory
.venv\Scripts\python.exe -m apps.policy_qa "出差打车费用怎么报销？" --store memory
.venv\Scripts\python.exe -m apps.policy_qa "报销发票抬头怎么开？" --store memory --category 财务
.venv\Scripts\python.exe -m apps.policy_qa "今天中午吃什么？" --store memory --json

# 四个实验与测试
.venv\Scripts\python.exe -m experiments.exp01_document_chunking
.venv\Scripts\python.exe -m experiments.exp02_context_prompt
.venv\Scripts\python.exe -m experiments.exp03_rag_qa
.venv\Scripts\python.exe -m experiments.exp04_evaluation
.venv\Scripts\python.exe -m pytest -q
```

### 4.2 pgvector 主项目模式

```powershell
docker compose up -d
$env:LLM_USE_MOCK="true"  # 或复制 .env.example 为 .env 并配置真实模型
.venv\Scripts\python.exe -m apps.policy_qa --store pg --rebuild
.venv\Scripts\python.exe -m apps.policy_qa "VPN 使用规定是什么？" --store pg
.venv\Scripts\python.exe -m experiments.exp04_evaluation --store pg
.venv\Scripts\python.exe -m pytest -q
```

本周实例为 `localhost:5434/ai_agent_lab_rag`，与 week06 的 5433 隔离。已实测容器健康、14 个 Chunk 入库、30 条测试全过；pgvector 评估报告为 `data/evaluation_results_pg.json`。week01 的 5 条任务/5 条事件与 week06 的 14 条制度记录也均已确认保留。

### 4.3 真实模型复跑（已验收）

本次实际组合：Kimi Code `kimi-for-coding` + 阿里云百炼 `text-embedding-v4`（1024 维）+ pgvector。

```powershell
# .env 已配置时：先检查连接与真实向量维度
.venv\Scripts\python.exe -m scripts.smoke_real_models
.venv\Scripts\python.exe -m scripts.smoke_real_rag

# 只重建 Embedding 索引，不额外调用生成模型
.venv\Scripts\python.exe -m scripts.reindex

# 50 条真实评估；每条写入 checkpoint，中断后加 --resume 续跑
.venv\Scripts\python.exe -m experiments.exp04_evaluation --store pg --output data/evaluation_results_real.json --resume
```

- Kimi API 开放平台与 Kimi Code 是两套独立服务。本次使用 Kimi Code Key，对应 `https://api.kimi.com/coding/v1`；`kimi-for-coding` 要求 `temperature=1`。
- 百炼 v3/v4 通过 `EMBEDDING_DIMENSIONS=1024` 与当前 `vector(1024)` Schema 对齐，并设置 `EMBEDDING_BATCH_SIZE=10`，避免超过接口单批上限。
- 换 Embedding 模型或维度时，必须重新生成全部向量；不要向同一列混写不同维度。
- MockEmbedding 用主题词块 + 字符哈希，MockLLM 只抽取 Top-1 证据。它们用于验证链路和 CI，**不代表真实模型的语义能力**。
- `.env` 和 API Key 不进入 Git；报告只保存模型名、答案和指标，不保存密钥。

## 五、核心概念

### 5.1 Retrieval 不等于 RAG

week06 返回的是“相关文本列表”；本周增加了两步：

1. 把检索结果组织为带编号的 Context，约束模型只能根据证据回答。
2. 把答案中的 `[n]` 映射回本次真实检索的 Chunk，形成可核验 Citation。

```text
搜索系统：query → documents
RAG 系统：query → evidence → grounded answer → citations
```

### 5.2 Chunk 是索引与引用的基本单位

每块保留：`document_id`、`chunk_id`、标题、类目、来源、原文 `start/end`。这样既能回答“来自哪份文档”，也能定位到具体片段。

- `chunk_size` 太小：语义被切断，需要更多检索与拼接。
- `chunk_size` 太大：无关内容稀释向量，也浪费上下文 Token。
- `overlap` 缓解边界断句，但会增加重复与存储，不是越大越好。
- 换 Embedding 模型或维度时，需要重新计算全部 Chunk 向量。

### 5.3 两级拒答

1. **检索层拒答**：没有结果达到 `min_score`，直接返回“根据现有资料无法回答。”，不调用 LLM，节省成本并避免硬答。
2. **生成层拒答**：有召回但证据内容不足，模型按 Prompt 返回同一句拒答；响应记录 `refusal_reason=insufficient_context`。

阈值不是事实保证：高分只表示向量相似，不表示文档一定包含答案。第 8 周继续用更系统的检索与评估手段改进。

### 5.4 Citation 不是模型说了算

`RAGService` 只接受 `[1]..[N]` 范围内的编号，并映射到本次检索结果：

- 无编号 → `citation_status=missing`
- 越界编号（如 `[99]`）→ `invalid`，不会生成虚假来源
- 合法编号 → 返回标题、来源、Chunk ID、原文和检索分数

这能防止“模型编一个看似真实的来源”，但不能自动证明每句话都被引用内容支持；引用准确性与答案忠实性仍需评估。

## 六、50 条验收结果

### 6.1 Mock 基线

`data/evaluation_results.json`（内存库）与 `data/evaluation_results_pg.json`（pgvector）保存了 Mock 的逐条结果；两种存储的指标及 3 个失败案例完全一致。

| 指标 | Mock：内存库 / pgvector | 口径 |
|---|---:|---|
| Retrieval Recall | 95.24%（40/42） | 可答题 Top-3 是否召回目标文档 |
| Answer Keyword Accuracy | 92.86%（39/42） | 答案包含标注关键词且未拒答；只是离线代理指标 |
| Citation Accuracy | 92.86%（39/42） | 引用非空、属于目标文档且无非法编号 |
| Refusal Accuracy | 100%（8/8） | 无关问题正确拒答 |
| Overall Pass Rate | **94%（47/50）** | 可答题要求召回、答案、引用同时通过；不可答题要求拒答 |

保留的 3 个失败案例：

1. **Q06 阈值误拒答**：「团队聚餐超标部分谁承担？」没有显式说“报销”，Mock 向量得分低于阈值。说明拒答阈值需要结合问题集调参。
2. **Q22 Top-1 偏移**：「密码多少天更换一次？」Top-3 已召回 P08，但 MockLLM 只抽取了排名第一的 P09。说明“召回成功”不等于“最终答案成功”。
3. **Q25 主题串扰**：「远程访问内网必须使用什么？」召回了 P06 远程办公而非 P09 VPN。说明简单主题向量会把近主题混淆。

这些失败不是需要藏起来的噪声，而是下周学习 Hybrid Search、Rerank 和更细评估的动机。

### 6.2 真实 Kimi + 百炼 Embedding + pgvector

原始逐条结果见 `data/evaluation_results_real.json`，配置为 Top-3、min_score=0.35、`text-embedding-v4` 1024 维、`kimi-for-coding`。

| 指标 | Mock pgvector | 真实模型 | 观察 |
|---|---:|---:|---|
| Retrieval Recall | 95.24% | **100%（42/42）** | 真实 Embedding 让所有可答题目标文档进入 Top-3 |
| Answer Keyword Accuracy | 92.86% | **95.24%（40/42）** | Q22 是“90天/九十天”归一化问题，Q39 是真实误拒答 |
| Citation Accuracy | 92.86% | **88.10%（37/42）** | 严格口径不允许目标集合之外的额外引用 |
| Refusal Accuracy | 100% | **100%（8/8）** | 8 个无关问题全部拒答 |
| Overall Pass Rate | 94% | **88%（44/50）** | 原始自动结果，不做赛后修改 |

真实评估有 6 个原始失败。人工逐条复核见 `data/evaluation_real_manual_review.md`：

- Q01、Q03、Q21、Q28：答案正确，额外引用的文档也支持相关陈述；当前“引用集合必须完全属于 expected_document_ids”的口径过严。
- Q22：“90天”语义正确，但未精确命中标注关键词“九十天”。
- **Q39：真正的生成层误拒答**。P13 已进入 Top-3，原文明确“纪要并归档保存”，模型仍回答无法回答。
- 因此，本次样本人工复核后的语义答案为 49/50；这只是对本次报告的人工解释，**不替代 44/50 的原始自动分数，也不证明重复运行稳定**。

关键结论：真实模型的检索明显优于 Mock，但整体 RAG 质量不能只看召回；引用评估需要判断“陈述是否被证据支持”，关键词评估需要数字归一化，生成层仍可能漏用已召回证据。

## 七、本周问题清单

1. **解析时丢掉文档边界**：若把多个制度拼成一整段再切，引用只能指向整个文件。本周 Markdown/Word 用二级标题 `ID | 类目 | 标题` 先拆文档，再切块。
2. **overlap 可能导致死循环**：必须满足 `0 <= overlap < chunk_size`，并保证下一轮 start 严格前进。
3. **引用编号越界**：不能直接信任模型输出；引用只能从本次检索结果白名单映射。
4. **无检索结果仍调用模型**：既增加成本又诱导幻觉；本周在 Retrieval 后提前返回。
5. **Mock 指标被误当真实效果**：Mock 是确定性教学替身；答案关键词也只是可复现代理，不等于人工语义评分。
6. **真实向量维度不一致**：`vector(1024)` 不能接受 256 维向量。最终统一为百炼 v4 的 1024 维输出，并重新生成 14 个 Chunk；换模型必须检查真实维度。
7. **Kimi 产品/端点混淆**：Kimi Code Key 在开放平台端点返回 401；改用 Coding 端点后又发现模型仅允许 temperature=1。鉴权、模型 ID 和参数约束必须分别验证。
8. **真实批量接口限制**：百炼 v3/v4 单批最多 10 条，14 个 Chunk 需要分批 Embedding 并保持返回顺序。
9. **评估中断与重复费用**：首次真实评估遇临时 HTTP 错误后丢失内存中的进度；现已增加逐条 checkpoint、`--resume`、重试状态日志与调用间隔。
10. **测试污染主表**：早期 pgvector 测试 teardown 会清空主表；现使用独立 `rag_chunks_test`，测试结束后删除测试表，不再触碰主索引。
11. **自动指标不等于语义结论**：真实评估中“90天”及合法的额外引用被严格规则误判；保留原始分数，同时增加人工复核记录。

## 八、自测题

1. RAG 相比直接把问题发给 LLM，多了哪些可观测、可测试的环节？
2. 为什么 Chunk 需要保留 `document_id`、原文位置和来源？
3. `chunk_size` 与 `overlap` 分别怎样影响召回、上下文冗余和成本？
4. 为什么“Top-K 召回了正确文档”仍可能得到错误答案？
5. 检索层拒答和生成层拒答有什么区别？各有什么误判风险？
6. 为什么不能直接相信模型输出的来源 URL 或 `[99]`？
7. 答案带合法引用，是否就证明答案一定忠实于原文？还需要什么评估？
8. 为什么换 Embedding 模型后必须重建向量，而不仅是更新配置？
9. 本周哪三个失败案例适合用第 8 周的哪种技术继续排查？

## 九、验收自查

- [x] 能解析 JSON / Markdown / TXT / Word / 文本型 PDF，并统一文档结构
- [x] 能实现带 overlap 的基础 Chunking，保留来源与原文位置
- [x] 能完成 Embedding → Top-K → Context → Generation 的完整闭环
- [x] 答案引用可追溯到实际 Chunk，非法引用会被标记
- [x] 无相关检索结果时不调用 LLM，直接拒答
- [x] 建立 50 条问题集并保存逐条评估结果
- [x] 30 条测试全部通过，包含真实 pgvector 写入、检索、过滤与问答集成测试
- [x] 50 条问题集已在内存库和真实 pgvector 上分别复跑，结果一致
- [x] 真实 Kimi + 百炼 `text-embedding-v4` + pgvector 已完成 50 条复跑，原始严格结果 44/50
- [x] 6 个真实失败案例已全部人工复核并单独记录；确认 Q39 为真实生成层误拒答
- [x] 当前主表保留 14 个 1024 维真实 Chunk，数据库测试使用独立表，不再清空主索引

## 十、下周预告（week08）

- BM25 + 向量 Hybrid Search
- Rerank，以及 Top-K 与 Rerank 候选数的关系
- Query Rewrite / Multi Query / Parent-Child Retrieval
- 更细的 RAG Evaluation：检索、忠实性、引用支持度分别判断
