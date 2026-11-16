# Week06 ｜ Vector Search：pgvector / Top-K / Metadata Filter（第2阶段第2周）

> 《AI → 智能体》24周系统学习路线 · 第2阶段第2周（总第6周）
> 实验目标：把 week05 的内存向量库升级为 PostgreSQL + pgvector；理解距离操作符、Top-K 下推、Metadata Filter、向量索引与执行计划、索引维护。
> 验收标准：能用 SQL 完成向量检索与元数据过滤；能读懂 EXPLAIN 的 Seq Scan / Index Scan；能说明索引的价值与局限；18 个 pytest（真实数据库）全过。

## 一、本周做了什么

| 实验 | 内容 | 结果 |
|------|------|------|
| 实验6-A | pgvector 基础：扩展、三种距离操作符对照（L2 / 负内积 / cosine） | ✅ |
| 实验6-B | Top-K + Metadata Filter（全库 vs 财务 vs IT，阈值拒答） | ✅ |
| 实验6-C | 索引对照：Seq Scan → HNSW Index Scan、耗时、内存库一致性 | ✅ |
| 实验6-D | 索引维护：upsert 更新 / delete 删除 / 恢复，检索随之变化 | ✅ |
| **应用** | **企业制度语义搜索 pgvector 版**（week05 应用的数据库升级） | ✅ |
| 测试 | pytest 18 条（真实 pgvector，DB 不可用自动跳过） | ✅ 18 passed |

## 二、架构图

```text
┌────────────────────────────────────────────────────────────┐
│ apps/policy_search_pg.py      experiments/exp01 ~ exp04     │
└──────────────────────────┬─────────────────────────────────┘
                           │
                ┌──────────▼──────────┐
                │   PgPolicySearch    │
                │ 建索引/检索/拒答     │
                └──────────┬──────────┘
                           │ embed(query)
        ┌──────────────────┼───────────────────┐
        │                  │                   │
┌───────▼────────┐  ┌──────▼────────┐  ┌───────▼────────┐
│ EmbeddingClient│  │MockEmbedding  │  │ VectorStore    │
│ / Mock（同w05） │  │（离线确定性）  │  │（内存版·对照组）│
└────────────────┘  └───────────────┘  └────────────────┘
                           │
              ┌────────────▼─────────────┐
              │      PgVectorStore       │
              │ upsert/search/delete/    │
              │ explain/create_index     │
              └────────────┬─────────────┘
                           │ psycopg（参数化 SQL）
              ┌────────────▼─────────────┐
              │ PostgreSQL 16 + pgvector │  docker: ai_agent_lab_pgvector
              │ policy_docs              │  端口 5433（与 week01 独立）
              │  · embedding vector(256) │
              │  · category 普通索引      │
              │  · HNSW/IVFFlat 向量索引  │
              └──────────────────────────┘
```

## 三、目录结构

```text
week06/
├── llm/
│   ├── pg_store.py         # PgVectorStore（本周新代码：检索/过滤/索引/执行计划）
│   ├── embedding.py        # 同 week05（真实 API + Mock）
│   ├── similarity.py       # 同 week05
│   ├── vector_store.py     # 内存向量库（对照组）
│   └── tokenizer.py
├── db/init.sql             # CREATE EXTENSION vector + policy_docs 表
├── docker-compose.yml      # pgvector/pgvector:pg16（端口 5433）
├── apps/policy_search_pg.py
├── experiments/
│   ├── exp01_pgvector_basics.py
│   ├── exp02_topk_and_filter.py
│   ├── exp03_index_benchmark.py
│   └── exp04_index_maintenance.py
├── data/
│   ├── docs/policies.json      # 14 条制度语料（同 week05）
│   ├── pg_search_demo.json     # 演示查询结果（含过滤）
│   └── benchmark_results.json  # 索引对照数据（执行计划/耗时/一致性）
├── tests/                    # 18 个 pytest（真实 DB）
└── README.md
```

## 四、如何运行

```bash
# 1. 启动 pgvector（首次自动执行 init.sql）
docker compose up -d

# 2. 准备环境
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. 建索引 + 演示（离线设置 LLM_USE_MOCK=true）
python -m apps.policy_search_pg --rebuild
python -m apps.policy_search_pg "报销流程" --category 财务

# 4. 实验
python -m experiments.exp01_pgvector_basics
python -m experiments.exp02_topk_and_filter
python -m experiments.exp03_index_benchmark
python -m experiments.exp04_index_maintenance

# 5. 测试（需要 DB 运行；无 DB 时自动 skip）
pytest
```

## 五、本周学习要点

### 5.1 pgvector 三种距离操作符（实测）

对「出差打车费用怎么报销」的检索（Top-1 均为 P01《差旅报销制度》）：

| 操作符 | 含义 | 实测 distance | 适用 |
|---|---|---:|---|
| `<->` | L2 欧氏距离 | 0.126351 | 欧氏语义/聚类 |
| `<#>` | 负内积 | -0.992018 | 归一化后 ≡ 余弦排序 |
| `<=>` | cosine distance | 0.007982 | **语义检索默认** |

要点：cosine distance = 1 - 余弦相似度，**越小越相似**；`ORDER BY embedding <=> q LIMIT k` 就是 Top-K。

### 5.2 Metadata Filter：过滤与检索是一条 SQL

```sql
SELECT id, title, 1 - (embedding <=> %s::vector) AS score
FROM policy_docs
WHERE category = %s                       -- 元数据过滤
  AND embedding <=> %s::vector <= 0.65    -- min_score = 0.35（拒答阈值）
ORDER BY embedding <=> %s::vector         -- 语义排序
LIMIT 3;                                  -- Top-K
```

实测「报销需要哪些材料」：
- 全库：P02(0.962)、P01(0.901)，第三名已是其它类目的 0.055
- 仅财务：只剩两条财务制度（干净）
- 仅 IT：最高 0.055 → 低于阈值 0.35 → **拒答**（类目不符就不该硬答）

**过滤下推**（WHERE 在库内）优于「检索后过滤」（拿回 100 条再筛）：后者既浪费又可能把正确结果挤出 Top-K。

### 5.3 索引与执行计划：Seq Scan → Index Scan

实测（`data/benchmark_results.json`）：

```text
无索引：  Limit -> Sort -> Seq Scan on policy_docs
建 HNSW： Limit -> Index Scan using idx_policy_docs_embedding on policy_docs
```

- 建索引 DDL：`USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`
- 小表上优化器可能仍选 Seq Scan（索引不划算），可 `SET enable_seqscan=off` 强制展示索引路径（教学技巧）
- 本周 14 条数据耗时 24.88ms vs 29.47ms（含连接开销），**索引没有优势**——索引的价值在十万级以上（暴力 O(N·D) → ANN 近似对数级）
- **索引只影响性能，不影响结果**：pg 与 week05 内存库 Top-3 完全一致（P01/P02/P14），测试已固化

HNSW vs IVFFlat：

| 索引 | 特点 | 参数 |
|---|---|---|
| HNSW | 图索引，召回高、查询快，建索引慢、占内存 | m / ef_construction（建），ef_search（查） |
| IVFFlat | 倒排聚类，建索引快，召回依赖 lists/probes | lists（建），ivfflat.probes（查） |

### 5.4 索引维护：upsert / delete，而非全量重建

实验6-D 实测：upsert 修改 P02 后检索分数即时变化；delete P13 后「会议室」查询最高分仅 0.057。
- 文档更新/删除 → 按主键 upsert/delete，索引由数据库自动维护
- **只有换 embedding 模型或维度才需要全量重建**（vector(256) 维度是表级约束）

### 5.5 与 week05 的关系

week05 回答「向量怎么算相似度」；本周回答「向量怎么在真实数据库里存、查、过滤、加速」。
检索语义完全一致（同一 Mock 下结果逐条对齐），变化的是：**存储下推、过滤下推、排序下推、可观测（EXPLAIN）、可维护（upsert/delete）**。
下周 week07 把检索结果交给 LLM 生成答案并附引用，才是完整 RAG。

## 六、本周问题清单

1. **排序断言先验错误**：以为 D1（差旅）一定比 D2（日常报销）更贴近报销查询，实测内存 cosine 是 D2(0.9918) > D1(0.9838)——D1 还命中 travel 主题，方向被摊薄。**先算实际分数再写断言**，数据库只是如实返回。
2. **DDL 不能参数化**：CREATE INDEX 的 m/lists、SET 等不能用 `%s` 占位 → 一律 int() 强转拼接，表名正则白名单，杜绝注入。
3. **小表的 Seq Scan**：14 行数据优化器认为索引不划算，属正常；教学演示用 `enable_seqscan=off`。
4. **短连接开销**：每次操作新建连接约几 ms~20ms，压过检索本身 → 本周 benchmark 数字含连接开销，生产应上连接池。
5. **psycopg 无 pgvector 类型适配**：不装额外包，统一 `'[x,y,...]'::vector` 文本传参，简单可靠。

## 七、自测题

1. cosine distance 和余弦相似度是什么关系？为什么 Top-K 是 `ORDER BY distance ASC`？
2. 「检索后过滤」和「过滤下推到 SQL」有什么本质差别？各有什么坑？
3. 为什么小表建了索引优化器也不用？这和索引有没有失效是一回事吗？
4. HNSW 的 ef_search 调大和调小分别影响什么？
5. 为什么「索引不影响检索正确性」可以写成测试？IVFFlat 也严格成立吗？（提示：近似索引）
6. 文档更新为什么要连向量一起 upsert？只改 text 不改 embedding 会怎样？
7. 换 embedding 模型（256 维 → 1024 维）需要做什么？为什么不能直接 upsert？
8. 生产环境高并发下，本周的「每次操作新建连接」有什么风险？怎么改？

## 八、验收自查

- [x] 能用 docker compose 启动 pgvector 并初始化扩展与表
- [x] 能用 SQL 完成向量 Top-K、min_score 拒答、category 过滤
- [x] 能创建 HNSW / IVFFlat 索引并用 EXPLAIN 验证执行计划
- [x] 能验证 pg 与内存库检索结果一致
- [x] 能用 upsert/delete 维护索引文档
- [x] 18 个 pytest 全部通过（真实数据库）

## 九、下周预告（week07）

- RAG 完整闭环：文档解析 → Chunking → Embedding → Retrieval → Context → Generation → Citation
- 把本周 pgvector 检索接入 LLM，答案附引用来源
- 建立 50 条问题集，评估召回、答案准确性、引用正确性、无答案拒答（第2阶段验收）
