# Week05 ｜ Embedding：向量、相似度与语义搜索（第2阶段第1周）

> 《AI → 智能体》24周系统学习路线 · 第2阶段第1周（总第5周）
> 实验目标：彻底理解 Embedding——向量、相似度、Embedding 模型、语义搜索；交付企业制度语义搜索 + FAQ 语义去重。
> 验收标准：能手算并解释余弦相似度；能调用 OpenAI 兼容 Embedding API（含 Mock）；能搭建「索引 → Top-K 检索 → 阈值拒答」的最小闭环；41 个 pytest 全过。

## 一、本周做了什么

| 实验 | 内容 | 结果 |
|------|------|------|
| 实验5-A | 向量与相似度直觉（余弦 vs 欧氏、归一化、向量算术、Top-K） | ✅ |
| 实验5-B | 语义相似度矩阵（8 句两两对比，同主题 0.94~0.97，跨主题≈0） | ✅ |
| 实验5-C | 语义搜索最小闭环（14 条制度索引 + 4 查询 + 拒答演示） | ✅ |
| 实验5-D | FAQ 去重 + 阈值敏感性（0.75/0.9 → 7 簇，0.99 → 10 簇） | ✅ |
| **应用1** | **企业制度语义搜索**（批量 Embedding + 向量库 + 阈值拒答） | ✅ |
| **应用2** | **FAQ 语义去重**（14 条 → 7 簇，客服知识库清洗） | ✅ |
| 测试 | pytest 41 条 | ✅ 41 passed |

## 二、架构图

```text
┌──────────────────────────────────────────────────────────────┐
│ apps/semantic_search.py        apps/faq_dedup.py              │
│ experiments/exp01 ~ exp04                                     │
└───────────────────────────┬──────────────────────────────────┘
                            │
          ┌─────────────────┼──────────────────┐
          │                 │                  │
┌─────────▼────────┐ ┌──────▼───────┐ ┌────────▼─────────┐
│  PolicySearch    │ │ dedup_faqs   │ │   实验脚本        │
│  建索引/检索/拒答 │ │ 贪心聚类去重  │ │                  │
└─────────┬────────┘ └──────┬───────┘ └────────┬─────────┘
          │                 │                  │
┌─────────▼─────────────────▼──────────────────▼─────────┐
│ VectorStore（内存向量库）                                │
│ upsert / search(Top-K+min_score) / delete / save / load │
│ 接口向数据库看齐 → week06 平替为 pgvector                │
└─────────────────────────┬──────────────────────────────┘
                          │ cosine_similarity / top_k
┌─────────────────────────▼──────────────────────────────┐
│ similarity.py（纯 Python 向量数学）                      │
│ dot / l2_distance / cosine / normalize / matrix / top_k │
└─────────────────────────┬──────────────────────────────┘
                          │
              ┌───────────┴────────────┐
              │                        │
   ┌──────────▼──────────┐  ┌──────────▼───────────┐
   │  EmbeddingClient    │  │  MockEmbeddingClient │
   │  POST /embeddings   │  │  主题词块 + 2-gram   │
   │  批量/重试/token    │  │  哈希 + 归一化       │
   │  (阿里/OpenAI/本地) │  │  确定性、离线、CI    │
   └─────────────────────┘  └──────────────────────┘
```

## 三、目录结构

```text
week05/
├── llm/
│   ├── similarity.py       # 向量数学：dot / l2 / cosine / normalize / top_k
│   ├── embedding.py        # EmbeddingClient（真实 API）+ MockEmbeddingClient
│   ├── vector_store.py     # 极简向量库：upsert / search / delete / JSON 持久化
│   └── tokenizer.py        # 离线 token 估算（供 Mock 统计）
├── experiments/
│   ├── exp01_vector_basics.py
│   ├── exp02_similarity_matrix.py
│   ├── exp03_semantic_search.py
│   └── exp04_faq_dedup.py
├── apps/
│   ├── semantic_search.py    # 应用1：制度语义搜索（含拒答）
│   └── faq_dedup.py          # 应用2：FAQ 语义去重
├── data/
│   ├── docs/policies.json    # 14 条企业制度语料
│   ├── faq_raw.json          # 14 条原始 FAQ（含 4 组近义）
│   ├── policy_index.json     # 向量索引（运行生成）
│   ├── search_demo.json      # 演示查询结果
│   └── faq_groups.json       # 去重分组结果
├── tests/                    # 41 个 pytest
├── .env.example
├── requirements.txt
└── README.md
```

## 四、如何运行

```bash
# 1. 准备环境（依赖与前几周相同，纯 Python 向量计算，无需 numpy）
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 离线模式：LLM_USE_MOCK=true；真实模式：复制 .env.example 为 .env
#    配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL
#    （阿里 text-embedding-v3 / OpenAI text-embedding-3-small / Ollama bge-m3）

# 3. 跑实验
python -m experiments.exp01_vector_basics
python -m experiments.exp02_similarity_matrix
python -m experiments.exp03_semantic_search
python -m experiments.exp04_faq_dedup

# 4. 跑应用
python -m apps.semantic_search --rebuild        # 建索引 + 演示 4 个查询
python -m apps.semantic_search "打车费怎么报销"  # 单次查询
python -m apps.faq_dedup --threshold 0.75       # FAQ 去重

# 5. 跑测试
pytest
```

## 五、本周学习要点

### 5.1 Embedding：把语义编码成空间方向

文本 → 向量（本周 Mock 256 维，真实模型常见 1024/1536/3072 维）。
**语义相近的文本，向量方向相近**——所以「出差打车费用怎么报销」能命中《差旅报销制度》，尽管两者字面几乎不重合。
向量算术（king - man + woman ≈ queen）说明：语义关系被编码成了空间中的偏移方向。

### 5.2 三种相似度/距离怎么选

| 度量 | 公式直觉 | 受长度影响？ | 适用 |
|---|---|---|---|
| 余弦相似度 | 只看夹角 | 否 | **语义搜索默认选择** |
| 点积 | 方向 × 长度 | 是 | 归一化后 ≡ 余弦（检索更快） |
| 欧氏距离 | 空间直线距离 | 是 | 聚类/异常检测等 |

**工程技巧**：入库前先归一化，检索时用点积代替余弦，省一次范数除法；pgvector 的 `<=>`（cosine）与 `<#>`（负内积）在归一化后等价。

### 5.3 语义搜索最小闭环与实测

```text
文档 → Embedding → 向量库
查询 → Embedding → 余弦 Top-K → min_score 过滤 → 结果 or 拒答
```

本周实测（Mock，14 条制度）：

| 查询 | Top-1 | 分数 | 判定 |
|---|---|---:|---|
| 出差打车费用怎么报销 | 《差旅报销制度》 | 0.992 | ✅ |
| 年假能休几天 | 《休假与考勤制度》 | 0.983 | ✅ |
| 服务器权限怎么申请 | 《VPN 与服务器使用》 | 0.936 | ✅ |
| 今天中午吃什么 | 最高分仅 0.225 | <0.35 | 🚫 拒答 |

**拒答（min_score 阈值）是检索系统的关键能力**：宁可说"没找到"，也不要把不相关的制度硬塞给用户（week07 接 LLM 后，这就是防幻觉的第一道闸）。

暴力 Top-K 是 O(N·D)：本周 14 条无所谓；十万级以上需要 ANN 索引（HNSW）——下周 pgvector 解决。

### 5.4 FAQ 去重：阈值是精度与召回的权衡

贪心聚类：每条 FAQ 与簇代表（首条）比较，相似度 ≥ 阈值则并入。
实测：阈值 0.75 / 0.9 都是 7 簇（组内相似度 >0.9）；阈值 0.99 拆成 10 簇（年假组、密码组被拆开）。
- 阈值低 → 误并风险（把不同问题合并）
- 阈值高 → 漏并风险（同义问题没合并）
- 簇代表取首条是启发式；更精细可用簇质心或两两比较

### 5.5 Mock 的设计与边界（重要）

MockEmbeddingClient = 主题词块加权（主导）+ 字符 2-gram 哈希（细粒度）+ 归一化。
它**确定、离线、有主题语义**，足以演示和测试检索逻辑；但它不是真语义模型：
- 「同主题 ≠ 同义」：加班费怎么算 / 加班能补休吗，在 Mock 看来很相似，真实语义模型能分得更细
- 测试断言的是**检索与聚类的代码逻辑**；真实检索质量必须用真实模型 + 评估集验证（第2阶段验收要求 50 条问题集，week07/08 完成）

切换到真实模型只需换 client（`get_embedding_client()`），业务代码零改动。

## 六、本周问题清单

1. **测试阈值算错**：`min_score=0.99` 想只留自身，却忘了 [0.9,0.1] 与 [1,0] 的余弦≈0.9939 也会通过 → 改 0.999。教训：写相似度断言前先算一遍实际分数。
2. **批量 Embedding 比逐条省 N-1 次网络往返**：建索引必须一次批量调用，且返回要按 `index` 排序对齐输入。
3. **零向量处理**：空文本/无信号文本会产生零向量，余弦公式除零 → 约定返回 0.0（无方向即无相似）。
4. **Mock 跨主题"串扰"**：无关查询最高分到 0.225（2-gram 哈希碰撞），阈值定 0.35 才拦得住 → 阈值不是拍脑袋，要看分数分布。
5. **向量库 id 冲突**：重复 add 同 id 选择 upsert（覆盖），与数据库语义对齐，为 week06 平替 pgvector 铺路。

## 七、自测题

1. 为什么语义搜索默认用余弦相似度而不是欧氏距离？什么情况下点积可以替代余弦？
2. Embedding 维度是不是越大越好？维度对存储和检索成本有什么影响？
3. 为什么建索引用批量 Embedding？真实 API 的批量有什么限制（条数/token）？
4. `min_score` 阈值定高了和定低了分别会出现什么用户体验问题？怎么科学地定阈值？
5. 暴力 Top-K 的复杂度是多少？什么规模下必须上 ANN 索引？
6. FAQ 去重中「簇代表取首条」有什么缺陷？如何改进？
7. Mock  embedding 为什么不能用来评估真实检索质量？它适合验证什么？
8. 如果文档更新了，向量索引应该怎么维护？（提示：upsert / delete / 重建，为 week06 热身）

## 八、验收自查

- [x] 能手算并解释余弦相似度、点积、欧氏距离的差异
- [x] 能调用 OpenAI 兼容 Embedding API（批量、重试、token 统计），Mock 接口一致
- [x] 能实现向量库（upsert/search/delete/持久化）
- [x] 能搭建语义搜索闭环并用阈值实现拒答
- [x] 能用 Embedding 做 FAQ 语义去重并解释阈值权衡
- [x] 41 个 pytest 全部通过

## 九、下周预告（week06）

- Vector Search：pgvector 安装与表设计
- Top-K 索引加速（HNSW / IVFFlat）与暴力检索对照
- Metadata Filter（按部门/类别过滤检索）
- 把本周 VectorStore 的接口平替成 PostgreSQL 实现
