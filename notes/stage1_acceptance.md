# 第1阶段自查报告（week01 ~ week04）

> 自查时间：2026-11-06 ｜ 方式：对照各周 README 验收清单 + 检查产出文件 + 实跑全部测试与关键应用
> 结论：**第1阶段的学习、实验、验收内容已完成，准予进入第2阶段**。
> 111 条 pytest 中 109 条通过；剩余 2 条为 week01 数据库端到端测试，因当前机器无 Docker/PostgreSQL 未能复验（非代码缺陷，历史已验证通过）。

## 一、总览

| 周 | 主题 | pytest | 本次实跑 | 应用复验（Mock） | 产出文件 |
|---|---|---:|---|---|---|
| week01 | 开发环境：Python API / Docker PostgreSQL / FastAPI / SQL / Git | 4 | ⚠️ 2 passed + 2 failed（缺 DB） | 无需 Mock，依赖 DB | ✅ init.sql / docker-compose.yml |
| week02 | LLM 基础 + API 调用 + Prompt 工程 | 24 | ✅ 24 passed | ✅ 日报生成器 | ✅ report_input.json |
| week03 | 结构化输出 + Tool Calling + Streaming | 22 | ✅ 22 passed | ✅ 合同提取 + 模型对比 | ✅ 合同样本 / 提取结果 / 对比表 |
| week04 | 多模态 + 上下文与记忆 + 模型选型与成本 | 61 | ✅ 61 passed | ✅ 发票提取 + 成本仿真 + 综合助手 | ✅ 发票图 / 音频 / 提取结果 / 成本报告 |
| 合计 | | **111** | **109 passed / 2 环境受阻** | | |

## 二、逐周验收清单核对

### week01（验收标准：独立建项目、调 HTTP API、写 SQL、Docker 起 DB、提交 Git）

- [x] 独立创建 Python 项目（venv + requirements.txt）
- [x] 调用 HTTP API（同步/异步/JSON/错误处理）——`scripts/api_call.py`
- [x] 写 SQL（建表/增删改查/JOIN/聚合）——`scripts/sql_basics.py` + `db/init.sql`
- [⚠️] Docker Compose 启动 PostgreSQL —— **本次未能复验**：当前机器未检测到 Docker Desktop 安装，5432 端口无监听、无 PostgreSQL 服务。`test_health` 与 `test_validation_error` 通过；`test_create_get_update_delete_task`、`test_filter_by_status` 因连接失败而 failed。README 已记录历史 4 passed。
- [x] 提交 Git（kekang / 1650778095@qq.com）

### week02（验收标准：独立调 OpenAI 兼容 API、Prompt 解决任务、判断 token 超窗）

- [x] Token / Context Window / Messages 关系（实验 0-A，含实测数据表）
- [x] httpx 实现 OpenAI 兼容 API 调用（含重试、token 统计）
- [x] 五要素 Prompt（实验 0-B）
- [x] Few-shot 固定格式输出（实验 0-C + 日报生成器）
- [x] 接口一致的 Mock（CI/离线）
- [x] 24 个 pytest 通过（本次复跑确认）

### week03（验收标准：Pydantic 强约束、Tool Calling 闭环、流式与非流式差异）

- [x] 复杂 schema（嵌套/可选/日期/枚举）+ StructuredClient 失败回喂重试
- [x] Tool Calling 完整闭环（注册→调用→执行→回喂→回答）
- [x] Streaming / SSE 解析 / TTFT 测量
- [x] 模型对比实验框架（多变量、多指标）
- [x] 22 个 pytest 通过（本次复跑确认）；合同提取与模型对比应用 Mock 复跑成功

### week04（验收标准：多模态、上下文策略、跨会话记忆、模型路由、综合组装）

- [x] 图像/音频 content parts；发票图片 → Pydantic JSON 闭环
- [x] 三种上下文策略量化对比（sliding 遗忘 vs summary 保留）
- [x] 长期记忆持久化 + 显式「记住/忘掉」
- [x] 预算约束、决策可解释的模型路由器（仿真节省 91.1%）
- [x] ChatSession 综合组装（对话+记忆+工具+上下文+成本）
- [x] 61 个 pytest 通过（本次复跑确认）

## 三、全局约定核对（根 README）

| 约定 | 结果 |
|---|---|
| 每周产出：代码 | ✅ 四周均有可运行代码 |
| README | ✅ 四周均有，含架构图 |
| 架构图 | ✅ 均在各周 README 第二节 |
| 测试数据 | ✅ 种子 SQL / 日报输入 / 合同文本 / 发票图片与任务集 |
| 测试结果 | ✅ 各周 README 记录 + 本次复跑；week03/04 结果文件入库 |
| 问题清单 | ✅ 各周 README「本周问题清单」 |
| 自测题 | ✅ week01 8 题 / week02 8 题 / week03 8 题 / week04 8 题 |
| 实验都在 D 盘本仓库、依赖进各周 .venv | ✅ |

## 四、未覆盖项（如实记录）

1. **week01 数据库端到端测试未能在当前机器复验**（环境缺失，非代码问题）。
   复验方法：安装并启动 Docker Desktop 后执行
   ```bash
   cd week01
   docker compose up -d
   .venv\Scripts\python.exe -m pytest tests -q   # 预期 4 passed
   ```
2. **真实 LLM API 未验证**：week02~week04 全部以 Mock 离线复验，流程与代码路径已覆盖，但真实模型的输出质量、延迟、成本未实测（week03 README 已标注「结论待真实 API 填充」）。
   如需实测：配置 `.env`（LLM_API_KEY）后取消 `LLM_USE_MOCK`，重跑各周实验；多模态需换视觉模型。
3. **成本数字为教学示例价**：week04 价格表非实时报价，节省比例（91.1%）用于说明方法论，不代表真实账单。

## 五、结论与建议

- 第1阶段「工程底座 → LLM 认知 → 可靠交互 → 生产思维」四条主线均有代码、实验、测试与文档支撑，**内容层面已完成**。
- 建议：有空时补验 week01 的 DB 两项测试；配置一次真实 API key 做小规模真实对照，即可消除全部未覆盖项。
- 可以进入第2阶段（week05：文档解析 / Chunking / Embedding / RAG）。
