# ai-agent-lab

《AI → 智能体》24周系统学习路线（企业AI/FDE方向）的统一主项目：企业AI实验室。

整个24周不做十个互不相关的 Demo，而是建立一个持续演进的项目：

```
V1 LLM聊天 → V2 Structured Output → V3 文档解析 → V4 Embedding与搜索
→ V5 RAG → V6 企业知识库 → V7 Tool Calling → V8 MCP → V9 单Agent
→ V10 Agentic RAG → V11 Workflow → V12 Multi-Agent → V13 Evaluation
→ V14 Security → V15 Observability → V16 本地模型/私有化实验
```

最终形态：
`用户 → AI入口 → Agent → Knowledge/RAG + Tools/MCP → 模拟企业系统 → Evaluation/Trace/Security`

## 目录

| 目录 | 内容 |
|------|------|
| [week01/](week01/README.md) | 第1周：开发环境（Python API 调用、Docker PostgreSQL、FastAPI Web API、SQL、Git） |
| [week02/](week02/README.md) | 第2周：LLM 基础认知 + API 调用 + Prompt 工程（Token/Context Window、Prompt 五要素、Few-shot、文本工具箱、项目日报生成器） |
| [week03/](week03/README.md) | 第3周：结构化输出 + Tool Calling + Streaming（JSON Schema/Pydantic、Tool 装饰器与注册表、SSE 流式、合同提取器、模型对比框架） |
| [week04/](week04/README.md) | 第4周：多模态 + Context Engineering + 模型选型与成本（图像/音频输入、上下文裁剪与摘要、长期记忆、模型路由、第1阶段综合验收） |

## 全局约定

- Git 身份：`kekang <1650778095@qq.com>`
- 所有实验都在 D 盘本仓库内完成，Python 依赖装进各周目录下的 `.venv`
- 每周产出：代码、README、架构图、测试数据、测试结果、问题清单、自测题
