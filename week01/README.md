# Week01 ｜ 开发环境与 AI 编程基础

> 《AI → 智能体》24周系统学习路线 · 第0阶段（第1周）
> 实验目标：建立 ai-agent-lab 仓库；完成 Python API 调用；Docker 运行 PostgreSQL；写一个简单 Web API。
> 验收标准：能独立创建 Python 项目、安装依赖、调用 HTTP API、写 SQL、提交 Git。

## 一、本周做了什么

| 实验 | 内容 | 结果 |
|------|------|------|
| 实验1 | Python 调用 HTTP API（httpx 同步/异步、JSON、超时重试、外部API） | ✅ 通过 |
| 实验2 | FastAPI Web API（tasks CRUD + 参数校验 + 事件日志） | ✅ 通过 |
| 实验3 | Docker Compose 运行 PostgreSQL，自动执行 init.sql | ✅ 通过 |
| 实验4 | Python 中写并执行 SQL（SELECT/INSERT/UPDATE/JOIN/聚合/DELETE） | ✅ 通过 |
| 测试 | pytest 端到端测试 4 条 | ✅ 4 passed |

## 二、架构图

```text
┌────────────────────────────────────────────────────┐
│  scripts/api_call.py (httpx 客户端)                 │
│  scripts/sql_basics.py (SQL 实验)                   │
│  tests/test_api.py (pytest)                         │
└───────────────┬────────────────────────────────────┘
                │ HTTP (JSON)                 │ SQL
┌───────────────▼──────────────┐  ┌───────────▼────────────┐
│  FastAPI Web API (uvicorn)   │  │  PostgreSQL 16 (Docker) │
│  app/main.py  tasks CRUD     │  │  tasks / task_events    │
│  Pydantic 参数校验            │  │  init.sql 自动初始化     │
└──────────────────────────────┘  └────────────────────────┘
```

请求链路：
`客户端 → GET/POST/PATCH/DELETE /tasks → FastAPI → Pydantic 校验 → psycopg → PostgreSQL → JSON 响应`

## 三、目录结构

```text
week01/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI 入口：/health、/tasks CRUD、/tasks/{id}/events
│   └── db.py            # psycopg 连接与 fetch_all/fetch_one/execute 辅助函数
├── scripts/
│   ├── api_call.py      # 实验1：同步/异步 HTTP 调用、超时重试、外部 API
│   └── sql_basics.py    # 实验4：基础 SQL 操作
├── db/
│   └── init.sql         # 建表 + 种子数据（容器首次启动自动执行）
├── tests/
│   └── test_api.py      # pytest 端到端测试
├── docker-compose.yml   # PostgreSQL 16
├── requirements.txt
└── README.md
```

## 四、如何运行（从一台全新机器复现）

```bash
# 1. 启动数据库（自动初始化 schema 和种子数据）
docker compose up -d

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. 启动 Web API
uvicorn app.main:app --reload
# 交互式文档: http://127.0.0.1:8000/docs

# 4. 跑实验
python -m scripts.api_call     # 实验1：HTTP API 调用
python -m scripts.sql_basics   # 实验4：SQL 基础

# 5. 跑测试
pytest tests -v
```

数据库连接串（可用环境变量 `DATABASE_URL` 覆盖）：
`postgresql://lab:lab123@localhost:5432/ai_agent_lab`

## 五、API 一览

| 方法 | 路径 | 说明 | 状态码 |
|------|------|------|--------|
| GET | /health | 健康检查 | 200 |
| GET | /tasks?status=&limit= | 任务列表，可按状态过滤 | 200 |
| POST | /tasks | 新建任务（自动写事件日志） | 201 |
| GET | /tasks/{id} | 任务详情 | 200 / 404 |
| PATCH | /tasks/{id} | 部分更新 | 200 / 404 |
| DELETE | /tasks/{id} | 删除任务及其事件 | 204 / 404 |
| GET | /tasks/{id}/events | 任务事件日志 | 200 |

## 六、测试数据（种子）

`db/init.sql` 内置 5 条任务 + 5 条事件日志：

| id | title | status | priority |
|----|-------|--------|----------|
| 1 | 搭建 ai-agent-lab 仓库 | done | 1 |
| 2 | Docker 运行 PostgreSQL | doing | 1 |
| 3 | 写一个简单的 Web API | todo | 2 |
| 4 | Python 调用 HTTP API | todo | 2 |
| 5 | 复盘并写 README | todo | 3 |

## 七、测试结果

```text
tests/test_api.py::test_health PASSED                    [ 25%]
tests/test_api.py::test_create_get_update_delete_task PASSED [ 50%]
tests/test_api.py::test_validation_error PASSED          [ 75%]
tests/test_api.py::test_filter_by_status PASSED          [100%]
4 passed, 1 warning in 1.14s
```

关键验证点：
- 创建→查询→更新→删除全链路，删除后再次查询返回 **404**；
- 空标题 / 非法 status 被 Pydantic 拦截返回 **422**；
- 任务创建/更新会写入 `task_events` 事件日志，可用 JOIN 查询；
- httpx 5 个异步并发请求全部 200，总耗时 0.088s。

## 八、本周问题清单（遇到的坑与解决）

1. **Docker Desktop 未启动**：`docker info` 连接失败 → 先启动 Docker Desktop 再执行 compose。
2. **5432 端口被旧容器占用**：之前实验残留容器 `ai-agent-lab-postgres` 占用端口 → `docker rm -f` 清理后按仓库 `docker-compose.yml` 统一重建，保证可复现。
3. **外网 API 超时**：本机网络代理限制导致 jsonplaceholder 请求超时 → 在代码中捕获 `httpx.TimeoutException` 并降级打印，体会“调用外部 API 必须处理失败”。
4. **PowerShell 中文乱码**：终端默认非 UTF-8 → 设置 `[Console]::OutputEncoding = UTF8` 且 `PYTHONIOENCODING=utf-8`。
5. **直接 INSERT 的任务没有事件日志**：绕过 API 插数据导致 JOIN 查不到事件 → 数据一致性应由应用层保证（API 里创建任务同事务写事件）。

## 九、自测题（5~10道）

1. REST 中 GET/POST/PATCH/DELETE 语义分别是什么？为什么 PATCH 而不是 PUT？
2. HTTP 状态码 200/201/204/400/404/422 各代表什么？422 是谁产生的（Web 框架还是数据库）？
3. `async/await` 和同步请求的区别是什么？什么时候必须用异步？
4. `docker compose up -d` 的 `-d` 是什么意思？`init.sql` 为什么放在 `/docker-entrypoint-initdb.d/`？
5. psycopg 的 `%s` 占位符和 Python f-string 拼 SQL 有什么区别？为什么后者危险（SQL 注入）？
6. 为什么创建任务要写 `task_events` 事件日志？这对后续“Observability/Trace”周有什么帮助？
7. 如果把 `POSTGRES_PASSWORD` 直接写进 docker-compose.yml 并提交 Git，有什么风险？应该怎么改？
8. 如何给 `/tasks` 加分页（limit/offset）？offset 深翻页有什么缺点？

## 十、验收自查

- [x] 独立创建 Python 项目（venv + requirements.txt + pip 安装依赖）
- [x] 调用 HTTP API（同步、异步、JSON 解析、错误处理）
- [x] 写 SQL（建表、增删改查、JOIN、聚合）
- [x] Docker Compose 启动 PostgreSQL 并持久化数据（volume）
- [x] 提交 Git（git config: kekang / 1650778095@qq.com）
