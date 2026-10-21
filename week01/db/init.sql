-- Week01 初始化：建表 + 种子数据（首次启动容器时自动执行）

CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'todo'
        CHECK (status IN ('todo', 'doing', 'done')),
    priority INTEGER NOT NULL DEFAULT 3 CHECK (priority BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS task_events (
    id SERIAL PRIMARY KEY,
    task_id INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL,
    detail TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 种子数据
INSERT INTO tasks (title, description, status, priority) VALUES
    ('搭建 ai-agent-lab 仓库', '创建仓库、git config、目录结构', 'done', 1),
    ('Docker 运行 PostgreSQL', '用 docker-compose 启动数据库', 'doing', 1),
    ('写一个简单的 Web API', 'FastAPI 实现 tasks CRUD', 'todo', 2),
    ('Python 调用 HTTP API', 'httpx 同步/异步请求实验', 'todo', 2),
    ('复盘并写 README', '记录测试结果与问题清单', 'todo', 3);

INSERT INTO task_events (task_id, action, detail)
SELECT id, 'created', 'seed data' FROM tasks;
