-- week06：pgvector 初始化（容器首次启动自动执行）
CREATE EXTENSION IF NOT EXISTS vector;

-- 企业制度文档向量表
-- embedding 维度与本周 Mock/真实模型输出对齐（Mock=256；换真实模型需重建表）
CREATE TABLE IF NOT EXISTS policy_docs (
    id          TEXT PRIMARY KEY,           -- 文档编号，如 P01
    title       TEXT NOT NULL,              -- 制度标题
    category    TEXT NOT NULL,              -- 元数据：财务/人事/IT...
    text        TEXT NOT NULL,              -- 原文（title + 正文）
    embedding   vector(256) NOT NULL,       -- 向量
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 元数据过滤常用字段的普通索引（与向量索引互补）
CREATE INDEX IF NOT EXISTS idx_policy_docs_category ON policy_docs(category);
