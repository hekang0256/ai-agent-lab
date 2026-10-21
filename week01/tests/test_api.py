"""Web API 的端到端测试（需要先启动 PostgreSQL）。

运行方式：
    docker compose up -d
    pytest tests -v
"""
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault(
    "DATABASE_URL", "postgresql://lab:lab123@localhost:5432/ai_agent_lab"
)

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_get_update_delete_task():
    # CREATE
    r = client.post(
        "/tasks", json={"title": "pytest 任务", "priority": 1}
    )
    assert r.status_code == 201
    task = r.json()
    assert task["title"] == "pytest 任务"
    assert task["status"] == "todo"

    # GET
    r = client.get(f"/tasks/{task['id']}")
    assert r.status_code == 200

    # UPDATE
    r = client.patch(f"/tasks/{task['id']}", json={"status": "done"})
    assert r.status_code == 200
    assert r.json()["status"] == "done"

    # 事件日志已记录 create/update 两次事件
    r = client.get(f"/tasks/{task['id']}/events")
    actions = [e["action"] for e in r.json()]
    assert "created" in actions and "updated" in actions

    # DELETE
    r = client.delete(f"/tasks/{task['id']}")
    assert r.status_code == 204
    r = client.get(f"/tasks/{task['id']}")
    assert r.status_code == 404


def test_validation_error():
    # 空标题应触发 422（FastAPI/Pydantic 校验）
    r = client.post("/tasks", json={"title": ""})
    assert r.status_code == 422
    # 非法状态值应触发 422
    r = client.post("/tasks", json={"title": "x", "status": "nope"})
    assert r.status_code == 422


def test_filter_by_status():
    r = client.get("/tasks", params={"status": "done"})
    assert r.status_code == 200
    assert all(t["status"] == "done" for t in r.json())
