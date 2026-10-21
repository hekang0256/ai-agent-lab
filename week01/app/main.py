from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.db import execute, fetch_all, fetch_one

app = FastAPI(title="ai-agent-lab Week01 API", version="0.1.0")

Status = Literal["todo", "doing", "done"]


class TaskCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    status: Status = "todo"
    priority: int = Field(default=3, ge=1, le=5)


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    status: Status | None = None
    priority: int | None = Field(default=None, ge=1, le=5)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ai-agent-lab-week01"}


@app.get("/tasks")
def list_tasks(
    status: Status | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict]:
    """按状态过滤任务，返回 JSON 列表。"""
    if status:
        return fetch_all(
            "SELECT * FROM tasks WHERE status = %s ORDER BY id LIMIT %s",
            (status, limit),
        )
    return fetch_all("SELECT * FROM tasks ORDER BY id LIMIT %s", (limit,))


@app.post("/tasks", status_code=201)
def create_task(task: TaskCreate) -> dict:
    row = fetch_one(
        "INSERT INTO tasks (title, description, status, priority)"
        " VALUES (%s, %s, %s, %s) RETURNING *",
        (task.title, task.description, task.status, task.priority),
    )
    execute(
        "INSERT INTO task_events (task_id, action) VALUES (%s, 'created')",
        (row["id"],),
    )
    return row


@app.get("/tasks/{task_id}")
def get_task(task_id: int) -> dict:
    row = fetch_one("SELECT * FROM tasks WHERE id = %s", (task_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return row


@app.patch("/tasks/{task_id}")
def update_task(task_id: int, task: TaskUpdate) -> dict:
    # 只更新传入的字段
    fields = task.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")
    sets = ", ".join(f"{k} = %s" for k in fields)
    values = tuple(fields.values()) + (task_id,)
    row = fetch_one(
        f"UPDATE tasks SET {sets}, updated_at = now() WHERE id = %s RETURNING *",
        values,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    execute(
        "INSERT INTO task_events (task_id, action) VALUES (%s, 'updated')",
        (task_id,),
    )
    return row


@app.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: int) -> None:
    execute("DELETE FROM task_events WHERE task_id = %s", (task_id,))
    row = fetch_one("DELETE FROM tasks WHERE id = %s RETURNING id", (task_id,))
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")


@app.get("/tasks/{task_id}/events")
def task_events(task_id: int) -> list[dict]:
    return fetch_all(
        "SELECT * FROM task_events WHERE task_id = %s ORDER BY id", (task_id,)
    )
