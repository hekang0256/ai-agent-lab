"""实验补充：在 Python 里写并执行 SQL。

演示 SELECT / INSERT / UPDATE / DELETE / JOIN / 聚合，
对应 Week01 学习目标“PostgreSQL 基础 SQL”。

运行前先启动数据库：
    docker compose up -d
再执行：
    python scripts/sql_basics.py
"""
from app.db import execute, fetch_all, fetch_one


def main() -> None:
    print("== 1. SELECT：全部任务 ==")
    for row in fetch_all("SELECT id, title, status, priority FROM tasks ORDER BY id"):
        print(row)

    print("\n== 2. WHERE + ORDER BY：高优先级未完成任务 ==")
    rows = fetch_all(
        "SELECT id, title, priority FROM tasks"
        " WHERE status != 'done' AND priority <= 2 ORDER BY priority, id"
    )
    for row in rows:
        print(row)

    print("\n== 3. INSERT + RETURNING：新增任务 ==")
    new = fetch_one(
        "INSERT INTO tasks (title, status, priority) VALUES (%s, %s, %s) RETURNING *",
        ("sql_basics.py 新增的任务", "doing", 2),
    )
    execute(
        "INSERT INTO task_events (task_id, action) VALUES (%s, 'created')",
        (new["id"],),
    )
    print(new)

    print("\n== 4. UPDATE：修改状态 ==")
    updated = fetch_one(
        "UPDATE tasks SET status = %s, updated_at = now() WHERE id = %s RETURNING *",
        ("done", new["id"]),
    )
    print(updated)

    print("\n== 5. JOIN：任务 + 事件日志 ==")
    rows = fetch_all(
        "SELECT t.id, t.title, e.action, e.created_at"
        " FROM tasks t JOIN task_events e ON e.task_id = t.id"
        " WHERE t.id = %s ORDER BY e.id",
        (new["id"],),
    )
    for row in rows:
        print(row)

    print("\n== 6. 聚合：按状态统计 ==")
    for row in fetch_all(
        "SELECT status, COUNT(*) AS cnt, AVG(priority) AS avg_priority"
        " FROM tasks GROUP BY status ORDER BY cnt DESC"
    ):
        print(row)

    print("\n== 7. DELETE：清理测试数据 ==")
    execute("DELETE FROM tasks WHERE id = %s", (new["id"],))
    print(f"已删除任务 id={new['id']}")


if __name__ == "__main__":
    main()
