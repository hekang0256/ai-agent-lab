"""实验1：用 Python 调用 HTTP API。

演示内容：
1. httpx 同步 GET / POST，解析 JSON
2. httpx 异步并发请求（async/await）
3. HTTP 状态码与错误处理、超时与重试

运行前先启动本项目的 Web API：
    uvicorn app.main:app --reload
再执行：
    python scripts/api_call.py
"""
import asyncio
import time

import httpx

BASE = "http://127.0.0.1:8000"


def sync_demo() -> None:
    print("== 1. 同步调用 ==")
    with httpx.Client(base_url=BASE, timeout=5.0) as client:
        r = client.get("/health")
        print("GET /health ->", r.status_code, r.json())
        assert r.status_code == 200

        # POST 提交 JSON，服务端返回创建的资源
        r = client.post(
            "/tasks",
            json={"title": "来自 api_call.py 的任务", "priority": 2},
        )
        print("POST /tasks ->", r.status_code, r.json())
        assert r.status_code == 201
        task_id = r.json()["id"]

        # GET 路径参数
        r = client.get(f"/tasks/{task_id}")
        print(f"GET /tasks/{task_id} ->", r.status_code, r.json())

        # DELETE 清理
        r = client.delete(f"/tasks/{task_id}")
        print(f"DELETE /tasks/{task_id} ->", r.status_code)
        assert r.status_code == 204

        # 404 错误处理
        r = client.get(f"/tasks/{task_id}")
        print(f"GET /tasks/{task_id} (again) ->", r.status_code, r.json())
        assert r.status_code == 404


def timeout_and_retry_demo() -> None:
    print("\n== 2. 超时与重试 ==")
    for attempt in range(3):
        try:
            r = httpx.get(f"{BASE}/health", timeout=2.0)
            print(f"attempt {attempt + 1}: status={r.status_code}")
            return
        except httpx.TimeoutException as e:
            print(f"attempt {attempt + 1}: 超时 -> {type(e).__name__}")
        except httpx.ConnectError as e:
            print(f"attempt {attempt + 1}: 连接失败（服务未启动？）-> {e}")
            return
    print("重试 3 次仍失败")


async def _get_health(client: httpx.AsyncClient, n: int) -> int:
    r = await client.get("/health")
    print(f"  async request #{n} -> {r.status_code}")
    return r.status_code


async def async_demo() -> None:
    print("\n== 3. 异步并发（async/await）==")
    start = time.perf_counter()
    async with httpx.AsyncClient(base_url=BASE, timeout=5.0) as client:
        # 并发发出 5 个请求，而不是串行等待
        results = await asyncio.gather(*[_get_health(client, i) for i in range(1, 6)])
    elapsed = time.perf_counter() - start
    print(f"5 个并发请求全部完成: {results}，总耗时 {elapsed:.3f}s")


def public_api_demo() -> None:
    """调用外部公开 API（JSONPlaceholder），验证调用外部 HTTP API 的能力。

    注意：本机网络受限时外网请求可能超时，这里把异常捕获并打印，
    演示“调用外部 API 必须考虑失败情况”。
    """
    print("\n== 4. 调用外部公开 API ==")
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get("https://jsonplaceholder.typicode.com/todos/1")
            print("GET jsonplaceholder /todos/1 ->", r.status_code)
            print("响应 JSON:", r.json())
    except httpx.TimeoutException:
        print("外网请求超时（本机网络/代理限制）——生产代码必须处理这种情况")
    except httpx.HTTPError as e:
        print(f"外网请求失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    sync_demo()
    timeout_and_retry_demo()
    asyncio.run(async_demo())
    public_api_demo()
