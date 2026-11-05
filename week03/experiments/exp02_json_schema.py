"""实验3-B：JSON Schema + Pydantic —— 强约束结构化输出。

演示 StructuredClient 如何让 LLM 输出严格符合 Pydantic 模型的数据。

运行：python -m experiments.exp02_json_schema
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from llm import get_client
from llm.structured import StructuredClient


class Person(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(ge=0, le=150, description="年龄")
    job: str | None = Field(default=None, description="职业，可选")


def demo() -> None:
    llm = get_client()
    client = StructuredClient(llm)

    print("=" * 60)
    print("Pydantic Schema 自动转 JSON Schema 注入 Prompt")
    print("=" * 60)
    import json
    print(json.dumps(Person.model_json_schema(), ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("调用 StructuredClient.extract()")
    print("=" * 60)
    try:
        person = client.extract(
            schema=Person,
            system="你是信息提取专家，从用户文本中提取人物信息。",
            user="张三今年 28 岁，是个工程师。",
        )
        print(f"✅ 提取成功: {person}")
        print(f"   name={person.name!r}  age={person.age}  job={person.job!r}")
    except Exception as e:
        print(f"⚠️  提取失败（Mock 不返回 Person schema 是预期的）: {type(e).__name__}: {e}")
        print("    真实 API 下会成功；这里演示的是接口与重试机制。")


if __name__ == "__main__":
    demo()
