"""实验6-D：索引维护 —— upsert 更新文档、delete 删除文档、检索结果随之变化。

对应 week05 自测题 8：文档更新了，向量索引怎么维护？
答案：按业务主键 upsert / delete，索引由数据库自动维护，无需全量重建。
"""
from __future__ import annotations

import json
from pathlib import Path

from apps.policy_search_pg import PgPolicySearch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    search = PgPolicySearch()
    store = search.store
    if store.count() == 0:
        search.build_index()
    print(f"初始文档数：{store.count()}")

    # 1. 更新前：查询「电子发票」相关度一般
    q = "电子发票可以报销吗"
    print(f"\n== 更新前：{q} ==")
    for r in search.search(q, top_k=2):
        print(f"   {r.score:.3f} 《{r.metadata['title']}》")

    # 2. upsert 修改 P02：明确支持电子发票
    doc = store.get("P02")
    new_text = doc["text"] + "\n补充：全面支持电子发票报销，与纸质发票同等效力。"
    new_vector = search.client.embed([new_text]).embeddings[0]
    store.upsert("P02", doc["title"], doc["category"], new_text, new_vector)
    print("\n已 upsert P02（补充电子发票条款）")

    print(f"\n== 更新后：{q} ==")
    for r in search.search(q, top_k=2):
        print(f"   {r.score:.3f} 《{r.metadata['title']}》")

    # 3. delete P13（会议室）
    print(f"\n== 删除 P13（会议室与会议纪要） ==")
    print(f"delete 返回：{store.delete('P13')}，当前文档数：{store.count()}")
    print("查询「会议室怎么预约」Top-2：")
    for r in search.search("会议室怎么预约", top_k=2):
        print(f"   {r.score:.3f} 《{r.metadata['title']}》（P13 已消失）")

    # 4. 恢复现场：从语料文件重新 upsert P02 与 P13
    docs = json.loads((ROOT / "data" / "docs" / "policies.json").read_text(encoding="utf-8"))
    restore = [d for d in docs if d["id"] in ("P02", "P13")]
    texts = [f"{d['title']}\n{d['text']}" for d in restore]
    vectors = search.client.embed(texts).embeddings
    store.upsert_many([
        {"id": d["id"], "title": d["title"], "category": d["category"],
         "text": t, "vector": v}
        for d, t, v in zip(restore, texts, vectors)
    ])
    print(f"\n已恢复 P02/P13，最终文档数：{store.count()}")
    print("要点：upsert/delete 是文档级维护，索引自动更新；只有「换 embedding 模型/维度」才需要全量重建。")


if __name__ == "__main__":
    main()
