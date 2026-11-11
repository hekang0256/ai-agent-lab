"""实验5-D：FAQ 语义去重 + 阈值敏感性。

同一个 FAQ 集合，分别用 0.75 / 0.9 / 0.99 三个阈值聚类：
阈值越低，合并越激进（误并风险高）；阈值越高，越保守（漏并风险高）。
"""
from __future__ import annotations

import json
from pathlib import Path

from apps.faq_dedup import dedup_faqs
from llm.embedding import get_embedding_client

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    faqs = json.loads((ROOT / "data" / "faq_raw.json").read_text(encoding="utf-8"))
    client = get_embedding_client()

    for threshold in (0.75, 0.9, 0.99):
        clusters = dedup_faqs(faqs, client, threshold=threshold)
        print(f"\n===== 阈值 {threshold}：{len(faqs)} 条 → {len(clusters)} 簇 =====")
        for i, cluster in enumerate(clusters, 1):
            if len(cluster) > 1:
                print(f"  簇{i}: " + " ｜ ".join(cluster))
        singles = [c[0] for c in clusters if len(c) == 1]
        print(f"  独立问题 {len(singles)} 条：{'；'.join(singles)}")


if __name__ == "__main__":
    main()
