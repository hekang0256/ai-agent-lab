"""应用2：FAQ 语义去重（客服知识库清洗）。

场景：客服 FAQ 来自多个渠道，存在大量「措辞不同但问的是同一件事」的问题。
做法：批量 Embedding → 贪心聚类（与簇代表的余弦相似度 ≥ 阈值则并入）。

用法：python -m apps.faq_dedup [--threshold 0.75]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm.embedding import get_embedding_client
from llm.similarity import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]


def dedup_faqs(
    faqs: list[str],
    client=None,
    *,
    threshold: float = 0.75,
) -> list[list[str]]:
    """贪心聚类：每条问题与现有簇的代表（首条）比较，相似度达标则并入。

    返回簇列表；每个簇就是「同一个问题的不同问法」。
    注意：簇代表是首条问题，这是启发式；更精细的做法是簇内两两比较或取质心。
    """
    if not faqs:
        return []
    client = client or get_embedding_client()
    vectors = client.embed(faqs).embeddings

    clusters: list[list[str]] = []
    representatives: list[list[float]] = []
    for text, vec in zip(faqs, vectors):
        best_cluster, best_score = -1, threshold
        for ci, rep in enumerate(representatives):
            score = cosine_similarity(vec, rep)
            if score >= best_score:
                best_cluster, best_score = ci, score
        if best_cluster == -1:
            clusters.append([text])
            representatives.append(vec)
        else:
            clusters[best_cluster].append(text)
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ 语义去重")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()

    faqs = json.loads((ROOT / "data" / "faq_raw.json").read_text(encoding="utf-8"))
    clusters = dedup_faqs(faqs, threshold=args.threshold)

    print(f"原始问题 {len(faqs)} 条 → 去重后 {len(clusters)} 簇（阈值 {args.threshold}）\n")
    for i, cluster in enumerate(clusters, 1):
        print(f"簇 {i}（{len(cluster)} 条）")
        for text in cluster:
            print(f"  - {text}")

    out = ROOT / "data" / "faq_groups.json"
    out.write_text(json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n分组结果已写入 {out}")


if __name__ == "__main__":
    main()
