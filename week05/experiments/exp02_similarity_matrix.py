"""实验5-B：语义相似度矩阵 —— 直观看到「字面不同，语义相近」。

用 MockEmbeddingClient 离线生成 8 个句子的向量，输出两两余弦矩阵，
并对每句话找出最相似的另一句。真实模型只需换 client，代码不变。
"""
from __future__ import annotations

from llm.embedding import get_embedding_client
from llm.similarity import cosine_matrix

SENTENCES = [
    "出差打车费用怎么报销",   # 0 报销
    "打车发票可以报销吗",     # 1 报销（措辞不同）
    "年假有几天",           # 2 休假
    "带薪年假是多少天",       # 3 休假（措辞不同）
    "服务器权限怎么申请",     # 4 IT
    "VPN 连不上内网怎么办",   # 5 IT
    "今天中午吃什么",        # 6 无关
    "附近哪家餐厅好吃",      # 7 无关（但与 6 的措辞有重合）
]


def main() -> None:
    client = get_embedding_client()
    resp = client.embed(SENTENCES)
    print(f"模型={resp.model} 维度={resp.dim} 估算tokens={resp.total_tokens}")

    matrix = cosine_matrix(resp.embeddings)
    print("\n== 两两余弦相似度矩阵 ==")
    print("     " + "".join(f"  S{i}   " for i in range(len(SENTENCES))))
    for i, row in enumerate(matrix):
        print(f"S{i}   " + "".join(f"{v:6.2f} " for v in row))

    print("\n== 每句话最相似的另一句 ==")
    for i, row in enumerate(matrix):
        best_j = max((j for j in range(len(SENTENCES)) if j != i), key=lambda j: row[j])
        print(f"S{i}「{SENTENCES[i]}」\n   → S{best_j}「{SENTENCES[best_j]}」 相似度 {row[best_j]:.2f}")


if __name__ == "__main__":
    main()
