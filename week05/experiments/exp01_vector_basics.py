"""实验5-A：向量与相似度的直觉（手算小向量，不依赖任何模型）。

回答三个问题：
1. 为什么语义搜索用余弦相似度而不是欧氏距离？（方向 vs 长度）
2. 归一化带来什么好处？
3. 「词向量可以做算术」（king - man + woman ≈ queen）到底是什么意思？
"""
from __future__ import annotations

from llm.similarity import cosine_similarity, dot, l2_distance, normalize, top_k


def main() -> None:
    print("== 1. 余弦看方向，欧氏看距离 ==")
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    c = [2.0, 0.0]  # 与 a 同方向，但长度是 2 倍
    print(f"a={a}, b={b}, c={c}")
    print(f"cos(a,b)={cosine_similarity(a, b):.2f}  → 正交，完全不相关")
    print(f"cos(a,c)={cosine_similarity(a, c):.2f}  → 同方向，语义相同（长度无关）")
    print(f"l2(a,c)={l2_distance(a, c):.2f}      → 欧氏距离却认为它们不同")
    print(f"dot(a,c)={dot(a, c):.2f}          → 点积压光了「同向」信号，但受长度影响")

    print("\n== 2. 归一化后：点积 == 余弦 ==")
    na, nc = normalize(a), normalize(c)
    print(f"normalize(c)={nc}")
    print(f"dot(norm(a), norm(c))={dot(na, nc):.4f} == cos(a,c)={cosine_similarity(a, c):.4f}")
    print("工程意义：向量入库前先归一化，检索时用点积即可，省一次除法；pgvector 同理。")

    print("\n== 3. 向量算术（原理示意，用人工构造的 4 维向量） ==")
    # 维度含义（人为指定）：[王室, 男性, 女性, 普通]
    king = [1.0, 1.0, 0.0, 0.0]
    man = [0.0, 1.0, 0.0, 1.0]
    woman = [0.0, 0.0, 1.0, 1.0]
    queen = [1.0, 0.0, 1.0, 0.0]
    result = [king[i] - man[i] + woman[i] for i in range(4)]
    print(f"king - man + woman = {result}")
    print(f"与 queen={queen} 的余弦相似度：{cosine_similarity(result, queen):.4f}")
    print("含义：embedding 把「语义关系」编码成了「空间中的方向与偏移」。")

    print("\n== 4. Top-K 排序 ==")
    vectors = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    for idx, score in top_k([1.0, 0.05], vectors, k=3):
        print(f"  向量{idx} {vectors[idx]}  相似度 {score:.4f}")


if __name__ == "__main__":
    main()
