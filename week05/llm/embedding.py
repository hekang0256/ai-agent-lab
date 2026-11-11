"""Embedding 客户端：真实 OpenAI 兼容 API + 离线 Mock。

真实客户端：POST {base_url}/embeddings，OpenAI 兼容协议（阿里 DashScope 兼容模式、
OpenAI、vLLM、Ollama 等都支持同一格式），支持批量输入与重试。

MockEmbeddingClient 的设计目标：**确定性 + 保留主题语义**，让语义搜索可以离线演示和测试：
1. 内置主题词表，命中主题 → 该主题的维度块加权（语义信号，占主导）
2. 字符 2-gram 哈希散布 → 细粒度措辞信号（较弱）
3. L2 归一化输出
它不是真正的语义模型：同主题≠同语义，仅用于教学与 CI。真实效果请用真实模型复验。
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

from .similarity import normalize
from .tokenizer import estimate_text_tokens

load_dotenv()


@dataclass
class EmbeddingResponse:
    embeddings: list[list[float]]
    model: str
    total_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def dim(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0


class EmbeddingError(Exception):
    pass


# ---------- 真实 API ----------
class EmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.base_url = (
            base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
        self.max_retries = max_retries
        if not self.api_key:
            raise EmbeddingError("未配置 EMBEDDING_API_KEY / LLM_API_KEY，或设置 LLM_USE_MOCK=true。")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=timeout,
        )

    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResponse:
        if not texts:
            return EmbeddingResponse(embeddings=[], model=model or self.model)
        payload = {"model": model or self.model, "input": texts}
        started = time.perf_counter()
        data = self._post_with_retry("/embeddings", payload)
        latency_ms = (time.perf_counter() - started) * 1000
        # 按 index 排序，保证返回顺序与输入一致
        items = sorted(data["data"], key=lambda d: d["index"])
        usage = data.get("usage", {}) or {}
        return EmbeddingResponse(
            embeddings=[item["embedding"] for item in items],
            model=data.get("model", model or self.model),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency_ms,
        )

    def _post_with_retry(self, path: str, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(path, json=payload)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    raise EmbeddingError(f"Embedding 业务错误 {resp.status_code}: {resp.text}")
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise httpx.HTTPStatusError(f"server {resp.status_code}", request=resp.request, response=resp)
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_err = e
                time.sleep(2**attempt * 0.5)
        raise EmbeddingError(f"Embedding 调用失败，已重试 {self.max_retries} 次: {last_err}")

    def close(self) -> None:
        self._client.close()


# ---------- 离线 Mock ----------
# 主题词表：每个主题占 8 个连续维度。覆盖本周企业制度语料与常见问法。
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "expense": ["报销", "费用", "发票", "住宿", "机票", "打车", "差旅"],
    "leave": ["休假", "请假", "年假", "病假", "调休", "事假", "考勤"],
    "overtime": ["加班", "加班费", "补休", "调班"],
    "security": ["保密", "机密", "泄露", "涉密", "竞业"],
    "remote": ["远程", "在家办公", "居家", "混合办公"],
    "purchase": ["采购", "供应商", "招标", "比价"],
    "it": ["服务器", "vpn", "密码", "账号", "权限", "邮箱", "电脑"],
    "contract": ["合同", "签约", "违约", "法务"],
    "salary": ["工资", "薪酬", "奖金", "薪资", "发薪"],
    "onboard": ["入职", "离职", "转正", "试用期"],
    "meeting": ["会议", "纪要", "会议室"],
    "travel": ["出差", "行程", "审批"],
}
TOPIC_BLOCK = 8


class MockEmbeddingClient:
    def __init__(self, dim: int = 256):
        if dim < len(TOPIC_KEYWORDS) * TOPIC_BLOCK:
            raise ValueError(f"dim 至少为 {len(TOPIC_KEYWORDS) * TOPIC_BLOCK}")
        self.dim = dim
        self.model = "mock-embedding-v1"

    def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResponse:
        started = time.perf_counter()
        vectors = [self._embed_one(t) for t in texts]
        return EmbeddingResponse(
            embeddings=vectors,
            model=model or self.model,
            total_tokens=sum(estimate_text_tokens(t) for t in texts),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _embed_one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        lowered = text.lower()

        # 1) 主题信号（主导）：命中一个关键词，该主题块每个维度 +0.6
        for topic_idx, keywords in enumerate(TOPIC_KEYWORDS.values()):
            hits = sum(1 for kw in keywords if kw in lowered)
            if hits:
                base = topic_idx * TOPIC_BLOCK
                for j in range(base, base + TOPIC_BLOCK):
                    v[j] += 0.6 * hits

        # 2) 措辞信号（较弱）：字符 2-gram 哈希散布，让同主题不同措辞也有区分度
        grams = [text[i: i + 2] for i in range(max(0, len(text) - 1))]
        for gram in grams:
            h = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:4], "big")
            v[h % self.dim] += 0.12
            v[(h // self.dim) % self.dim] += 0.06

        return normalize(v)


def get_embedding_client() -> "EmbeddingClient | MockEmbeddingClient":
    if os.getenv("LLM_USE_MOCK", "").lower() == "true":
        return MockEmbeddingClient()
    return EmbeddingClient()
