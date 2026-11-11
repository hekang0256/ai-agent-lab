"""应用层测试：制度语义搜索 + FAQ 去重（Mock 离线）。"""
import json

from apps.faq_dedup import dedup_faqs
from apps.semantic_search import PolicySearch
from llm.embedding import MockEmbeddingClient


def make_search() -> PolicySearch:
    search = PolicySearch(MockEmbeddingClient())
    search.build_index()
    return search


# ---------- 语义搜索 ----------
def test_build_index():
    search = make_search()
    assert len(search.store) == 14
    assert search.store.dim == 256


def test_expense_query_hits_expense_policy():
    search = make_search()
    results = search.search("出差打车费用怎么报销", top_k=3)
    assert results[0].id == "P01"
    assert results[0].score > 0.9


def test_leave_query_hits_leave_policy():
    search = make_search()
    assert search.search("年假能休几天", top_k=1)[0].id == "P03"


def test_it_query_hits_it_policy():
    search = make_search()
    assert search.search("服务器权限怎么申请", top_k=1)[0].id == "P09"


def test_irrelevant_query_is_rejected():
    search = make_search()
    assert search.search("今天中午吃什么", top_k=3, min_score=0.35) == []
    assert "未找到相关制度" in search.answer("今天中午吃什么")


def test_answer_contains_title_and_score():
    search = make_search()
    text = search.answer("打车费怎么报销")
    assert "差旅报销制度" in text and "相关度" in text


def test_index_persistence(tmp_path):
    search = make_search()
    path = tmp_path / "index.json"
    search.save_index(path)
    fresh = PolicySearch(MockEmbeddingClient())
    fresh.load_index(path)
    assert len(fresh.store) == 14
    assert fresh.search("年假能休几天", top_k=1)[0].id == "P03"


# ---------- FAQ 去重 ----------
def load_faqs() -> list[str]:
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "faq_raw.json"
    return json.loads(path.read_text(encoding="utf-8"))


def cluster_of(clusters, text):
    for cluster in clusters:
        if text in cluster:
            return set(cluster)
    raise AssertionError(f"{text} 不在任何簇中")


def test_dedup_groups_synonymous_questions():
    faqs = load_faqs()
    clusters = dedup_faqs(faqs, MockEmbeddingClient(), threshold=0.75)
    assert len(clusters) == 7  # 实测：4 组近义 + 3 条独立
    # 打车报销三问同簇
    assert cluster_of(clusters, "怎么报销打车费？") == cluster_of(clusters, "打车发票可以报销吗？")
    # 年假三问同簇
    assert cluster_of(clusters, "年假有几天？") == cluster_of(clusters, "带薪年假是多少天？")


def test_dedup_keeps_unrelated_questions_separate():
    faqs = load_faqs()
    clusters = dedup_faqs(faqs, MockEmbeddingClient(), threshold=0.75)
    assert cluster_of(clusters, "会议室怎么预约？") == {"会议室怎么预约？"}
    assert cluster_of(clusters, "如何申请远程办公？") == {"如何申请远程办公？"}


def test_dedup_higher_threshold_splits_more():
    faqs = load_faqs()
    client = MockEmbeddingClient()
    loose = dedup_faqs(faqs, client, threshold=0.75)
    strict = dedup_faqs(faqs, client, threshold=0.99)
    assert len(strict) > len(loose)


def test_dedup_empty_input():
    assert dedup_faqs([], MockEmbeddingClient()) == []
