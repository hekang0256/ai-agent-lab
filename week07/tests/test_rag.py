from dataclasses import dataclass

import pytest

from llm.document import SourceDocument
from llm.mock import INSUFFICIENT_ANSWER
from llm.rag import RAGService


@dataclass
class StubResponse:
    content: str
    model: str = "stub"
    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15
    latency_ms: float = 1.0


class StubLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return StubResponse(self.content)


def test_index_and_grounded_answer_with_citation(service):
    assert service.store.count() == 14
    response = service.answer("出差打车费用怎么报销？")
    assert not response.refused
    assert "发票" in response.answer
    assert "五个工作日" in response.answer
    assert response.citation_status == "ok"
    assert response.citations[0].document_id == "P01"
    assert response.citations[0].chunk_id == "P01:chunk:000"


def test_irrelevant_question_refuses_without_calling_llm(service, llm_client):
    before = len(llm_client.calls)
    response = service.answer("今天中午吃什么？")
    assert response.refused
    assert response.refusal_reason == "no_relevant_context"
    assert response.answer == INSUFFICIENT_ANSWER
    assert response.citations == []
    assert len(llm_client.calls) == before


def test_category_filter_scopes_evidence(service):
    response = service.answer("报销发票抬头怎么开？", category="IT")
    assert response.refused
    assert all(item.metadata["category"] == "IT" for item in response.retrieved)


def test_context_contains_traceable_source_metadata(service):
    results = service.retrieve("离职当天账号权限怎么处理？", top_k=1)
    context = service.build_context(results)
    assert "document_id=P08" in context
    assert "chunk_id=P08:chunk:000" in context
    assert "policies.md" in context


def test_citation_maps_only_to_actual_retrieved_chunk(service):
    stub = StubLLM("根据第二块资料回答。[2]")
    service.llm_client = stub
    response = service.answer("报销需要哪些材料？", top_k=2)
    assert response.citation_status == "ok"
    assert response.citations[0].index == 2
    assert response.citations[0].chunk_id == response.retrieved[1].id


def test_out_of_range_citation_is_flagged_not_accepted(service):
    service.llm_client = StubLLM("这是一个带伪造来源的回答。[99]")
    response = service.answer("报销需要哪些材料？")
    assert response.citation_status == "invalid"
    assert response.invalid_citation_indexes == [99]
    assert response.citations == []


def test_missing_citation_is_visible(service):
    service.llm_client = StubLLM("没有编号的回答。")
    response = service.answer("报销需要哪些材料？")
    assert response.citation_status == "missing"
    assert response.citations == []


def test_llm_can_refuse_when_retrieved_context_is_insufficient(service):
    service.llm_client = StubLLM(INSUFFICIENT_ANSWER)
    response = service.answer("报销需要哪些材料？")
    assert response.refused
    assert response.refusal_reason == "insufficient_context"
    assert response.retrieved


def test_prompt_forbids_unguaranteed_knowledge(service):
    results = service.retrieve("年假规定", top_k=1)
    messages = service.build_messages("年假规定", results)
    assert "只能根据" in messages[0].content
    assert "不得使用资料之外的知识" in messages[0].content
    assert messages[1].content.startswith("问题：年假规定")


def test_duplicate_document_ids_are_rejected(service):
    docs = [SourceDocument("DUP", "A", "正文A"), SourceDocument("DUP", "B", "正文B")]
    with pytest.raises(ValueError, match="ID 冲突"):
        service.index_documents(docs)


def test_empty_query_and_zero_top_k_rejected(service):
    with pytest.raises(ValueError):
        service.answer("  ")
    with pytest.raises(ValueError):
        service.retrieve("报销", top_k=0)
