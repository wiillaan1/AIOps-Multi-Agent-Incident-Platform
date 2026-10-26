"""
Tests for RAG Re-ranker (Phase 2c).
"""

from __future__ import annotations

import json

import pytest

from app.agents.rag_reranker import LLMReranker
from app.nlu.intent_llm import LLMIntentResult, MockLLMProvider


def _candidates(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"kb-{i}",
            "content": f"故障案例 {i}: order-service 出现 high_latency，建议扩容。",
            "metadata": {"category": "performance"},
            "distance": 0.1 * (i + 1),
        }
        for i in range(n)
    ]


# ============================ RuleOnly 默认行为 ============================


def test_rerank_with_rule_only_keeps_original_order():
    reranker = LLMReranker()  # 默认 RuleOnly
    candidates = _candidates(5)
    result = reranker.rerank("order CPU 慢", candidates, top_k=3)
    assert result.llm_used is False
    # 原顺序：kb-0, kb-1, kb-2
    assert [i.doc_id for i in result.items] == ["kb-0", "kb-1", "kb-2"]


def test_rerank_with_rule_only_does_not_mark_changed():
    reranker = LLMReranker()
    result = reranker.rerank("x", _candidates(5), top_k=3)
    assert result.rerank_changed is False


def test_rerank_with_disabled_flag_skips_llm():
    provider = MockLLMProvider(
        response=LLMIntentResult(intent="x", confidence=0.9, reasoning=json.dumps(["kb-4", "kb-0"])),
    )
    reranker = LLMReranker(llm_provider=provider, enable=False)
    result = reranker.rerank("x", _candidates(5), top_k=3)
    assert result.llm_used is False
    # 仍然原顺序
    assert [i.doc_id for i in result.items] == ["kb-0", "kb-1", "kb-2"]


# ============================ LLM 重排 ============================


def test_rerank_uses_llm_when_enabled():
    """LLM 给出顺序后，rerank 应按 LLM 顺序返回。"""
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning=json.dumps(["kb-3", "kb-1", "kb-4"]),
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("order CPU 慢", _candidates(5), top_k=3)
    assert result.llm_used is True
    assert [i.doc_id for i in result.items] == ["kb-3", "kb-1", "kb-4"]


def test_rerank_marks_changed_when_order_differs():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning=json.dumps(["kb-4", "kb-3"]),
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(5), top_k=2)
    assert result.rerank_changed is True
    assert [i.doc_id for i in result.items] == ["kb-4", "kb-3"]


def test_rerank_no_change_when_llm_returns_same_order():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning=json.dumps(["kb-0", "kb-1", "kb-2"]),
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(5), top_k=3)
    assert result.rerank_changed is False


def test_rerank_handles_empty_candidates():
    reranker = LLMReranker()
    result = reranker.rerank("x", [], top_k=3)
    assert result.items == []
    assert result.llm_used is False


def test_rerank_respects_top_k():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning=json.dumps(["kb-4", "kb-3", "kb-2", "kb-1", "kb-0"]),
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(5), top_k=2)
    assert len(result.items) == 2


def test_rerank_fills_missing_with_original_order():
    """LLM 只返回部分 doc_id 时，未返回的按原顺序补齐。"""
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning=json.dumps(["kb-3"]),  # 只返回 1 个
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(5), top_k=3)
    # 期望：kb-3 (LLM), 然后按原顺序补充 kb-0, kb-1
    assert result.items[0].doc_id == "kb-3"
    assert result.items[1].doc_id == "kb-0"
    assert result.items[2].doc_id == "kb-1"


def test_rerank_unparseable_json_falls_back_to_original():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning="not a json",
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(5), top_k=3)
    assert result.llm_used is False
    assert result.fallback_reason == "llm_output_unparseable"
    # 原顺序保留
    assert [i.doc_id for i in result.items] == ["kb-0", "kb-1", "kb-2"]


def test_rerank_llm_timeout_falls_back():
    provider = MockLLMProvider(
        response=LLMIntentResult(intent="x", confidence=0.9, reasoning=json.dumps(["kb-3"])),
        delay_seconds=0.5,
    )
    reranker = LLMReranker(llm_provider=provider, timeout_seconds=0.1)
    result = reranker.rerank("x", _candidates(5), top_k=3)
    assert result.llm_used is False
    assert "llm_failed" in result.fallback_reason
    # 仍然返回原顺序
    assert [i.doc_id for i in result.items] == ["kb-0", "kb-1", "kb-2"]


def test_rerank_llm_exception_falls_back():
    provider = MockLLMProvider(raise_on_call=RuntimeError("rate limit"))
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(5), top_k=3)
    assert result.llm_used is False
    assert "rate limit" in result.fallback_reason


def test_rerank_records_latency():
    reranker = LLMReranker()
    result = reranker.rerank("x", _candidates(3), top_k=2)
    assert result.latency_ms >= 0


def test_rerank_assigns_rerank_score():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x", confidence=0.9,
            reasoning=json.dumps(["kb-1", "kb-0"]),
        )
    )
    reranker = LLMReranker(llm_provider=provider)
    result = reranker.rerank("x", _candidates(3), top_k=2)
    # Top-1 应得 1.0 分，Top-2 应得 0.9 分
    assert result.items[0].rerank_score == 1.0
    assert result.items[1].rerank_score == 0.9


def test_rerank_preserves_metadata():
    candidates = [
        {
            "id": "kb-0",
            "content": "x",
            "metadata": {"category": "perf", "service": "order"},
            "distance": 0.1,
        }
    ]
    reranker = LLMReranker()
    result = reranker.rerank("x", candidates, top_k=1)
    assert result.items[0].metadata == {"category": "perf", "service": "order"}


def test_rerank_total_candidates_reflects_input():
    reranker = LLMReranker()
    result = reranker.rerank("x", _candidates(7), top_k=3)
    assert result.total_candidates == 7


def test_rerank_to_dict_includes_all_fields():
    reranker = LLMReranker()
    result = reranker.rerank("x", _candidates(3), top_k=2)
    blob = result.to_dict()
    for key in ["query", "top_k", "total_candidates", "items",
                "rerank_changed", "llm_used", "latency_ms"]:
        assert key in blob
