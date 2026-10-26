"""
Tests for RCA Natural Language Explainer (Phase 2a).
"""

from __future__ import annotations

import pytest

from app.agents.rca_explainer import RCAExplainer
from app.models.events import RCAEvent


def _rca(
    root_cause: str = "resource_exhaustion",
    confidence: float = 0.78,
    impact_chain: list[str] | None = None,
    evidence: dict | None = None,
    actions: list[str] | None = None,
) -> RCAEvent:
    return RCAEvent(
        event_id="rca-test",
        correlation_id="corr",
        source="rca_agent",
        incident_id="INC-1",
        root_cause=root_cause,
        confidence=confidence,
        impact_chain=impact_chain or ["order-service", "payment-service"],
        evidence=evidence or {},
        recommended_actions=actions or ["scale_up", "check_resource_limits"],
    )


# ============================ 模板方式 ============================


def test_explain_builds_summary_with_root_cause_and_confidence():
    explainer = RCAExplainer()
    rca = _rca(root_cause="resource_exhaustion", confidence=0.78)
    expl = explainer.explain(rca)
    assert "resource_exhaustion" in expl.summary
    assert "78" in expl.summary  # 78%


def test_explain_includes_root_cause_section():
    explainer = RCAExplainer()
    rca = _rca(root_cause="network_issue", confidence=0.85)
    expl = explainer.explain(rca)
    headings = [s.heading for s in expl.sections]
    assert "根因结论" in headings


def test_explain_evidence_count_reflects_references():
    explainer = RCAExplainer()
    rca = _rca(
        evidence={
            "bayesian_top_causes": [{"cause": "rc", "posterior": 0.5}],
            "rag_matches": [{"id": "kb-1", "category": "perf"}],
            "log_evidence": {"available": True, "entries_count": 3, "boosted_cause": "rc"},
        }
    )
    expl = explainer.explain(rca)
    assert expl.evidence_count >= 3  # 至少 3 类证据


def test_explain_marks_low_confidence_warning():
    explainer = RCAExplainer()
    rca = _rca(confidence=0.3)
    expl = explainer.explain(rca)
    assert any(s.heading == "⚠️ 证据不足" for s in expl.sections)


def test_explain_no_low_confidence_warning_when_high():
    explainer = RCAExplainer()
    rca = _rca(confidence=0.85)
    expl = explainer.explain(rca)
    assert all(s.heading != "⚠️ 证据不足" for s in expl.sections)


def test_explain_includes_recommended_actions_section():
    explainer = RCAExplainer()
    rca = _rca(actions=["scale_up", "check_cpu", "restart_pod"])
    expl = explainer.explain(rca)
    action_section = next((s for s in expl.sections if s.heading == "建议操作"), None)
    assert action_section is not None
    assert "scale_up" in action_section.body
    assert "check_cpu" in action_section.body


def test_explain_includes_impact_chain_section():
    explainer = RCAExplainer()
    rca = _rca(impact_chain=["a", "b", "c", "d", "e", "f", "g"])
    expl = explainer.explain(rca)
    impact = next((s for s in expl.sections if s.heading == "影响链路"), None)
    assert impact is not None
    assert "a" in impact.body
    # 超过 5 个会显示"等 N 个"
    assert "等 2 个" in impact.body or "g" in impact.body


def test_explain_handles_log_evidence_corroboration():
    explainer = RCAExplainer()
    rca = _rca(
        evidence={
            "log_evidence": {
                "available": True,
                "entries_count": 5,
                "boosted_cause": "resource_exhaustion",
                "corroborates_bayesian": True,
                "diverges_from_bayesian": False,
            }
        }
    )
    expl = explainer.explain(rca)
    evidence_section = next((s for s in expl.sections if s.heading == "证据链"), None)
    assert evidence_section is not None
    assert "佐证" in evidence_section.body


def test_explain_handles_log_evidence_divergence():
    explainer = RCAExplainer()
    rca = _rca(
        evidence={
            "log_evidence": {
                "available": True,
                "entries_count": 5,
                "boosted_cause": "code_bug",
                "corroborates_bayesian": False,
                "diverges_from_bayesian": True,
            }
        }
    )
    expl = explainer.explain(rca)
    evidence_section = next((s for s in expl.sections if s.heading == "证据链"), None)
    assert evidence_section is not None
    assert "分歧" in evidence_section.body


def test_explain_handles_empty_evidence_gracefully():
    explainer = RCAExplainer()
    rca = _rca(evidence={})
    expl = explainer.explain(rca)
    # 仍应输出根因 + 影响 + 操作 + 低置信度警告
    assert "根因结论" in [s.heading for s in expl.sections]
    assert "影响链路" in [s.heading for s in expl.sections]


def test_explain_marks_confidence_level():
    explainer = RCAExplainer()
    rca_high = _rca(confidence=0.95)
    rca_low = _rca(confidence=0.45)
    assert explainer.explain(rca_high).confidence_level == "高"
    assert explainer.explain(rca_low).confidence_level == "低"


def test_explain_default_uses_template_not_llm():
    explainer = RCAExplainer()  # 默认 RuleOnly
    rca = _rca(confidence=0.85)
    expl = explainer.explain(rca)
    assert expl.llm_used is False


def test_explain_to_dict_includes_all_fields():
    explainer = RCAExplainer()
    rca = _rca(confidence=0.78)
    expl = explainer.explain(rca)
    blob = expl.to_dict()
    assert "summary" in blob
    assert "sections" in blob
    assert "evidence_count" in blob
    assert "confidence_level" in blob
    assert "llm_used" in blob


def test_explain_each_section_has_evidence_refs():
    explainer = RCAExplainer()
    rca = _rca(
        evidence={
            "bayesian_top_causes": [{"cause": "rc", "posterior": 0.5}],
            "rag_matches": [{"id": "kb-1", "category": "perf"}],
        }
    )
    expl = explainer.explain(rca)
    # 证据链 / 根因结论 / 影响链路 / 建议操作 都有 refs
    for section in expl.sections:
        assert isinstance(section.evidence_refs, list)


# ============================ LLM 润色 ============================


def test_llm_polish_uses_llm_when_high_confidence():
    from app.nlu.intent_llm import LLMIntentResult, MockLLMProvider

    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning="LLM 生成的简洁总结：根因是 OOM，需要扩容。",
        )
    )
    explainer = RCAExplainer(llm_provider=provider)
    # 关键：evidence 非空 + confidence >= 0.7 才调 LLM
    rca = _rca(
        confidence=0.85,
        evidence={"bayesian_top_causes": [{"cause": "rc", "posterior": 0.5}]},
    )
    expl = explainer.explain(rca)
    assert expl.llm_used is True
    assert "LLM 生成" in expl.summary


def test_llm_skipped_when_low_confidence():
    from app.nlu.intent_llm import MockLLMProvider

    provider = MockLLMProvider()  # 即使注入 provider，低置信度不调
    explainer = RCAExplainer(llm_provider=provider)
    rca = _rca(
        confidence=0.3,
        evidence={"bayesian_top_causes": [{"cause": "rc", "posterior": 0.5}]},
    )
    expl = explainer.explain(rca)
    assert expl.llm_used is False


def test_llm_skipped_when_evidence_empty():
    """evidence 为空（falsy）时不调 LLM（防幻觉）。"""
    from app.nlu.intent_llm import MockLLMProvider

    provider = MockLLMProvider()
    explainer = RCAExplainer(llm_provider=provider)
    rca = _rca(confidence=0.85, evidence={})  # 空 evidence
    expl = explainer.explain(rca)
    assert expl.llm_used is False


def test_llm_call_failure_falls_back_to_template():
    from app.nlu.intent_llm import MockLLMProvider

    provider = MockLLMProvider(raise_on_call=RuntimeError("network"))
    explainer = RCAExplainer(llm_provider=provider)
    rca = _rca(
        confidence=0.85,
        evidence={"bayesian_top_causes": [{"cause": "rc", "posterior": 0.5}]},
    )
    expl = explainer.explain(rca)
    # 失败时回退到模板（仍可用）
    assert expl.llm_used is False
    assert "根因结论" in [s.heading for s in expl.sections]
    assert "network" in expl.fallback_reason
