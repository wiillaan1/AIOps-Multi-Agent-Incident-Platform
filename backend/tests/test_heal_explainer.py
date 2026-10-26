"""
Tests for Heal Decision Explainer (Phase 2b).
"""

from __future__ import annotations

import pytest

from app.agents.heal_explainer import HealDecisionExplanation, HealExplainer


def _chosen(
    playbook_id: str = "restart_pod",
    playbook_name: str = "重启 Pod",
    risk_level: str = "low",
    blast_ratio: float = 0.05,
    match_score: float = 0.85,
    success_rate: float = 0.85,
    affected_count: int = 1,
    total_count: int = 10,
    critical_affected: list[str] | None = None,
):
    return {
        "playbook_id": playbook_id,
        "playbook_name": playbook_name,
        "match_score": match_score,
        "blast_radius": {
            "affected_service_count": affected_count,
            "total_service_count": total_count,
            "blast_radius_ratio": blast_ratio,
            "critical_services_affected": critical_affected or [],
            "risk_level": risk_level,
        },
        "estimated_success_rate": success_rate,
        "risk_level": risk_level,
    }


def _act_result(chosen=None, candidates=None, reflection_reason="top_1_acceptable", escalation=False):
    return {
        "playbook": chosen,
        "candidates": candidates or ([chosen] if chosen else []),
        "reflection_reason": reflection_reason,
        "escalation_needed": escalation,
    }


# ============================ 模板方式 ============================


def test_explain_includes_summary_with_playbook_name():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen()))
    assert "重启 Pod" in expl.summary
    assert "low" in expl.summary
    assert "5%" in expl.summary  # 爆炸半径 0.05 → 5%


def test_explain_describes_match_score_and_success_rate():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen(match_score=0.92, success_rate=0.88)))
    assert "0.92" in expl.why_this_playbook
    assert "88%" in expl.why_this_playbook


def test_explain_risk_level_explanation_low():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen(risk_level="low")))
    assert "L0" in expl.why_this_risk_level
    assert "自动执行" in expl.why_this_risk_level


def test_explain_risk_level_explanation_medium():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen(risk_level="medium")))
    assert "L1" in expl.why_this_risk_level
    assert "oncall" in expl.why_this_risk_level.lower() or "确认" in expl.why_this_risk_level


def test_explain_risk_level_explanation_high():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen(risk_level="high")))
    assert "L2" in expl.why_this_risk_level
    assert "审批" in expl.why_this_risk_level


def test_explain_risk_level_explanation_critical():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen(risk_level="critical")))
    assert "L2" in expl.why_this_risk_level
    assert "SRE" in expl.why_this_risk_level


def test_explain_blast_radius_summary_includes_count():
    e = HealExplainer()
    expl = e.explain(
        _act_result(chosen=_chosen(affected_count=3, total_count=10, blast_ratio=0.3))
    )
    assert "3 / 10" in expl.blast_radius_summary
    assert "30%" in expl.blast_radius_summary


def test_explain_blast_radius_lists_critical_services():
    e = HealExplainer()
    expl = e.explain(
        _act_result(chosen=_chosen(critical_affected=["mysql-primary", "kafka"]))
    )
    assert "mysql-primary" in expl.blast_radius_summary
    assert "kafka" in expl.blast_radius_summary


def test_explain_rollback_summary_mentions_dry_run():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen()))
    assert "dry-run" in expl.rollback_summary
    assert "回滚" in expl.rollback_summary


def test_explain_alternatives_excludes_chosen():
    chosen = _chosen(playbook_id="A")
    other = _chosen(playbook_id="B")
    third = _chosen(playbook_id="C")
    e = HealExplainer()
    expl = e.explain(_act_result(
        chosen=chosen,
        candidates=[chosen, other, third],
    ))
    alt_ids = [a["playbook_id"] for a in expl.alternatives]
    assert "A" not in alt_ids
    assert "B" in alt_ids
    assert "C" in alt_ids


def test_explain_includes_reflection_note_when_downgraded():
    e = HealExplainer()
    expl = e.explain(
        _act_result(
            chosen=_chosen(playbook_id="B"),
            candidates=[_chosen(playbook_id="A"), _chosen(playbook_id="B")],
            reflection_reason="top_1_too_risky_downgraded_to_low_risk",
        )
    )
    assert "top_1_too_risky" in expl.reflection_note
    assert "降级" in expl.why_this_playbook or "调整" in expl.why_this_playbook


def test_explain_escalation_when_no_chosen_playbook():
    e = HealExplainer()
    candidates = [_chosen(playbook_id="X", risk_level="high"),
                  _chosen(playbook_id="Y", risk_level="critical")]
    expl = e.explain(_act_result(
        chosen=None, candidates=candidates,
        reflection_reason="all_candidates_too_risky", escalation=True,
    ))
    assert "无可用" in expl.summary
    assert "无可用 playbook" in expl.why_this_playbook
    assert expl.rollback_summary == "无（未执行任何自愈）"


def test_explain_default_uses_template_not_llm():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen()))
    assert expl.llm_used is False


def test_explain_to_dict_includes_all_fields():
    e = HealExplainer()
    expl = e.explain(_act_result(chosen=_chosen()))
    blob = expl.to_dict()
    for key in [
        "summary", "why_this_playbook", "why_this_risk_level",
        "blast_radius_summary", "rollback_summary", "alternatives",
        "reflection_note", "llm_used",
    ]:
        assert key in blob


# ============================ LLM 润色 ============================


def test_llm_polish_updates_summary_when_high_confidence():
    from app.nlu.intent_llm import LLMIntentResult, MockLLMProvider

    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning="LLM 润色的运维友好总结：选了最稳妥的方案。",
        )
    )
    e = HealExplainer(llm_provider=provider)
    expl = e.explain(_act_result(chosen=_chosen()))
    assert expl.llm_used is True
    assert "LLM 润色" in expl.summary


def test_llm_failure_falls_back_to_template():
    from app.nlu.intent_llm import MockLLMProvider

    provider = MockLLMProvider(raise_on_call=RuntimeError("rate_limit"))
    e = HealExplainer(llm_provider=provider)
    expl = e.explain(_act_result(chosen=_chosen()))
    # 失败时回退
    assert expl.llm_used is False
    assert "rate_limit" in expl.fallback_reason
    # 模板仍然有效
    assert expl.summary  # 非空
    assert expl.why_this_playbook  # 非空


def test_explain_summary_reflects_match_position():
    """备选 top-1 时 summary 包含 'top-1'，top-2 时包含 'top-2'。"""
    e = HealExplainer()
    chosen = _chosen(playbook_id="B")
    expl = e.explain(_act_result(
        chosen=chosen,
        candidates=[_chosen(playbook_id="A"), chosen],
    ))
    # candidates.index(chosen) == 1 → "top-2" 出现在 why_this_playbook
    assert "top-2" in expl.why_this_playbook
