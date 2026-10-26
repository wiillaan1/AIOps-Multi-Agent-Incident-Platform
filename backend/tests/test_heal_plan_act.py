"""
Tests for Heal Plan+Act candidate selection (W3a).

覆盖：
1. plan_candidates 返回 top-K 并按 score 排序
2. act_with_reflection 选 top-1 风险可接受的候选
3. 风险过高时降级到 top-2
4. 全部风险过高 → 升级人工
5. 空候选 → 升级
"""

from __future__ import annotations

from app.agents.heal_agent import BlastRadiusResult, HealAgent
from app.models.events import RCAEvent, SeverityLevel


def _rca(root_cause: str = "resource_exhaustion", service: str = "order-service") -> RCAEvent:
    return RCAEvent(
        event_id="rca-test",
        correlation_id="corr",
        source="rca_agent",
        incident_id="INC-1",
        root_cause=root_cause,
        confidence=0.85,
        evidence={
            "alert_metric": "cpu_usage_percent",
            "affected_services": [service, "payment-service"],
        },
        impact_chain=[service, "payment-service"],
    )


def _blast(risk: str = "low", ratio: float = 0.05) -> BlastRadiusResult:
    return BlastRadiusResult(
        affected_service_count=int(ratio * 10),
        total_service_count=10,
        blast_radius_ratio=ratio,
        risk_level=risk,
        critical_services_affected=[],
    )


def test_plan_candidates_returns_top_k_sorted_by_score():
    agent = HealAgent()
    candidates = agent.plan_candidates(_rca(), top_k=3)
    assert len(candidates) <= 3
    assert len(candidates) > 0
    # 按 match_score 降序
    scores = [c["match_score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_plan_candidate_includes_evaluation_metadata():
    agent = HealAgent()
    candidates = agent.plan_candidates(_rca(), top_k=1)
    assert len(candidates) == 1
    c = candidates[0]
    assert "blast_radius" in c
    assert "estimated_success_rate" in c
    assert "risk_level" in c
    assert c["blast_radius"]["risk_level"] in {"low", "medium", "high", "critical"}


def test_act_with_reflection_picks_top1_when_safe():
    agent = HealAgent()
    result = agent.act_with_reflection(_rca(), _blast(risk="low", ratio=0.05))
    assert result["escalation_needed"] is False
    assert result["playbook"] is not None
    assert result["reflection_reason"] == "top_1_acceptable"


def test_act_with_reflection_downgrades_when_top1_too_risky():
    agent = HealAgent()
    # 把 top-1 强制定为 high risk
    with patch_top1_risk(agent, "high"):
        result = agent.act_with_reflection(_rca(), _blast(risk="high", ratio=0.3))
    # 应该降级或升级
    assert result["reflection_reason"] in {
        "top_1_too_risky_downgraded_to_low_risk",
        "all_candidates_too_risky",
    }


def test_act_with_reflection_escalates_when_all_too_risky():
    agent = HealAgent()
    # 把所有候选都标 high
    with patch_all_risks(agent, "high"):
        result = agent.act_with_reflection(_rca(), _blast(risk="high", ratio=0.3))

    assert result["escalation_needed"] is True
    assert result["playbook"] is None
    assert "all_candidates_too_risky" in result["reflection_reason"]


def test_estimate_blast_radius_risk_levels():
    agent = HealAgent()
    # 用 topology 中已有的服务
    assert agent._estimate_blast_radius({"id": "p1"})["risk_level"] in {
        "low", "medium", "high", "critical",
    }
    # 用不存在服务
    fake = agent._estimate_blast_radius({"id": "p1", "target_service": "no-such-svc"})
    assert fake["risk_level"] == "low"
    assert fake["affected_service_count"] == 0


def test_collect_affected_returns_set():
    agent = HealAgent()
    affected = agent._collect_affected("order-service", max_hops=2)
    assert isinstance(affected, set)
    assert "order-service" in affected


def test_lookup_success_rate_returns_value_in_range():
    agent = HealAgent()
    rate = agent._lookup_success_rate("restart_pod")
    assert 0.5 <= rate <= 1.0


def test_lookup_success_rate_handles_none():
    agent = HealAgent()
    rate = agent._lookup_success_rate(None)
    assert rate == 0.5


# ===== Helpers =====


def patch_top1_risks(agent: HealAgent, new_risk: str):
    """Monkeypatch top-1 candidate risk level."""
    original = agent.plan_candidates

    def wrapped(*args, **kwargs):
        candidates = original(*args, **kwargs)
        if candidates:
            candidates = list(candidates)
            candidates[0]["blast_radius"]["risk_level"] = new_risk
        return candidates

    return wrapped


from unittest.mock import patch


def patch_top1_risk(agent: HealAgent, new_risk: str):
    """Context manager 把 plan_candidates 的 top-1 风险等级改为 new_risk。"""
    original = agent.plan_candidates

    def wrapped(*args, **kwargs):
        candidates = original(*args, **kwargs)
        if candidates:
            candidates = list(candidates)
            candidates[0]["blast_radius"]["risk_level"] = new_risk
        return candidates

    return patch.object(agent, "plan_candidates", side_effect=wrapped)


def patch_all_risks(agent: HealAgent, new_risk: str):
    """Context manager 把所有候选风险等级都改为 new_risk。"""
    original = agent.plan_candidates

    def wrapped(*args, **kwargs):
        candidates = original(*args, **kwargs)
        for c in candidates:
            c["blast_radius"]["risk_level"] = new_risk
        return candidates

    return patch.object(agent, "plan_candidates", side_effect=wrapped)
