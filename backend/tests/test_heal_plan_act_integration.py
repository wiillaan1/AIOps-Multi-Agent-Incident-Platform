"""Test HealAgent.process integrates plan_candidates + act_with_reflection."""

import asyncio
from app.agents.heal_agent import HealAgent, HealInput, BlastRadiusResult
from app.models.events import HealEvent, RCAEvent, SeverityLevel


def _rca():
    return RCAEvent(
        event_id="evt-rca",
        correlation_id="corr",
        source="rca_agent",
        incident_id="INC-H1",
        root_cause="resource_exhaustion",
        confidence=0.85,
        evidence={
            "alert_metric": "cpu_usage_percent",
            "affected_services": ["order-service"],
            "critical_services_affected": [],
        },
        impact_chain=["order-service", "payment-service"],
    )


def _heal_input():
    return HealInput(rca_event=_rca(), incident_id="INC-H1", dry_run=True)


def _ctx():
    from app.models.agent import AgentExecutionContext
    return AgentExecutionContext(incident_id="INC-H1")


def test_heal_process_includes_plan_act_selection_when_top1_acceptable():
    agent = HealAgent()
    result = asyncio.run(agent.process(_heal_input(), _ctx()))
    assert result.success
    sel = result.output_data.get("plan_act_selection")
    assert sel is not None
    assert sel["reflection_reason"] in {
        "top_1_acceptable",
        "top_1_too_risky_downgraded_to_low_risk",
        "all_candidates_too_risky",
    }
    # 原 playbook_matched 字段仍然存在
    assert "playbook_matched" in result.output_data


def test_heal_process_plan_act_selection_falls_back_on_reflection_failure():
    """若 act_with_reflection 抛异常，process 不应崩溃且回退到 _match_playbook 结果。"""
    agent = HealAgent()
    original = agent.act_with_reflection
    agent.act_with_reflection = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("reflection boom"))
    result = asyncio.run(agent.process(_heal_input(), _ctx()))
    assert result.success
    assert "playbook_matched" in result.output_data
    # plan_act_selection 字段缺失或 reason=fallback
    sel = result.output_data.get("plan_act_selection", {})
    assert sel.get("reflection_reason") in {"fallback_on_error", ""} or "playbook" in sel