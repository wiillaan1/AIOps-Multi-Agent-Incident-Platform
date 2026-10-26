"""Unit tests for verify_rca_against_metric / verify_heal_and_try_plan_b."""

from app.api.routes import (
    verify_rca_against_metric,
    verify_heal_and_try_plan_b,
)
from app.models.events import HealEvent, RCAEvent, SeverityLevel


def _rca(root="resource_exhaustion", confidence=0.55):
    return RCAEvent(
        event_id="rca-1",
        correlation_id="corr",
        source="rca_agent",
        incident_id="INC-1",
        root_cause=root,
        confidence=confidence,
        evidence={"alert_metric": "cpu_usage_percent", "affected_services": ["order-service"]},
        impact_chain=["order-service"],
    )


def test_verify_rca_match_when_predicted_high_actual_high():
    out = verify_rca_against_metric(
        rca=_rca(), pre_value=92.0, threshold=80.0,
    )
    assert out["match_status"] == "match"
    assert out["should_rerun"] is False


def test_verify_rca_mismatch_when_predicted_high_actual_low():
    out = verify_rca_against_metric(
        rca=_rca(), pre_value=20.0, threshold=80.0,
    )
    assert out["match_status"] == "mismatch"
    assert out["should_rerun"] is True
    assert out["suggested_context"]["evidence_boost"] == "log"


def test_verify_rca_unavailable_when_pre_value_none():
    out = verify_rca_against_metric(rca=_rca(), pre_value=None, threshold=80.0)
    assert out["match_status"] == "unavailable"
    assert out["should_rerun"] is False


def test_verify_rca_unknown_root_never_mismatches():
    out = verify_rca_against_metric(rca=_rca(root="unknown"), pre_value=20.0, threshold=80.0)
    assert out["match_status"] in {"match", "unavailable"}


def test_verify_heal_recovered_when_post_value_below_threshold():
    heal = HealEvent(event_id="h", correlation_id="c", source="heal_agent",
                     incident_id="INC-1", action="scale_up", action_category="x",
                     level="L0", target_resource="order-service", dry_run=True,
                     dry_run_result={"all_executable": True})
    out = verify_heal_and_try_plan_b(
        heal_event=heal, post_value=60.0, threshold=80.0,
        incident_severity=SeverityLevel.HIGH,
    )
    assert out["recovered"] is True
    assert out["plan_b_triggered"] is False


def test_verify_heal_triggers_plan_b_when_not_recovered_high_severity():
    heal = HealEvent(event_id="h", correlation_id="c", source="heal_agent",
                     incident_id="INC-1", action="scale_up", action_category="x",
                     level="L0", target_resource="order-service", dry_run=True,
                     dry_run_result={"all_executable": True})
    out = verify_heal_and_try_plan_b(
        heal_event=heal, post_value=95.0, threshold=80.0,
        incident_severity=SeverityLevel.CRITICAL,
    )
    assert out["recovered"] is False
    # plan_b 候选将由 caller 决定；本函数返回 plan_b_eligible=True
    assert out["plan_b_eligible"] is True


def test_verify_heal_skips_plan_b_for_low_severity():
    heal = HealEvent(event_id="h", correlation_id="c", source="heal_agent",
                     incident_id="INC-1", action="scale_up", action_category="x",
                     level="L0", target_resource="order-service", dry_run=True,
                     dry_run_result={"all_executable": True})
    out = verify_heal_and_try_plan_b(
        heal_event=heal, post_value=95.0, threshold=80.0,
        incident_severity=SeverityLevel.LOW,
    )
    assert out["plan_b_eligible"] is False
