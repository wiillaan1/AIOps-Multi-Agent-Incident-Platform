"""TDD coverage for the bounded, evidence-first investigation loop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.agents.base import AgentResult
from app.models.events import AlertEvent, RCAEvent, SeverityLevel
from app.models.incident import Incident
from app.models.investigation import (
    EvidenceSource,
    HealExecutionReceipt,
    InvestigationState,
    RecoveryOutcome,
    RootCauseVerificationProfile,
    VerificationStatus,
)
from app.services.assertion_evaluator import QueryResult
from app.services.investigation_store import InvestigationStore
from app.services.recovery_verifier import RecoverySample


NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _profile(*, root_cause: str = "resource_exhaustion", manual_only: bool = False):
    return RootCauseVerificationProfile.model_validate(
        {
            "root_cause": root_cause,
            "version": "1.0",
            "manual_only": manual_only,
            "assertions": [] if manual_only else [
                {
                    "id": "cpu-high",
                    "source": "prometheus",
                    "query_template": 'cpu{service="{{ service }}",target="{{ target }}"}',
                    "evidence_level": "direct",
                    "weight": 0.8,
                    "expectation": {"operator": "greater_than", "threshold": 80},
                },
                {
                    "id": "oom-log",
                    "source": "loki",
                    "query_template": '{service="{{ service }}",target="{{ target }}"} |= "oom"',
                    "evidence_level": "supporting",
                    "weight": 0.2,
                    "expectation": {"operator": "min_hits", "threshold": 1},
                },
            ],
            "counter_evidence": [] if manual_only else [
                {
                    "id": "healthy-counter",
                    "source": "prometheus",
                    "query_template": 'cpu{service="{{ service }}",target="{{ target }}"}',
                    "weight": 0.5,
                    "expectation": {"operator": "less_than", "threshold": 10},
                }
            ],
            "decision": {
                "minimum_score": 0.8 if not manual_only else 0.0,
                "require_direct_evidence": not manual_only,
                "maximum_counter_evidence": 0.1,
            },
            "next_strategies": {
                "inconclusive": ["expand_window"],
                "contradicted": ["inspect_dependency"],
            },
            "recovery": {
                "settle_seconds": 0,
                "observation_seconds": 300,
                "consecutive_samples": 2,
            },
        }
    )


class FakeProfiles:
    def __init__(self, profiles: dict[str, RootCauseVerificationProfile]) -> None:
        self.profiles = profiles

    def get(self, root_cause: str):
        profile = self.profiles.get(root_cause)
        return profile.model_copy(deep=True) if profile else None


class FakeRCA:
    def __init__(self, causes: list[str]) -> None:
        self.causes = iter(causes)
        self.inputs = []

    async def execute(self, input_data, context=None):
        self.inputs.append(input_data)
        root_cause = next(self.causes)
        event = RCAEvent(
            incident_id=input_data.incident_id,
            root_cause=root_cause,
            confidence=0.9,
            evidence={"alert_metric": input_data.alert.metric, "affected_services": [input_data.alert.service]},
        )
        return AgentResult.success_result("fake_rca", {"rca_event": event.model_dump()})


class FakeHeal:
    def __init__(self, candidates: list[dict] | None = None) -> None:
        self.inputs = []
        self._candidates = candidates or [
            {"playbook_id": "restart-orders", "playbook_name": "Restart orders"},
            {"playbook_id": "scale-orders", "playbook_name": "Scale orders"},
        ]

    async def execute(self, input_data, context=None):
        self.inputs.append(input_data)
        return AgentResult.success_result(
            "fake_heal",
            {
                "heal_event": {
                    "incident_id": input_data.incident_id,
                    "action_id": "dry-run-action",
                    "action": "restart",
                    "action_category": "restart-orders",
                    "target_resource": "orders-api",
                    "dry_run": input_data.dry_run,
                    "dry_run_result": {"all_executable": True},
                    "status": "pending",
                }
            },
        )

    def plan_candidates(self, rca_event, top_k=3):
        return self._candidates[:top_k]


def _incident() -> Incident:
    alert = AlertEvent(
        service="orders",
        metric="cpu_usage_percent",
        value=95,
        threshold=80,
        severity=SeverityLevel.HIGH,
        labels={"target": "orders-api", "environment": "prod"},
    )
    return Incident.from_alert(alert)


def _queries(*, contradicted: bool = False):
    async def metric_query(query: str, **kwargs):
        return QueryResult(
            source=EvidenceSource.PROMETHEUS,
            available=True,
            value=5 if contradicted else 95,
            hits=0,
            summary={"query": query},
        )

    async def log_query(query: str, **kwargs):
        return QueryResult(
            source=EvidenceSource.LOKI,
            available=True,
            value=None,
            hits=2,
            summary={"query": query},
        )

    return metric_query, log_query


def _engine(*, profiles, rca, heal, contradicted=False):
    from app.services.investigation_loop_engine import InvestigationLoopEngine

    metric_query, log_query = _queries(contradicted=contradicted)
    return InvestigationLoopEngine(
        store=InvestigationStore(),
        profiles=FakeProfiles(profiles),
        rca_agent=rca,
        heal_agent=heal,
        metric_query=metric_query,
        log_query=log_query,
        clock=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_confirmed_profile_records_rendered_evidence_and_only_requests_dry_run():
    rca = FakeRCA(["resource_exhaustion"])
    heal = FakeHeal()
    engine = _engine(profiles={"resource_exhaustion": _profile()}, rca=rca, heal=heal)

    run = await engine.investigate(_incident())

    assert run.state is InvestigationState.AWAITING_EXECUTION
    assert run.verification_status is VerificationStatus.CONFIRMED
    assert run.rca_round == 1
    assert run.heal_attempt == 1
    assert len(run.evidence_refs) == 3
    assert len(heal.inputs) == 1
    assert heal.inputs[0].dry_run is True
    assert run.audit_summary["dry_run"] == {
        "action_id": "dry-run-action",
        "playbook_id": "restart-orders",
        "target_resource": "orders-api",
        "executable": True,
    }
    assert run.state is not InvestigationState.RECOVERED


@pytest.mark.asyncio
async def test_missing_or_manual_profile_escalates_without_query_or_heal():
    rca = FakeRCA(["resource_exhaustion"])
    heal = FakeHeal()
    engine = _engine(profiles={}, rca=rca, heal=heal)

    run = await engine.investigate(_incident())

    assert run.state is InvestigationState.MANUAL_ESCALATION
    assert run.rca_round == 1
    assert run.heal_attempt == 0
    assert run.audit_summary["manual_escalation_reason"] == "profile_missing"
    assert heal.inputs == []


@pytest.mark.asyncio
async def test_contradicted_profile_replans_with_rejected_root_context_then_escalates_at_round_three():
    rca = FakeRCA(["resource_exhaustion", "resource_exhaustion", "resource_exhaustion"])
    heal = FakeHeal()
    engine = _engine(
        profiles={"resource_exhaustion": _profile()}, rca=rca, heal=heal, contradicted=True
    )

    run = await engine.investigate(_incident())

    assert run.state is InvestigationState.MANUAL_ESCALATION
    assert run.verification_status is VerificationStatus.CONTRADICTED
    assert run.rca_round == 3
    assert run.heal_attempt == 0
    assert len(run.hypotheses) == 3
    assert rca.inputs[1].reflection_context["rejected_root_causes"] == ["resource_exhaustion"]
    assert rca.inputs[2].reflection_context["rejected_root_causes"] == ["resource_exhaustion"]
    assert heal.inputs == []


@pytest.mark.asyncio
async def test_failed_or_rejected_receipt_never_starts_recovery_observation():
    engine = _engine(
        profiles={"resource_exhaustion": _profile()},
        rca=FakeRCA(["resource_exhaustion"]),
        heal=FakeHeal(),
    )
    incident = _incident()
    await engine.investigate(incident)
    receipt = HealExecutionReceipt(
        incident_id=incident.incident_id,
        action_id="dry-run-action",
        idempotency_key="failed-receipt",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status="failed",
        received_at=NOW,
    )

    run = await engine.accept_execution_receipt(incident, receipt)

    assert run.state is InvestigationState.HEAL_FAILED
    assert run.receipts[-1].status == "failed"
    assert run.recovery_verifications == []
    assert engine.suggest_plan_b(incident)["playbook_id"] == "scale-orders"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("playbook_id", "rollback-orders"),
        ("target_resource", "payments-api"),
        ("action_id", "other-dry-run-action"),
    ],
)
async def test_mismatched_succeeded_receipt_never_starts_recovery_observation(field, wrong_value):
    engine = _engine(
        profiles={"resource_exhaustion": _profile()},
        rca=FakeRCA(["resource_exhaustion"]),
        heal=FakeHeal(),
    )
    incident = _incident()
    await engine.investigate(incident)
    receipt_data = {
        "incident_id": incident.incident_id,
        "action_id": "dry-run-action",
        "idempotency_key": f"wrong-{field}",
        "playbook_id": "restart-orders",
        "target_resource": "orders-api",
        "status": "succeeded",
        "received_at": NOW,
        "completed_at": NOW,
    }
    receipt_data[field] = wrong_value

    run = await engine.accept_execution_receipt(incident, HealExecutionReceipt(**receipt_data))

    assert run.state is InvestigationState.HEAL_FAILED
    assert run.audit_summary["receipt_rejected_reason"] == f"dry_run_{field}_mismatch"
    assert run.receipts == []
    assert run.recovery_verifications == []


@pytest.mark.asyncio
async def test_recovery_accepts_only_bound_success_receipt_allows_one_extra_observation_and_never_executes_plan_b():
    engine = _engine(
        profiles={"resource_exhaustion": _profile()},
        rca=FakeRCA(["resource_exhaustion"]),
        heal=FakeHeal(),
    )
    incident = _incident()
    await engine.investigate(incident)
    receipt = HealExecutionReceipt(
        incident_id=incident.incident_id,
        action_id="dry-run-action",
        idempotency_key="succeeded-receipt",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status="succeeded",
        received_at=NOW,
        completed_at=NOW,
    )
    run = await engine.accept_execution_receipt(incident, receipt)
    assert run.state is InvestigationState.RECOVERY_OBSERVING

    degraded = [
        RecoverySample(NOW + timedelta(seconds=1), 40, 2, True, True),
        RecoverySample(NOW + timedelta(seconds=2), 40, 2, True, True),
    ]
    run = await engine.verify_recovery(incident, degraded)
    assert run.recovery_verifications[-1].outcome is RecoveryOutcome.DEGRADED
    assert run.state is InvestigationState.RECOVERY_OBSERVING
    assert run.audit_summary["extra_observation_used"] is True

    failed = [
        RecoverySample(NOW + timedelta(seconds=3), 95, 0, True, True),
        RecoverySample(NOW + timedelta(seconds=4), 95, 0, True, True),
    ]
    run = await engine.verify_recovery(incident, failed)
    assert run.recovery_verifications[-1].outcome is RecoveryOutcome.FAILED
    assert run.state is InvestigationState.HEAL_FAILED
    assert run.audit_summary["plan_b_suggestion"]["playbook_id"] == "scale-orders"
    assert run.heal_attempt == 1
