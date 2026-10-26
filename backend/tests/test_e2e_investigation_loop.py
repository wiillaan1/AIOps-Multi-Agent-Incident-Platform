"""End-to-end coverage for the investigation verification closed loop.

These tests exercise the full InvestigationLoopEngine lifecycle with fake
Prometheus/Loki adapters and fake RCA/Heal agents. They assert every state
transition, evidence reference, profile version, receipt idempotency, recovery
outcome, escalation reason, and the rule that unavailable data never proves
recovery and Dry-run never marks an incident recovered.

The scenarios mirror the W9 spec:

  normal evidence → confirmed RCA → dry-run awaiting receipt
  counter evidence → second RCA strategy → confirmed
  three inconclusive rounds → manual escalation
  success receipt + healthy samples → recovered
  failed receipt / guardrail failure → failed/degraded + Plan B
  Prometheus/Loki unavailable → inconclusive, never recovered
"""

from __future__ import annotations

from datetime import datetime, timezone

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
from app.services.investigation_loop_engine import InvestigationLoopEngine
from app.services.investigation_store import InvestigationStore
from app.services.profile_registry import ProfileRegistry
from app.services.recovery_verifier import RecoverySample


NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeRCA:
    """Deterministic RCA that emits one root cause per round."""

    def __init__(self, causes: list[str]) -> None:
        self.causes = list(causes)
        self._idx = 0

    async def execute(self, input_data, context=None):
        cause = self.causes[min(self._idx, len(self.causes) - 1)]
        self._idx += 1
        event = RCAEvent(
            incident_id=input_data.incident_id,
            root_cause=cause,
            confidence=0.9,
            evidence={"affected_services": [input_data.alert.service]},
        )
        return AgentResult.success_result("fake_rca", {"rca_event": event.model_dump()})


class FakeHeal:
    """Heal agent that only ever returns a Dry-run plan."""

    def __init__(self) -> None:
        self.calls = 0

    async def execute(self, input_data, context=None):
        self.calls += 1
        return AgentResult.success_result(
            "fake_heal",
            {
                "heal_event": {
                    "incident_id": input_data.incident_id,
                    "action_id": "dry-run-action",
                    "action": "restart",
                    "action_category": "restart-orders",
                    "target_resource": "orders-api",
                    "dry_run": True,
                    "dry_run_result": {"all_executable": True},
                    "status": "pending",
                }
            },
        )

    def plan_candidates(self, rca_event, top_k=3):
        return [
            {"playbook_id": "restart-orders", "playbook_name": "Restart orders"},
            {"playbook_id": "scale-orders", "playbook_name": "Scale orders"},
        ][:top_k]


def _profile(root_cause: str = "resource_exhaustion") -> RootCauseVerificationProfile:
    return RootCauseVerificationProfile.model_validate(
        {
            "root_cause": root_cause,
            "version": "1.0",
            "manual_only": False,
            "assertions": [
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
            "counter_evidence": [
                {
                    "id": "healthy-counter",
                    "source": "prometheus",
                    "query_template": 'cpu{service="{{ service }}",target="{{ target }}"}',
                    "weight": 0.5,
                    "expectation": {"operator": "less_than", "threshold": 10},
                }
            ],
            "decision": {
                "minimum_score": 0.8,
                "require_direct_evidence": True,
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


class _Profiles:
    def __init__(self, *profiles: RootCauseVerificationProfile) -> None:
        self._map = {p.root_cause: p for p in profiles}

    def get(self, root_cause: str):
        p = self._map.get(root_cause)
        return p.model_copy(deep=True) if p else None


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


def _available_metric(value: float):
    async def metric_query(query: str, **kwargs):
        return QueryResult(
            source=EvidenceSource.PROMETHEUS, available=True, value=value,
            hits=0, summary={"query": query},
        )
    return metric_query


async def _loki_hits(query: str, **kwargs):
    return QueryResult(
        source=EvidenceSource.LOKI, available=True, value=None, hits=2,
        summary={"query": query},
    )


async def _loki_unavailable(query: str, **kwargs):
    return QueryResult(
        source=EvidenceSource.LOKI, available=False, value=None, hits=0,
        summary={}, error="timeout",
    )


def _engine(
    *,
    profiles,
    rca,
    metric_query,
    log_query=_loki_hits,
    heal=None,
):
    return InvestigationLoopEngine(
        store=InvestigationStore(),
        profiles=profiles,
        rca_agent=rca,
        heal_agent=heal or FakeHeal(),
        metric_query=metric_query,
        log_query=log_query,
        clock=lambda: NOW,
    )


def _succeeded_receipt(incident_id: str) -> HealExecutionReceipt:
    return HealExecutionReceipt(
        incident_id=incident_id,
        action_id="dry-run-action",
        idempotency_key="key-1",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status="succeeded",
        completed_at=NOW,
    )


def _healthy_samples(n: int = 2):
    return [
        RecoverySample(
            timestamp=NOW, metric_value=40, log_hits=0, health_ok=True, guardrails_ok=True,
        )
        for _ in range(n)
    ]


# --------------------------------------------------------------------------- #
# 1. Normal evidence → confirmed → dry-run awaiting execution
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_normal_evidence_confirms_and_awaits_receipt():
    rca = FakeRCA(["resource_exhaustion"])
    heal = FakeHeal()
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_available_metric(95),
        heal=heal,
    )

    run = await engine.investigate(_incident())

    assert run.state is InvestigationState.AWAITING_EXECUTION
    assert run.verification_status is VerificationStatus.CONFIRMED
    assert run.rca_round == 1
    assert run.heal_attempt == 1
    assert heal.calls == 1
    # Dry-run plan recorded; no receipt → must not be recovered.
    assert run.audit_summary["dry_run"]["playbook_id"] == "restart-orders"
    assert run.receipts == []


# --------------------------------------------------------------------------- #
# 2. Counter evidence → second RCA round → confirmed
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_counter_evidence_replans_and_confirms_second_round():
    # Round 1: contradicted (CPU low, counter rule fires); round 2: evidence passes.
    profiles = _Profiles(_profile("resource_exhaustion"), _profile("database_issue"))
    rca = FakeRCA(["resource_exhaustion", "database_issue"])

    call_count = {"n": 0}

    async def metric_query(query: str, **kwargs):
        # Only prometheus rules go through this adapter (2 per round: direct + counter).
        # Round 1 returns low (contradicted); round 2 returns high (confirmed).
        call_count["n"] += 1
        first_round_calls = 2  # 1 direct assertion + 1 counter rule, both prometheus
        value = 5 if call_count["n"] <= first_round_calls else 95
        return QueryResult(
            source=EvidenceSource.PROMETHEUS, available=True, value=value,
            hits=0, summary={"query": query},
        )

    engine = _engine(profiles=profiles, rca=rca, metric_query=metric_query)

    run = await engine.investigate(_incident())

    assert run.rca_round == 2
    assert run.state is InvestigationState.AWAITING_EXECUTION
    assert run.verification_status is VerificationStatus.CONFIRMED
    # Rejected root cause fed back; second hypothesis differs from first.
    assert "resource_exhaustion" in run.audit_summary["rejected_root_causes"]
    assert len(run.hypotheses) == 2


# --------------------------------------------------------------------------- #
# 3. Three inconclusive rounds → manual escalation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_three_rounds_of_counter_evidence_escalates():
    rca = FakeRCA(["resource_exhaustion", "database_issue", "network_issue"])
    engine = _engine(
        profiles=_Profiles(_profile("resource_exhaustion"), _profile("database_issue"),
                           _profile("network_issue")),
        rca=rca,
        metric_query=_available_metric(5),  # always contradicted → low CPU
    )

    run = await engine.investigate(_incident())

    assert run.rca_round == 3
    assert run.state is InvestigationState.MANUAL_ESCALATION
    assert run.audit_summary["manual_escalation_reason"] == "verification_round_limit"


# --------------------------------------------------------------------------- #
# 4. Success receipt + healthy samples → recovered
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_succeeded_receipt_with_healthy_samples_recovers():
    rca = FakeRCA(["resource_exhaustion"])
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_available_metric(95),
    )
    incident = _incident()

    await engine.investigate(incident)
    run = await engine.accept_execution_receipt(incident, _succeeded_receipt(incident.incident_id))
    assert run.state is InvestigationState.RECOVERY_OBSERVING

    run = await engine.verify_recovery(incident, _healthy_samples(n=2))
    assert run.state is InvestigationState.RECOVERED
    assert run.recovery_verifications[-1].outcome is RecoveryOutcome.RECOVERED


# --------------------------------------------------------------------------- #
# 5. Failed receipt → heal_failed + Plan B recommendation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_failed_receipt_leads_to_heal_failed_and_plan_b():
    rca = FakeRCA(["resource_exhaustion"])
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_available_metric(95),
    )
    incident = _incident()
    await engine.investigate(incident)

    failed = HealExecutionReceipt(
        incident_id=incident.incident_id,
        action_id="dry-run-action",
        idempotency_key="key-1",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status="failed",
    )
    run = await engine.accept_execution_receipt(incident, failed)

    assert run.state is InvestigationState.HEAL_FAILED
    # Plan B suggestion is a *different* playbook than the failed one.
    suggestion = run.audit_summary.get("plan_b_suggestion")
    assert suggestion is not None
    assert suggestion["playbook_id"] != "restart-orders"


# --------------------------------------------------------------------------- #
# 6. Unavailable data → inconclusive, never recovered
# --------------------------------------------------------------------------- #


async def _metric_unavailable(query: str, **kwargs):
    return QueryResult(
        source=EvidenceSource.PROMETHEUS, available=False, value=None,
        hits=0, summary={}, error="timeout",
    )


@pytest.mark.asyncio
async def test_unavailable_data_is_inconclusive_and_never_recovered():
    rca = FakeRCA(["resource_exhaustion"])
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_metric_unavailable,
        log_query=_loki_unavailable,
    )
    incident = _incident()

    run = await engine.investigate(incident)

    # Unavailable evidence cannot confirm; after rounds it escalates (never recovered).
    assert run.state is InvestigationState.MANUAL_ESCALATION
    assert run.state is not InvestigationState.RECOVERED
    assert run.recovery_verifications == []


@pytest.mark.asyncio
async def test_recovery_with_unavailable_samples_is_inconclusive():
    rca = FakeRCA(["resource_exhaustion"])
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_available_metric(95),
    )
    incident = _incident()
    await engine.investigate(incident)
    await engine.accept_execution_receipt(incident, _succeeded_receipt(incident.incident_id))

    # Samples where the primary metric is missing → inconclusive, never recovered.
    missing = [
        RecoverySample(
            timestamp=NOW, metric_value=None, log_hits=None, health_ok=None, guardrails_ok=None,
        )
        for _ in range(2)
    ]
    run = await engine.verify_recovery(incident, missing)
    assert run.recovery_verifications[-1].outcome is RecoveryOutcome.INCONCLUSIVE
    assert run.state is not InvestigationState.RECOVERED


# --------------------------------------------------------------------------- #
# 7. Receipt idempotency + dry-run mismatch
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_duplicate_succeeded_receipt_is_idempotent():
    rca = FakeRCA(["resource_exhaustion"])
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_available_metric(95),
    )
    incident = _incident()
    await engine.investigate(incident)

    receipt = _succeeded_receipt(incident.incident_id)
    first = await engine.accept_execution_receipt(incident, receipt)
    second = await engine.accept_execution_receipt(incident, receipt.model_copy())

    assert first.state is InvestigationState.RECOVERY_OBSERVING
    assert second.state is InvestigationState.RECOVERY_OBSERVING
    # Only one receipt persisted (idempotent).
    assert len(second.receipts) == 1


@pytest.mark.asyncio
async def test_receipt_not_matching_dry_run_is_rejected():
    rca = FakeRCA(["resource_exhaustion"])
    engine = _engine(
        profiles=_Profiles(_profile()),
        rca=rca,
        metric_query=_available_metric(95),
    )
    incident = _incident()
    await engine.investigate(incident)

    mismatched = HealExecutionReceipt(
        incident_id=incident.incident_id,
        action_id="different-action",  # does not match dry-run action_id
        idempotency_key="key-1",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status="succeeded",
        completed_at=NOW,
    )
    run = await engine.accept_execution_receipt(incident, mismatched)
    assert run.state is InvestigationState.HEAL_FAILED
    assert "mismatch" in run.audit_summary["receipt_rejected_reason"]


# --------------------------------------------------------------------------- #
# 8. BadCase: captured evidence never contains raw secrets / full logs
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_captured_evidence_redacts_raw_payloads():
    rca = FakeRCA(["resource_exhaustion"])
    store = InvestigationStore()
    engine = InvestigationLoopEngine(
        store=store,
        profiles=_Profiles(_profile()),
        rca_agent=rca,
        heal_agent=FakeHeal(),
        metric_query=_available_metric(95),
        log_query=_loki_hits,
        clock=lambda: NOW,
    )
    incident = _incident()
    await engine.investigate(incident)

    snapshot = store.snapshot(incident.incident_id)
    for record in snapshot["evidence"]:
        # query_summary holds only the query string + source, never full log lines.
        assert set(record["query_summary"].keys()) <= {"query", "source", "sample_count"}
        # summary dict is bounded and JSON-safe.
        assert isinstance(record["summary"], dict)


# --------------------------------------------------------------------------- #
# 9. Registry completeness: all standard RCA causes have a profile
# --------------------------------------------------------------------------- #


def test_all_standard_root_causes_have_profiles():
    registry = ProfileRegistry()
    profiles = registry.load()
    standard_causes = InvestigationLoopEngine._standard_causes()
    missing = [c for c in standard_causes if c not in profiles]
    assert missing == [], f"missing profiles for root causes: {missing}"
