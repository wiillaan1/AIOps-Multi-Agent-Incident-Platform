"""Tests for deterministic post-receipt recovery verification."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models.investigation import (
    EvidenceSource,
    HealExecutionReceipt,
    RecoveryOutcome,
    RootCauseVerificationProfile,
)
from app.services.recovery_verifier import RecoverySample, RecoveryVerifier


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _profile(consecutive_samples: int = 3) -> RootCauseVerificationProfile:
    return RootCauseVerificationProfile.model_validate(
        {
            "root_cause": "traffic_spike",
            "version": "1.0",
            "assertions": [
                {
                    "id": "direct",
                    "source": "prometheus",
                    "query_template": 'rate{service="{{ service }}",target="{{ target }}"}',
                    "evidence_level": "direct",
                    "weight": 1.0,
                    "expectation": {"operator": "greater_than", "threshold": 10},
                }
            ],
            "counter_evidence": [
                {
                    "id": "counter",
                    "source": "prometheus",
                    "query_template": 'rate{service="{{ service }}",target="{{ target }}"}',
                    "weight": 1.0,
                    "expectation": {"operator": "less_than", "threshold": 1},
                }
            ],
            "decision": {
                "minimum_score": 1.0,
                "require_direct_evidence": True,
                "maximum_counter_evidence": 0.3,
            },
            "next_strategies": {
                "inconclusive": ["expand_window"],
                "contradicted": ["inspect_dependency"],
            },
            "recovery": {
                "settle_seconds": 30,
                "observation_seconds": 120,
                "consecutive_samples": consecutive_samples,
            },
        }
    )


def _receipt(
    *, status: str = "succeeded", completed_at: datetime | None = NOW
) -> HealExecutionReceipt:
    return HealExecutionReceipt(
        receipt_id="receipt-1",
        incident_id="inc-1",
        action_id="action-1",
        idempotency_key="key-1",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status=status,
        received_at=NOW - timedelta(seconds=1),
        completed_at=completed_at,
    )


def _sample(
    offset_seconds: int,
    *,
    metric_value: float | None = 40,
    log_hits: int | None = 0,
    health_ok: bool | None = True,
    guardrails_ok: bool | None = True,
) -> RecoverySample:
    return RecoverySample(
        timestamp=NOW + timedelta(seconds=offset_seconds),
        metric_value=metric_value,
        log_hits=log_hits,
        health_ok=health_ok,
        guardrails_ok=guardrails_ok,
    )


def _stable_samples() -> list[RecoverySample]:
    return [_sample(31), _sample(60), _sample(120)]


def test_log_hits_below_a_present_baseline_recover():
    result = RecoveryVerifier().verify(
        _profile(),
        _receipt(),
        {"threshold": 80, "log_hits": 5},
        [_sample(31, log_hits=4), _sample(60, log_hits=1), _sample(120, log_hits=0)],
    )

    assert result.outcome is RecoveryOutcome.RECOVERED


def test_missing_log_baseline_is_inconclusive_even_when_post_hits_are_zero():
    result = RecoveryVerifier().verify(
        _profile(), _receipt(), {"threshold": 80}, _stable_samples()
    )

    assert result.outcome is RecoveryOutcome.INCONCLUSIVE
    assert result.receipt_status == "succeeded"
    assert result.unavailable_sources == [EvidenceSource.LOKI]
    assert {signal["name"] for signal in result.signals} == {
        "primary_metric",
        "loki_errors",
        "health",
        "guardrail",
    }


def test_samples_before_receipt_completion_are_not_observations():
    result = RecoveryVerifier().verify(
        _profile(),
        _receipt(),
        {"threshold": 80, "log_hits": 5},
        [_sample(-60), _sample(-30), _sample(-1)],
    )

    assert result.outcome is RecoveryOutcome.INCONCLUSIVE
    assert set(result.unavailable_sources) == set(EvidenceSource)
    assert result.summary["observation_start"] == (NOW + timedelta(seconds=30)).isoformat()


def test_insufficient_required_samples_are_inconclusive_with_unavailable_sources():
    result = RecoveryVerifier().verify(
        _profile(), _receipt(), {"threshold": 80, "log_hits": 5}, _stable_samples()[:2]
    )

    assert result.outcome is RecoveryOutcome.INCONCLUSIVE
    assert set(result.unavailable_sources) == set(EvidenceSource)


def test_missing_required_source_data_is_inconclusive_and_names_source():
    result = RecoveryVerifier().verify(
        _profile(),
        _receipt(),
        {"threshold": 80, "log_hits": 5},
        [_sample(31, log_hits=None), _sample(60, log_hits=None), _sample(120, log_hits=None)],
    )

    assert result.outcome is RecoveryOutcome.INCONCLUSIVE
    assert result.unavailable_sources == [EvidenceSource.LOKI]


@pytest.mark.parametrize(
    "samples",
    [
        [_sample(31, metric_value=80), _sample(60), _sample(120)],
        [_sample(31), _sample(60, guardrails_ok=False), _sample(120)],
    ],
)
def test_explicit_primary_or_guardrail_failure_fails(samples: list[RecoverySample]):
    assert (
        RecoveryVerifier().verify(
            _profile(), _receipt(), {"threshold": 80, "log_hits": 5}, samples
        ).outcome
        is RecoveryOutcome.FAILED
    )


def test_loki_errors_not_reduced_is_degraded():
    result = RecoveryVerifier().verify(
        _profile(), _receipt(), {"threshold": 80, "log_hits": 5},
        [_sample(31, log_hits=5), _sample(60, log_hits=5), _sample(120, log_hits=5)],
    )

    assert result.outcome is RecoveryOutcome.DEGRADED
    assert result.unavailable_sources == []


def test_failed_health_is_degraded_when_no_primary_or_guardrail_failure():
    result = RecoveryVerifier().verify(
        _profile(), _receipt(), {"threshold": 80, "log_hits": 5},
        [_sample(31, health_ok=False), _sample(60), _sample(120)],
    )

    assert result.outcome is RecoveryOutcome.DEGRADED


def test_verify_rejects_non_succeeded_receipt():
    with pytest.raises(ValueError, match="succeeded"):
        RecoveryVerifier().verify(
            _profile(), _receipt(status="failed", completed_at=NOW),
            {"threshold": 80, "log_hits": 5}, _stable_samples()
        )


def test_should_observe_only_for_nonterminal_recovery_outcomes():
    verifier = RecoveryVerifier()

    assert verifier.should_observe("degraded") is True
    assert verifier.should_observe(RecoveryOutcome.INCONCLUSIVE) is True
    assert verifier.should_observe("recovered") is False
    assert verifier.should_observe("failed") is False
