from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.investigation import (
    AssertionOutcome,
    AssertionRule,
    EvidenceRecord,
    Expectation,
    HealExecutionReceipt,
    InvestigationRun,
    InvestigationState,
    RecoveryOutcome,
    RecoveryVerification,
    RootCauseHypothesis,
    RootCauseVerificationProfile,
    VerificationStatus,
)


def valid_profile() -> dict:
    return {
        "root_cause": "dependency_failure",
        "version": "1.0",
        "manual_only": False,
        "assertions": [
            {
                "id": "upstream_errors",
                "source": "prometheus",
                "query_template": 'rate(errors{service="{{ service }}",target="{{ target }}"}[5m])',
                "evidence_level": "direct",
                "weight": 0.7,
                "expectation": {"operator": "greater_than", "threshold": 0.05},
            }
        ],
        "counter_evidence": [
            {
                "id": "healthy_target",
                "source": "health",
                "query_template": 'health{service="{{ service }}",target="{{ target }}"}',
                "weight": 0.7,
                "expectation": {"operator": "contains", "threshold": "healthy"},
            }
        ],
        "decision": {
            "minimum_score": 0.65,
            "require_direct_evidence": True,
            "maximum_counter_evidence": 0.3,
        },
        "next_strategies": {
            "inconclusive": ["expand_time_window"],
            "contradicted": ["test_recent_deployment"],
        },
        "recovery": {
            "settle_seconds": 30,
            "observation_seconds": 120,
            "consecutive_samples": 3,
        },
    }


def test_profile_accepts_direct_assertion():
    profile = RootCauseVerificationProfile.model_validate(valid_profile())

    assert profile.root_cause == "dependency_failure"
    assert profile.assertions[0].evidence_level == "direct"


def test_non_manual_profile_requires_direct_assertion():
    data = valid_profile()
    data["assertions"][0]["evidence_level"] = "supporting"

    with pytest.raises(ValidationError):
        RootCauseVerificationProfile.model_validate(data)


def test_non_manual_profile_requires_counter_evidence():
    data = valid_profile()
    data["counter_evidence"] = []

    with pytest.raises(ValidationError, match="counter evidence"):
        RootCauseVerificationProfile.model_validate(data)


@pytest.mark.parametrize(
    "unsafe_template",
    [
        "up{}",
        "up",
        'metric{service=~".*",target="{{ target }}"}',
        "metric{service='{{ service }}',target=~'.*'}",
        '{service="{{ service }}",target="{{ target }}"} |~ ".*"',
    ],
)
def test_non_manual_profile_rejects_unscoped_or_broad_query_templates(unsafe_template: str):
    data = valid_profile()
    data["assertions"][0]["query_template"] = unsafe_template

    with pytest.raises(ValidationError, match="query_template"):
        RootCauseVerificationProfile.model_validate(data)


def test_non_manual_profile_requires_service_and_target_templates_on_counter_evidence():
    data = valid_profile()
    data["counter_evidence"][0]["query_template"] = 'health{service="{{ service }}"}'

    with pytest.raises(ValidationError, match="service and target"):
        RootCauseVerificationProfile.model_validate(data)


def test_profile_rejects_an_additional_selector_without_service_and_target_scope():
    data = valid_profile()
    data["assertions"][0]["query_template"] = (
        'sum(up{job="api"}) + max(up{service="{{ service }}",target="{{ target }}"})'
    )

    with pytest.raises(ValidationError, match="every selector"):
        RootCauseVerificationProfile.model_validate(data)


def test_profile_rejects_duplicate_ids_across_assertion_and_counter_rules():
    data = valid_profile()
    data["counter_evidence"][0]["id"] = data["assertions"][0]["id"]

    with pytest.raises(ValidationError, match="unique"):
        RootCauseVerificationProfile.model_validate(data)


@pytest.mark.parametrize(
    "invalid_placeholder",
    [
        "{service}",
        "{{ unknown }}",
        "{{ service }",
        "{{ service }} }",
    ],
)
def test_non_manual_profile_rejects_unknown_legacy_and_malformed_placeholders(invalid_placeholder: str):
    data = valid_profile()
    data["assertions"][0]["query_template"] = (
        f'errors{{service="{{{{ service }}}}",target="{{{{ target }}}}",bad="{invalid_placeholder}"}}'
    )

    with pytest.raises(ValidationError, match="placeholder"):
        RootCauseVerificationProfile.model_validate(data)


def test_manual_only_profile_may_skip_direct_assertion():
    data = valid_profile()
    data["manual_only"] = True
    data["assertions"] = []

    profile = RootCauseVerificationProfile.model_validate(data)

    assert profile.manual_only is True


def test_unknown_root_cause_is_rejected():
    data = valid_profile()
    data["root_cause"] = "not_a_supported_cause"

    with pytest.raises(ValidationError):
        RootCauseVerificationProfile.model_validate(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("assertions", 0, "weight"), 1.1),
        (("recovery", "settle_seconds"), 3601),
    ],
)
def test_profile_rejects_invalid_weight_or_window(path: tuple[str | int, ...], value: float | int):
    data = valid_profile()
    target = data
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError):
        RootCauseVerificationProfile.model_validate(data)


def test_profile_requires_next_strategies():
    data = valid_profile()
    data["next_strategies"] = {"inconclusive": []}

    with pytest.raises(ValidationError):
        RootCauseVerificationProfile.model_validate(data)


def test_domain_records_use_enum_states_and_timezone_aware_timestamps():
    now = datetime.now(timezone.utc)
    rule = AssertionRule(
        id="upstream_errors",
        source="prometheus",
        query_template='errors{service="{{ service }}",target="{{ target }}"}',
        evidence_level="direct",
        weight=0.7,
        expectation=Expectation(operator="greater_than", threshold=0.05),
    )
    evidence = EvidenceRecord(
        incident_id="inc-1",
        assertion_id=rule.id,
        source="prometheus",
        outcome=AssertionOutcome.PASS,
        observed_at=now,
        summary={"value": 0.2},
    )
    hypothesis = RootCauseHypothesis(root_cause="dependency_failure", confidence=0.8)
    receipt = HealExecutionReceipt(
        incident_id="inc-1",
        action_id="action-1",
        idempotency_key="key-1",
        playbook_id="playbook-1",
        target_resource="orders",
        status="succeeded",
        completed_at=now,
    )
    recovery = RecoveryVerification(
        incident_id="inc-1",
        receipt_id=receipt.receipt_id,
        receipt_status="succeeded",
        outcome=RecoveryOutcome.INCONCLUSIVE,
        verified_at=now,
        unavailable_sources=["loki"],
    )
    run = InvestigationRun(
        incident_id="inc-1",
        state=InvestigationState.AWAITING_EXECUTION,
        verification_status=VerificationStatus.CONFIRMED,
        hypotheses=[hypothesis],
        evidence_refs=[evidence.evidence_id],
        receipts=[receipt],
        recovery_verifications=[recovery],
    )

    assert evidence.observed_at.tzinfo is not None
    assert run.state is InvestigationState.AWAITING_EXECUTION
    assert run.verification_status is VerificationStatus.CONFIRMED
    assert recovery.outcome is RecoveryOutcome.INCONCLUSIVE
    assert recovery.unavailable_sources == ["loki"]


@pytest.mark.parametrize("root_cause", ["third_party_issue", "memory_leak", "traffic_spike", "hardware_failure", "unknown"])
def test_standard_root_causes_cover_rca_outputs_and_profile_candidates(root_cause: str):
    profile = valid_profile()
    profile["root_cause"] = root_cause
    profile["manual_only"] = root_cause == "unknown"
    if profile["manual_only"]:
        profile["assertions"] = []

    assert RootCauseVerificationProfile.model_validate(profile).root_cause == root_cause
    assert RootCauseHypothesis(root_cause=root_cause, confidence=0.5).root_cause == root_cause


def test_recovery_verification_requires_succeeded_receipt_boundary():
    with pytest.raises(ValidationError):
        RecoveryVerification(
            incident_id="inc-1",
            receipt_id="receipt-1",
            receipt_status="failed",
            outcome="inconclusive",
        )


def test_recovery_verification_tracks_unavailable_signal_sources():
    recovery = RecoveryVerification(
        incident_id="inc-1",
        receipt_id="receipt-1",
        receipt_status="succeeded",
        outcome="inconclusive",
        unavailable_sources=["prometheus", "loki"],
    )

    assert recovery.unavailable_sources == ["prometheus", "loki"]


@pytest.mark.parametrize(
    ("profile_mutation", "rule_payload"),
    [
        (("recovery", "observation_seconds"), None),
        (None, {"window_seconds": 0}),
    ],
)
def test_profile_rejects_zero_observation_or_assertion_window(
    profile_mutation: tuple[str, str] | None, rule_payload: dict | None
):
    profile = valid_profile()
    if profile_mutation:
        profile[profile_mutation[0]][profile_mutation[1]] = 0
    if rule_payload:
        profile["assertions"][0].update(rule_payload)

    with pytest.raises(ValidationError):
        RootCauseVerificationProfile.model_validate(profile)


def test_hypothesis_rejects_non_standard_root_cause():
    with pytest.raises(ValidationError):
        RootCauseHypothesis(root_cause="agent_freeform_cause", confidence=0.5)
