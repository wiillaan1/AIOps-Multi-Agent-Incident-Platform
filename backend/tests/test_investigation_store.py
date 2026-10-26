"""Tests for the in-memory, incident-scoped investigation store."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.models.investigation import (
    AssertionOutcome,
    EvidenceRecord,
    EvidenceSource,
    HealExecutionReceipt,
    RootCauseHypothesis,
)
from app.services.investigation_store import InvestigationStore


def _evidence(incident_id: str = "inc-1") -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="ev-1",
        incident_id=incident_id,
        assertion_id="cpu-high",
        source=EvidenceSource.PROMETHEUS,
        outcome=AssertionOutcome.PASS,
        query_summary={"value": 95},
    )


def _receipt(
    incident_id: str = "inc-1",
    key: str = "key-1",
    status: str = "succeeded",
) -> HealExecutionReceipt:
    now = datetime.now(timezone.utc)
    return HealExecutionReceipt(
        receipt_id="receipt-1",
        incident_id=incident_id,
        action_id="action-1",
        idempotency_key=key,
        playbook_id="restart-orders",
        target_resource="orders-api",
        status=status,
        received_at=now,
        completed_at=now + timedelta(seconds=1),
    )


def test_get_or_create_returns_one_incident_scoped_run():
    store = InvestigationStore()

    first = store.get_or_create("inc-1")
    second = store.get_or_create("inc-1")

    assert second.run_id == first.run_id
    assert store.get_run("other") is None


def test_store_records_only_records_for_their_incident_and_exposes_json_snapshot():
    store = InvestigationStore()
    evidence = _evidence()
    hypothesis = RootCauseHypothesis(
        hypothesis_id="hyp-1",
        root_cause="resource_exhaustion",
        confidence=0.8,
        evidence_refs=[evidence.evidence_id],
    )

    assert store.record_evidence("inc-1", evidence).evidence_id == evidence.evidence_id
    store.record_hypothesis("inc-1", hypothesis)
    snapshot = store.snapshot("inc-1")

    assert snapshot["incident_id"] == "inc-1"
    assert snapshot["evidence"][0]["evidence_id"] == "ev-1"
    assert snapshot["hypotheses"][0]["hypothesis_id"] == "hyp-1"
    with pytest.raises(ValueError, match="incident_id"):
        store.record_evidence("other", evidence)


def test_receipt_idempotency_returns_original_for_same_incident():
    store = InvestigationStore()
    receipt = _receipt()

    accepted, first = store.record_receipt(receipt)
    accepted_again, second = store.record_receipt(
        receipt.model_copy(update={"receipt_id": "receipt-retry"})
    )

    assert accepted is True
    assert accepted_again is False
    assert second.receipt_id == first.receipt_id


@pytest.mark.parametrize(
    "field",
    ["action_id", "playbook_id", "target_resource", "status"],
)
def test_receipt_idempotency_key_rejects_different_payload_for_same_incident(field: str):
    store = InvestigationStore()
    receipt = _receipt()
    store.record_receipt(receipt)
    replacement = {
        "action_id": "action-2",
        "playbook_id": "scale-orders",
        "target_resource": "payments-api",
        "status": "failed",
    }[field]

    with pytest.raises(ValueError, match="idempotency_key payload conflict"):
        store.record_receipt(receipt.model_copy(update={field: replacement}))


def test_receipt_key_cannot_be_reused_by_another_incident():
    store = InvestigationStore()
    store.record_receipt(_receipt(incident_id="inc-1", key="shared-key"))

    with pytest.raises(ValueError, match="idempotency_key"):
        store.record_receipt(_receipt(incident_id="inc-2", key="shared-key"))


@pytest.mark.parametrize("status", ["failed", "rejected"])
def test_failed_or_rejected_receipts_are_stored_without_recovery_verifications(status: str):
    store = InvestigationStore()

    accepted, _ = store.record_receipt(_receipt(status=status))
    snapshot = store.snapshot("inc-1")

    assert accepted is True
    assert snapshot["receipts"][0]["status"] == status
    assert snapshot["recovery_verifications"] == []


def test_succeeded_receipt_requires_completion_time():
    with pytest.raises(ValidationError, match="completed_at"):
        HealExecutionReceipt(
            incident_id="inc-1",
            action_id="action-1",
            idempotency_key="key-1",
            playbook_id="restart-orders",
            target_resource="orders-api",
            status="succeeded",
        )
