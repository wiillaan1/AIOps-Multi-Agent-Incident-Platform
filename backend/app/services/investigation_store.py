"""Incident-scoped, in-memory storage for bounded investigation records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.models.investigation import (
    EvidenceRecord,
    HealExecutionReceipt,
    InvestigationRun,
    RootCauseHypothesis,
)


class InvestigationStore:
    """Keep investigation records isolated by incident for this process lifetime."""

    def __init__(self) -> None:
        self._runs: dict[str, InvestigationRun] = {}
        self._evidence: dict[str, list[EvidenceRecord]] = {}
        self._receipt_keys: dict[str, tuple[str, HealExecutionReceipt]] = {}

    def get_or_create(self, incident_id: str) -> InvestigationRun:
        """Return the incident's run, creating its bounded record on first use."""
        if incident_id not in self._runs:
            self._runs[incident_id] = InvestigationRun(incident_id=incident_id)
        return self._runs[incident_id]

    def record_evidence(self, incident_id: str, record: EvidenceRecord) -> EvidenceRecord:
        """Append an incident's evidence and retain its reference in the run."""
        self._validate_incident(incident_id, record.incident_id, "evidence")
        run = self.get_or_create(incident_id)
        evidence = self._evidence.setdefault(incident_id, [])
        if len(evidence) >= 300 or len(run.evidence_refs) >= 300:
            raise ValueError("evidence limit reached for incident")
        evidence.append(record)
        run.evidence_refs.append(record.evidence_id)
        self._touch(run)
        return record

    def record_hypothesis(self, incident_id: str, hypothesis: RootCauseHypothesis) -> None:
        """Append a bounded RCA hypothesis to its incident run."""
        run = self.get_or_create(incident_id)
        if len(run.hypotheses) >= 3:
            raise ValueError("hypothesis limit reached for incident")
        run.hypotheses.append(hypothesis)
        self._touch(run)

    def record_receipt(
        self, receipt: HealExecutionReceipt
    ) -> tuple[bool, HealExecutionReceipt]:
        """Persist a receipt once; key reuse across incidents is rejected."""
        existing = self._receipt_keys.get(receipt.idempotency_key)
        if existing is not None:
            existing_incident_id, original = existing
            if existing_incident_id != receipt.incident_id:
                raise ValueError("idempotency_key cannot be reused across incidents")
            if self._receipt_payload(original) != self._receipt_payload(receipt):
                raise ValueError("idempotency_key payload conflict")
            return False, original

        run = self.get_or_create(receipt.incident_id)
        if len(run.receipts) >= 2:
            raise ValueError("receipt limit reached for incident")
        run.receipts.append(receipt)
        self._receipt_keys[receipt.idempotency_key] = (receipt.incident_id, receipt)
        self._touch(run)
        return True, receipt

    def get_run(self, incident_id: str) -> InvestigationRun | None:
        """Return a deep copy so callers cannot mutate stored state."""
        run = self._runs.get(incident_id)
        return run.model_copy(deep=True) if run is not None else None

    def find_receipt_by_key(self, idempotency_key: str) -> HealExecutionReceipt | None:
        """Return the stored receipt for a key, if any (deep copy)."""
        existing = self._receipt_keys.get(idempotency_key)
        if existing is None:
            return None
        _incident_id, original = existing
        return original.model_copy(deep=True)

    def snapshot(self, incident_id: str) -> dict[str, Any]:
        """Return a JSON-safe, incident-scoped snapshot for API consumers."""
        run = self.get_or_create(incident_id)
        return {
            **run.model_dump(mode="json"),
            "evidence": [
                record.model_dump(mode="json")
                for record in self._evidence.get(incident_id, [])
            ],
        }

    @staticmethod
    def _receipt_payload(receipt: HealExecutionReceipt) -> tuple[str, str, str, str, str]:
        """Fields that must stay immutable under a receipt idempotency key."""
        return (
            receipt.incident_id,
            receipt.action_id,
            receipt.playbook_id,
            receipt.target_resource,
            receipt.status,
        )

    @staticmethod
    def _validate_incident(expected: str, actual: str, record_type: str) -> None:
        if expected != actual:
            raise ValueError(f"{record_type} incident_id must match the target incident")

    @staticmethod
    def _touch(run: InvestigationRun) -> None:
        run.updated_at = datetime.now(timezone.utc)
