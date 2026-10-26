"""Deterministic, side-effect-free verification of recovery after execution.

The caller supplies normalized observations collected after a trusted execution
receipt.  This module deliberately performs no metrics, log, or health queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from app.models.investigation import (
    EvidenceSource,
    HealExecutionReceipt,
    RecoveryOutcome,
    RecoveryVerification,
    RootCauseVerificationProfile,
)


@dataclass(frozen=True)
class RecoverySignal:
    """One normalized observation signal included in a recovery decision."""

    name: str
    status: Literal["pass", "fail", "unavailable"]
    summary: dict[str, Any]


@dataclass(frozen=True)
class RecoverySample:
    """A bounded multi-source observation at one timezone-aware instant."""

    timestamp: datetime
    metric_value: float | None
    log_hits: int | None
    health_ok: bool | None
    guardrails_ok: bool | None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("recovery sample timestamp must be timezone-aware")


class RecoveryVerifier:
    """Apply a profile's stable-window policy to normalized recovery samples."""

    _ALL_SOURCES = (
        EvidenceSource.PROMETHEUS,
        EvidenceSource.LOKI,
        EvidenceSource.HEALTH,
        EvidenceSource.GUARDRAIL,
    )

    def should_observe(self, outcome: str | RecoveryOutcome) -> bool:
        """Return whether an outcome permits one further observation attempt."""
        return RecoveryOutcome(outcome) in {
            RecoveryOutcome.DEGRADED,
            RecoveryOutcome.INCONCLUSIVE,
        }

    def verify(
        self,
        profile: RootCauseVerificationProfile,
        receipt: HealExecutionReceipt,
        baseline: dict[str, Any],
        samples: list[RecoverySample],
    ) -> RecoveryVerification:
        """Assess post-completion samples without ever inferring recovery from gaps."""
        self._validate_receipt(receipt)
        threshold = self._number(baseline.get("threshold"))
        log_baseline = self._number(baseline.get("log_hits"))
        observation_start = receipt.completed_at + timedelta(
            seconds=profile.recovery.settle_seconds
        )
        observation_end = observation_start + timedelta(
            seconds=profile.recovery.observation_seconds
        )
        observations = sorted(
            (
                sample
                for sample in samples
                if observation_start <= sample.timestamp <= observation_end
            ),
            key=lambda sample: sample.timestamp,
        )

        unavailable = self._unavailable_sources(
            threshold, log_baseline, observations, profile.recovery.consecutive_samples
        )
        selected = observations[-profile.recovery.consecutive_samples :]
        signals = self._signals(threshold, log_baseline, selected, unavailable)
        outcome = self._outcome(signals, unavailable)

        return RecoveryVerification(
            incident_id=receipt.incident_id,
            receipt_id=receipt.receipt_id,
            receipt_status="succeeded",
            outcome=outcome,
            verified_at=datetime.now(timezone.utc),
            signals=[
                {"name": signal.name, "status": signal.status, "summary": signal.summary}
                for signal in signals
            ],
            unavailable_sources=unavailable,
            summary={
                "receipt_completed_at": receipt.completed_at.isoformat(),
                "observation_start": observation_start.isoformat(),
                "observation_end": observation_end.isoformat(),
                "required_consecutive_samples": profile.recovery.consecutive_samples,
                "observed_samples": len(observations),
            },
        )

    @staticmethod
    def _validate_receipt(receipt: HealExecutionReceipt) -> None:
        if receipt.status != "succeeded" or receipt.completed_at is None:
            raise ValueError("recovery verification requires a succeeded completed receipt")

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _unavailable_sources(
        self,
        threshold: float | None,
        log_baseline: float | None,
        observations: list[RecoverySample],
        required_samples: int,
    ) -> list[EvidenceSource]:
        if len(observations) < required_samples:
            return list(self._ALL_SOURCES)

        selected = observations[-required_samples:]
        unavailable: list[EvidenceSource] = []
        if threshold is None or any(sample.metric_value is None for sample in selected):
            unavailable.append(EvidenceSource.PROMETHEUS)
        if log_baseline is None or any(sample.log_hits is None for sample in selected):
            unavailable.append(EvidenceSource.LOKI)
        if any(sample.health_ok is None for sample in selected):
            unavailable.append(EvidenceSource.HEALTH)
        if any(sample.guardrails_ok is None for sample in selected):
            unavailable.append(EvidenceSource.GUARDRAIL)
        return unavailable

    def _signals(
        self,
        threshold: float | None,
        log_baseline: float | None,
        samples: list[RecoverySample],
        unavailable: list[EvidenceSource],
    ) -> list[RecoverySignal]:
        unavailable_set = set(unavailable)
        metric_status: Literal["pass", "fail", "unavailable"]
        if EvidenceSource.PROMETHEUS in unavailable_set:
            metric_status = "unavailable"
        elif all(sample.metric_value is not None and sample.metric_value < threshold for sample in samples):
            metric_status = "pass"
        else:
            metric_status = "fail"

        if EvidenceSource.LOKI in unavailable_set:
            log_status: Literal["pass", "fail", "unavailable"] = "unavailable"
        elif all(sample.log_hits is not None and sample.log_hits < log_baseline for sample in samples):
            log_status = "pass"
        else:
            log_status = "fail"

        if EvidenceSource.HEALTH in unavailable_set:
            health_status: Literal["pass", "fail", "unavailable"] = "unavailable"
        elif all(sample.health_ok for sample in samples):
            health_status = "pass"
        else:
            health_status = "fail"

        if EvidenceSource.GUARDRAIL in unavailable_set:
            guardrail_status: Literal["pass", "fail", "unavailable"] = "unavailable"
        elif all(sample.guardrails_ok for sample in samples):
            guardrail_status = "pass"
        else:
            guardrail_status = "fail"

        return [
            RecoverySignal("primary_metric", metric_status, {"threshold": threshold}),
            RecoverySignal("loki_errors", log_status, {"baseline_hits": log_baseline}),
            RecoverySignal("health", health_status, {}),
            RecoverySignal("guardrail", guardrail_status, {}),
        ]

    @staticmethod
    def _outcome(
        signals: list[RecoverySignal], unavailable: list[EvidenceSource]
    ) -> RecoveryOutcome:
        by_name = {signal.name: signal.status for signal in signals}
        if by_name["primary_metric"] == "fail" or by_name["guardrail"] == "fail":
            return RecoveryOutcome.FAILED
        if unavailable:
            return RecoveryOutcome.INCONCLUSIVE
        if all(signal.status == "pass" for signal in signals):
            return RecoveryOutcome.RECOVERED
        return RecoveryOutcome.DEGRADED
