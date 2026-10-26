"""Typed, auditable domain records for the RCA investigation loop.

These models describe verification and receipt boundaries only.  They do not execute
remediation actions or perform queries against external systems.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class InvestigationState(str, Enum):
    """Lifecycle states owned by the investigation verification loop."""

    PENDING = "pending"
    INVESTIGATING = "investigating"
    REPLANNING = "replanning"
    AWAITING_EXECUTION = "awaiting_execution"
    RECOVERY_OBSERVING = "recovery_observing"
    RECOVERED = "recovered"
    HEAL_FAILED = "heal_failed"
    MANUAL_ESCALATION = "manual_escalation"


class AssertionOutcome(str, Enum):
    """Normalized result of a single evidence assertion."""

    PASS = "pass"
    FAIL = "fail"
    COUNTER = "counter"
    UNAVAILABLE = "unavailable"


class VerificationStatus(str, Enum):
    """Profile-level root-cause verification decision."""

    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    MANUAL_ONLY = "manual_only"


class RecoveryOutcome(str, Enum):
    """Result of post-execution stable-window observation."""

    RECOVERED = "recovered"
    DEGRADED = "degraded"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class EvidenceSource(str, Enum):
    """Allowed evidence sources used in profiles and persisted records."""

    PROMETHEUS = "prometheus"
    LOKI = "loki"
    HEALTH = "health"
    GUARDRAIL = "guardrail"


class Expectation(BaseModel):
    """The expected shape of a safe query result."""

    operator: Literal["greater_than", "less_than", "between", "min_hits", "contains"]
    threshold: float | str | None = None
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "Expectation":
        if self.operator in {"greater_than", "less_than", "min_hits"} and self.threshold is None:
            raise ValueError(f"{self.operator} requires a threshold")
        if self.operator == "between":
            if self.minimum is None or self.maximum is None:
                raise ValueError("between requires minimum and maximum")
            if self.minimum > self.maximum:
                raise ValueError("between minimum must not exceed maximum")
        if self.operator == "contains" and not isinstance(self.threshold, str):
            raise ValueError("contains requires a string threshold")
        return self


class AssertionRule(BaseModel):
    """One profile-defined, query-safe supporting or direct assertion."""

    id: str = Field(min_length=1, max_length=128)
    source: EvidenceSource
    query_template: str = Field(min_length=1, max_length=4096)
    evidence_level: Literal["direct", "supporting"]
    weight: float = Field(ge=0.0, le=1.0)
    expectation: Expectation
    window_seconds: int = Field(default=300, ge=1, le=3600)
    timeout_seconds: int = Field(default=30, ge=1, le=3600)
    max_rows: int = Field(default=100, ge=1, le=10_000)


class CounterEvidenceRule(BaseModel):
    """A rule whose matching result counts against the candidate root cause."""

    id: str = Field(min_length=1, max_length=128)
    source: EvidenceSource
    query_template: str = Field(min_length=1, max_length=4096)
    weight: float = Field(ge=0.0, le=1.0)
    expectation: Expectation
    window_seconds: int = Field(default=300, ge=1, le=3600)
    timeout_seconds: int = Field(default=30, ge=1, le=3600)
    max_rows: int = Field(default=100, ge=1, le=10_000)


class DecisionPolicy(BaseModel):
    """Scoring limits for an automatic profile decision."""

    minimum_score: float = Field(ge=0.0, le=1.0)
    require_direct_evidence: bool = True
    maximum_counter_evidence: float = Field(ge=0.0, le=1.0)


class RecoveryPolicy(BaseModel):
    """Stable-window observation settings following a trusted execution receipt."""

    settle_seconds: int = Field(ge=0, le=3600)
    observation_seconds: int = Field(ge=1, le=3600)
    consecutive_samples: int = Field(ge=1, le=3600)

    @model_validator(mode="after")
    def validate_observation_window(self) -> "RecoveryPolicy":
        if self.observation_seconds < self.consecutive_samples:
            raise ValueError("observation_seconds must permit consecutive samples")
        return self


StandardRootCause = Literal[
    # Current RCAAgent PRIOR_PROBABILITIES / LIKELIHOODS outputs.
    "recent_deployment",
    "configuration_change",
    "resource_exhaustion",
    "dependency_failure",
    "network_issue",
    "database_issue",
    "code_bug",
    "traffic_spike",
    "hardware_failure",
    "third_party_issue",
    # Profile candidates retained for deterministic verification coverage and fallback.
    "memory_leak",
    "unknown",
]


class RootCauseVerificationProfile(BaseModel):
    """Versioned evidence policy for a standard RCA root cause."""

    _UNSCOPED_TEMPLATE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?:\{\s*\}|=~\s*['\"]\.\*['\"]|\|~\s*['\"]\.\*['\"])",
    )
    _PLACEHOLDER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}",
    )
    _LEGACY_PLACEHOLDER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\{[A-Za-z_][A-Za-z0-9_]*\}",
    )
    _SELECTOR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\{([^{}]*)\}")
    _ALLOWED_TEMPLATE_PLACEHOLDERS: ClassVar[frozenset[str]] = frozenset(
        {"service", "target", "metric", "environment"},
    )
    _REQUIRED_SCOPE_TEMPLATES: ClassVar[tuple[str, str]] = ("{{ service }}", "{{ target }}")

    root_cause: StandardRootCause
    version: str = Field(min_length=1, max_length=64)
    manual_only: bool = False
    assertions: list[AssertionRule] = Field(default_factory=list, max_length=100)
    counter_evidence: list[CounterEvidenceRule] = Field(default_factory=list, max_length=100)
    decision: DecisionPolicy
    next_strategies: dict[Literal["inconclusive", "contradicted"], list[str]]
    recovery: RecoveryPolicy

    @model_validator(mode="after")
    def validate_profile(self) -> "RootCauseVerificationProfile":
        if not self.manual_only:
            if not any(assertion.evidence_level == "direct" for assertion in self.assertions):
                raise ValueError("non-manual profiles require at least one direct assertion")
            if not self.counter_evidence:
                raise ValueError("non-manual profiles require at least one counter evidence rule")

        rules = [*self.assertions, *self.counter_evidence]
        rule_ids = [rule.id for rule in rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("assertion and counter evidence ids must be globally unique")
        for rule in rules:
            self._validate_scoped_query_template(rule.query_template)

        required_strategies = {"inconclusive", "contradicted"}
        if set(self.next_strategies) != required_strategies:
            raise ValueError("next_strategies must define inconclusive and contradicted paths")
        if any(not strategies or any(not strategy.strip() for strategy in strategies)
               for strategies in self.next_strategies.values()):
            raise ValueError("each next strategy path must contain a non-empty strategy")
        return self

    @classmethod
    def _validate_scoped_query_template(cls, template: str) -> None:
        placeholders = cls._PLACEHOLDER_PATTERN.findall(template)
        if any(name not in cls._ALLOWED_TEMPLATE_PLACEHOLDERS for name in placeholders):
            raise ValueError("query_template contains an unknown placeholder")
        rendered_without_placeholders = cls._PLACEHOLDER_PATTERN.sub("", template)
        if (
            template.count("{") != template.count("}")
            or "{{" in rendered_without_placeholders
            or "}}" in rendered_without_placeholders
            or re.search(r"}\s*}", rendered_without_placeholders)
            or cls._LEGACY_PLACEHOLDER_PATTERN.search(rendered_without_placeholders)
        ):
            raise ValueError("query_template contains a malformed or legacy placeholder")
        masked_template = cls._PLACEHOLDER_PATTERN.sub("placeholder", template)
        selectors = cls._SELECTOR_PATTERN.findall(masked_template)
        if not selectors:
            raise ValueError("query_template requires at least one scoped selector")
        for selector in selectors:
            if not all(
                re.search(rf"\b{scope}\s*=", selector)
                for scope in ("service", "target")
            ):
                raise ValueError("query_template requires every selector to scope service and target")
        if cls._UNSCOPED_TEMPLATE_PATTERN.search(template):
            raise ValueError("query_template rejects empty selectors and match-all patterns")


class EvidenceRecord(BaseModel):
    """Persisted, redaction-ready result of one profile assertion query."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    incident_id: str = Field(min_length=1, max_length=128)
    assertion_id: str = Field(min_length=1, max_length=128)
    source: EvidenceSource
    outcome: AssertionOutcome
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    query_summary: dict[str, Any] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    score: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("observed_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        return value


class RootCauseHypothesis(BaseModel):
    """A candidate root cause considered during a numbered RCA round."""

    hypothesis_id: str = Field(default_factory=lambda: str(uuid4()))
    root_cause: StandardRootCause
    confidence: float = Field(ge=0.0, le=1.0)
    rca_round: int = Field(default=1, ge=1, le=3)
    profile_version: str | None = Field(default=None, max_length=64)
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class HealExecutionReceipt(BaseModel):
    """Trusted external acknowledgement; a dry-run is not such a receipt."""

    receipt_id: str = Field(default_factory=lambda: str(uuid4()))
    incident_id: str = Field(min_length=1, max_length=128)
    action_id: str = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=1, max_length=256)
    playbook_id: str = Field(min_length=1, max_length=128)
    target_resource: str = Field(min_length=1, max_length=512)
    status: Literal["succeeded", "failed", "rejected"]
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("received_at", "completed_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("receipt timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_completion(self) -> "HealExecutionReceipt":
        if self.status == "succeeded" and self.completed_at is None:
            raise ValueError("succeeded receipt requires completed_at")
        return self


class RecoveryVerification(BaseModel):
    """Redaction-ready result of recovery observation after a succeeded receipt."""

    verification_id: str = Field(default_factory=lambda: str(uuid4()))
    incident_id: str = Field(min_length=1, max_length=128)
    receipt_id: str = Field(min_length=1, max_length=128)
    receipt_status: Literal["succeeded"]
    outcome: RecoveryOutcome
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signals: list[dict[str, Any]] = Field(default_factory=list, max_length=100)
    unavailable_sources: list[EvidenceSource] = Field(default_factory=list, max_length=4)
    summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("verified_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("verified_at must be timezone-aware")
        return value


class InvestigationRun(BaseModel):
    """Incident-associated, bounded state for an investigation verification run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    incident_id: str = Field(min_length=1, max_length=128)
    state: InvestigationState = InvestigationState.PENDING
    verification_status: VerificationStatus = VerificationStatus.INCONCLUSIVE
    rca_round: int = Field(default=0, ge=0, le=3)
    heal_attempt: int = Field(default=0, ge=0, le=2)
    hypotheses: list[RootCauseHypothesis] = Field(default_factory=list, max_length=3)
    evidence_refs: list[str] = Field(default_factory=list, max_length=300)
    receipts: list[HealExecutionReceipt] = Field(default_factory=list, max_length=2)
    recovery_verifications: list[RecoveryVerification] = Field(default_factory=list, max_length=10)
    audit_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run timestamps must be timezone-aware")
        return value
