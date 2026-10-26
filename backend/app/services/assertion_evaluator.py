"""Deterministic, side-effect-free evaluation of root-cause profile assertions.

This module deliberately receives normalized query results rather than clients.  It
never issues Prometheus or Loki requests; callers must apply profile query budgets
before producing a :class:`QueryResult`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import TypeAlias

from app.models.investigation import (
    AssertionOutcome,
    AssertionRule,
    CounterEvidenceRule,
    EvidenceSource,
    RootCauseVerificationProfile,
    VerificationStatus,
)


Rule: TypeAlias = AssertionRule | CounterEvidenceRule
_PLACEHOLDER = re.compile(r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}")
_LEGACY_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_SCOPED_LABEL = r'\b{label}\s*=\s*"(?:[^"\\]|\\.)*"'
_SELECTOR = re.compile(r'\{((?:[^"{}]|"(?:[^"\\]|\\.)*")*)\}')
_TOP_LEVEL_BINARY_OPERATOR = re.compile(r"\b(?:or|unless)\b")
_QUOTED_VALUE = re.compile(r'"(?:[^"\\]|\\.)*"')
_METRIC_NAME = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*$")


@dataclass(frozen=True)
class QueryContext:
    """Whitelisted values that may be interpolated into a profile query."""

    service: str
    target: str
    metric: str
    environment: str
    labels: dict[str, str]


@dataclass(frozen=True)
class QueryResult:
    """Normalized, bounded query output supplied by a query adapter."""

    source: EvidenceSource
    available: bool
    value: float | None
    hits: int
    summary: dict
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _coerce_evidence_source(self.source))


@dataclass(frozen=True)
class AssertionEvaluation:
    """Outcome of a single rule evaluated against an already obtained result."""

    assertion_id: str
    outcome: AssertionOutcome
    evidence_level: str | None
    weight: float
    source: EvidenceSource
    available: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", _coerce_evidence_source(self.source))


@dataclass(frozen=True)
class ProfileDecision:
    """Deterministic aggregate decision for one root-cause profile."""

    status: VerificationStatus
    supporting_score: float
    counter_score: float
    direct_evidence_present: bool
    unavailable_assertion_ids: tuple[str, ...]


def _coerce_evidence_source(source: EvidenceSource | str) -> EvidenceSource:
    try:
        return EvidenceSource(source)
    except ValueError as exc:
        raise ValueError(f"unsupported evidence source: {source}") from exc


def _escape_label_value(value: str) -> str:
    """Encode a value for use inside a PromQL/LogQL double-quoted label."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def render_query(template: str, context: QueryContext) -> str:
    """Safely render a profile query using only an explicit value whitelist.

    A rendered metric/log query must retain both ``service`` and ``target`` label
    selectors.  Unknown or malformed placeholders are rejected instead of being
    passed through to an external query service.
    """
    placeholder_names = _PLACEHOLDER.findall(template)
    if "metric" in placeholder_names and not _METRIC_NAME.fullmatch(context.metric):
        raise ValueError("metric placeholder must contain a metric identifier")
    values = {
        **context.labels,
        "service": context.service,
        "target": context.target,
        "metric": context.metric,
        "environment": context.environment,
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"unknown query placeholder: {name}")
        return _escape_label_value(values[name])

    rendered = _PLACEHOLDER.sub(replace, template)
    if "{{" in rendered or "}}" in rendered or _LEGACY_PLACEHOLDER.search(rendered):
        raise ValueError("query contains an unrecognized placeholder")
    query_without_quoted_values = _QUOTED_VALUE.sub('""', rendered)
    if _TOP_LEVEL_BINARY_OPERATOR.search(query_without_quoted_values):
        raise ValueError("query rejects top-level or/unless expressions")

    selectors = _SELECTOR.findall(rendered)
    if not selectors:
        raise ValueError("query requires at least one scoped selector")
    for selector in selectors:
        for required_label in ("service", "target"):
            if not re.search(_SCOPED_LABEL.format(label=re.escape(required_label)), selector):
                raise ValueError(
                    f"every query selector requires a {required_label} label selector"
                )
    return rendered


def _matches_expectation(rule: Rule, result: QueryResult) -> bool:
    expectation = rule.expectation
    operator = expectation.operator

    if operator == "greater_than":
        return result.value is not None and result.value > float(expectation.threshold)
    if operator == "less_than":
        return result.value is not None and result.value < float(expectation.threshold)
    if operator == "between":
        return (
            result.value is not None
            and expectation.minimum is not None
            and expectation.maximum is not None
            and expectation.minimum <= result.value <= expectation.maximum
        )
    if operator == "min_hits":
        return result.hits >= float(expectation.threshold)
    if operator == "contains":
        # Summary is supplied by a bounded/redacted adapter, never raw unbounded logs.
        return str(expectation.threshold) in json.dumps(result.summary, sort_keys=True)
    raise ValueError(f"unsupported expectation operator: {operator}")


def evaluate_assertion(rule: Rule, result: QueryResult) -> AssertionEvaluation:
    """Evaluate a normalized result without querying any external service."""
    if not result.available:
        outcome = AssertionOutcome.UNAVAILABLE
    elif _matches_expectation(rule, result):
        outcome = (
            AssertionOutcome.COUNTER
            if isinstance(rule, CounterEvidenceRule)
            else AssertionOutcome.PASS
        )
    else:
        outcome = AssertionOutcome.FAIL

    return AssertionEvaluation(
        assertion_id=rule.id,
        outcome=outcome,
        evidence_level=getattr(rule, "evidence_level", None),
        weight=rule.weight,
        source=result.source,
        available=result.available,
    )


def evaluate_profile(
    profile: RootCauseVerificationProfile,
    results: list[AssertionEvaluation],
) -> ProfileDecision:
    """Apply a profile's direct/supporting/counter decision policy.

    Missing evaluations and unavailable sources are insufficient evidence, not
    assertion failures.  A counter rule that actually matched is strong evidence
    against the candidate and therefore takes precedence over confirmation gates.
    """
    if profile.manual_only:
        return ProfileDecision(
            status=VerificationStatus.MANUAL_ONLY,
            supporting_score=0.0,
            counter_score=0.0,
            direct_evidence_present=False,
            unavailable_assertion_ids=(),
        )

    assertion_rules = {rule.id: rule for rule in profile.assertions}
    counter_rules = {rule.id: rule for rule in profile.counter_evidence}
    if set(assertion_rules).intersection(counter_rules):
        raise ValueError("profile assertion and counter evidence ids must be unique")
    evaluations = {result.assertion_id: result for result in results}
    required_rule_ids = [*assertion_rules, *counter_rules]
    unavailable_ids = tuple(
        rule_id
        for rule_id in required_rule_ids
        if evaluations.get(rule_id) is None
        or not evaluations[rule_id].available
        or evaluations[rule_id].outcome is AssertionOutcome.UNAVAILABLE
    )

    supporting_score = sum(
        rule.weight
        for rule_id, rule in assertion_rules.items()
        if evaluations.get(rule_id) is not None
        and evaluations[rule_id].outcome is AssertionOutcome.PASS
    )
    counter_score = sum(
        rule.weight
        for rule_id, rule in counter_rules.items()
        if evaluations.get(rule_id) is not None
        and evaluations[rule_id].outcome is AssertionOutcome.COUNTER
    )
    direct_evidence_present = any(
        rule.evidence_level == "direct"
        and evaluations.get(rule_id) is not None
        and evaluations[rule_id].outcome is AssertionOutcome.PASS
        for rule_id, rule in assertion_rules.items()
    )
    if counter_score > profile.decision.maximum_counter_evidence:
        status = VerificationStatus.CONTRADICTED
    elif unavailable_ids:
        status = VerificationStatus.INCONCLUSIVE
    elif (
        (not profile.decision.require_direct_evidence or direct_evidence_present)
        and supporting_score >= profile.decision.minimum_score
        and counter_score <= profile.decision.maximum_counter_evidence
    ):
        status = VerificationStatus.CONFIRMED
    else:
        status = VerificationStatus.INCONCLUSIVE

    return ProfileDecision(
        status=status,
        supporting_score=supporting_score,
        counter_score=counter_score,
        direct_evidence_present=direct_evidence_present,
        unavailable_assertion_ids=unavailable_ids,
    )
