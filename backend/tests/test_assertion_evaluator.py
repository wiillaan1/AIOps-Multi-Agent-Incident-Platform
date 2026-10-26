"""Tests for deterministic, query-safe root-cause assertion evaluation."""

from __future__ import annotations

import pytest

from app.models.investigation import (
    AssertionOutcome,
    AssertionRule,
    CounterEvidenceRule,
    EvidenceSource,
    RootCauseVerificationProfile,
    VerificationStatus,
)
from app.services.assertion_evaluator import (
    AssertionEvaluation,
    QueryContext,
    QueryResult,
    evaluate_assertion,
    evaluate_profile,
    render_query,
)


def _context(**labels: str) -> QueryContext:
    return QueryContext(
        service="orders",
        target="orders-api",
        metric="cpu_usage_percent",
        environment="prod",
        labels=labels,
    )


def _rule(operator: str, **expectation: object) -> AssertionRule:
    return AssertionRule.model_validate(
        {
            "id": "cpu",
            "source": "prometheus",
            "query_template": (
                'cpu_usage{service="{{ service }}",target="{{ target }}"}'
            ),
            "evidence_level": "direct",
            "weight": 1.0,
            "expectation": {"operator": operator, **expectation},
        }
    )


def _result(**kwargs: object) -> QueryResult:
    values: dict[str, object] = {
        "source": "prometheus",
        "available": True,
        "value": 95,
        "hits": 1,
        "summary": {},
    }
    values.update(kwargs)
    return QueryResult(**values)


def _profile() -> RootCauseVerificationProfile:
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
                    "weight": 0.7,
                    "expectation": {"operator": "greater_than", "threshold": 10},
                },
                {
                    "id": "supporting",
                    "source": "loki",
                    "query_template": '{service="{{ service }}",target="{{ target }}"} |= "spike"',
                    "evidence_level": "supporting",
                    "weight": 0.2,
                    "expectation": {"operator": "min_hits", "threshold": 1},
                },
            ],
            "counter_evidence": [
                {
                    "id": "counter",
                    "source": "prometheus",
                    "query_template": 'rate{service="{{ service }}",target="{{ target }}"}',
                    "weight": 0.5,
                    "expectation": {"operator": "less_than", "threshold": 5},
                }
            ],
            "decision": {
                "minimum_score": 0.65,
                "require_direct_evidence": True,
                "maximum_counter_evidence": 0.3,
            },
            "next_strategies": {
                "inconclusive": ["expand_window"],
                "contradicted": ["test_alternative"],
            },
            "recovery": {
                "settle_seconds": 0,
                "observation_seconds": 1,
                "consecutive_samples": 1,
            },
        }
    )


def _counter_rule() -> CounterEvidenceRule:
    return _profile().counter_evidence[0]


def test_render_query_uses_only_declared_context_values_and_escapes_labels():
    query = render_query(
        '{service="{{ service }}",target="{{ target }}",env="{{ environment }}",'
        'cluster="{{ cluster }}"}',
        _context(
            cluster='prod"} or up{service="attacker',
            service="must-not-override-context-service",
        ),
    )

    assert 'service="orders"' in query
    assert 'target="orders-api"' in query
    assert 'env="prod"' in query
    assert 'cluster="prod\\"} or up{service=\\"attacker"' in query


@pytest.mark.parametrize(
    "template",
    [
        '{service="{{ service }}",target="{{ target }}",zone="{{ unknown }}"}',
        '{service="{{ service }}",target="{{ target }}"} {{ unknown }}',
        '{service="{{ service }}",target="{{ target }}"} {legacy_placeholder}',
        'up{service="{{ service }}"}',
        'up',
        '{service="{{ service }}",target="{{ target }}"} or up',
        '{service="{{ service }}",target="{{ target }}"} unless up',
        'sum(up{job="api"}) + max(up{service="{{ service }}",target="{{ target }}"})',
    ],
)
def test_render_query_rejects_unknown_or_unscoped_templates(template: str):
    with pytest.raises(ValueError):
        render_query(template, _context())


@pytest.mark.parametrize(
    ("rule", "result"),
    [
        (_rule("greater_than", threshold=80), _result(value=95)),
        (_rule("less_than", threshold=100), _result(value=95)),
        (_rule("between", minimum=90, maximum=100), _result(value=95)),
        (_rule("min_hits", threshold=2), _result(hits=2)),
        (_rule("contains", threshold="timeout"), _result(summary={"line": "request timeout"})),
    ],
)
def test_supported_operators_match_as_pass(rule: AssertionRule, result: QueryResult):
    assert evaluate_assertion(rule, result).outcome is AssertionOutcome.PASS


def test_nonmatching_ordinary_assertion_is_fail():
    assert evaluate_assertion(_rule("greater_than", threshold=100), _result(value=95)).outcome is AssertionOutcome.FAIL


def test_query_evidence_sources_are_typed_enums():
    result = _result()
    evaluation = evaluate_assertion(_rule("greater_than", threshold=80), result)

    assert result.source is EvidenceSource.PROMETHEUS
    assert evaluation.source is EvidenceSource.PROMETHEUS


def test_render_query_accepts_a_single_fully_scoped_selector():
    assert render_query(
        'sum(rate(up{service="{{ service }}",target="{{ target }}"}[5m]))',
        _context(),
    ) == 'sum(rate(up{service="orders",target="orders-api"}[5m]))'


def test_metric_placeholder_rejects_an_expression_injection_vector():
    context = QueryContext(
        service="orders",
        target="orders-api",
        metric="up or on() vector(1)",
        environment="prod",
        labels={},
    )

    with pytest.raises(ValueError, match="metric"):
        render_query(
            '{{ metric }}{service="{{ service }}",target="{{ target }}"}',
            context,
        )


def test_assertion_evaluation_source_string_is_normalized_to_evidence_source():
    evaluation = AssertionEvaluation(
        assertion_id="cpu",
        outcome=AssertionOutcome.PASS,
        evidence_level="direct",
        weight=1.0,
        source="prometheus",
        available=True,
    )

    assert evaluation.source is EvidenceSource.PROMETHEUS


def test_matching_counter_rule_is_counter_and_nonmatching_is_fail():
    assert evaluate_assertion(_counter_rule(), _result(value=3)).outcome is AssertionOutcome.COUNTER
    assert evaluate_assertion(_counter_rule(), _result(value=8)).outcome is AssertionOutcome.FAIL


def test_unavailable_data_is_never_counter_even_for_counter_rule():
    evaluation = evaluate_assertion(
        _counter_rule(),
        _result(available=False, value=None, hits=0, error="timeout"),
    )

    assert evaluation.outcome is AssertionOutcome.UNAVAILABLE


def test_profile_is_confirmed_with_direct_support_and_no_strong_counter():
    profile = _profile()
    evaluations = [
        evaluate_assertion(profile.assertions[0], _result(value=20)),
        evaluate_assertion(profile.assertions[1], _result(source="loki", value=None, hits=1)),
        evaluate_assertion(profile.counter_evidence[0], _result(value=10)),
    ]

    decision = evaluate_profile(profile, evaluations)

    assert decision.status is VerificationStatus.CONFIRMED
    assert decision.supporting_score == pytest.approx(0.9)
    assert decision.counter_score == pytest.approx(0.0)


def test_profile_is_contradicted_when_counter_reaches_policy_limit():
    profile = _profile()
    evaluations = [
        evaluate_assertion(profile.assertions[0], _result(value=20)),
        evaluate_assertion(profile.counter_evidence[0], _result(value=3)),
    ]

    assert evaluate_profile(profile, evaluations).status is VerificationStatus.CONTRADICTED


def test_profile_is_inconclusive_when_evaluation_is_marked_unavailable_despite_pass():
    profile = _profile()
    evaluations = [
        AssertionEvaluation("direct", AssertionOutcome.PASS, "direct", 0.7, "prometheus", False),
        AssertionEvaluation("supporting", AssertionOutcome.PASS, "supporting", 0.2, "loki", True),
        AssertionEvaluation("counter", AssertionOutcome.FAIL, None, 0.5, "prometheus", True),
    ]

    decision = evaluate_profile(profile, evaluations)

    assert decision.status is VerificationStatus.INCONCLUSIVE
    assert decision.unavailable_assertion_ids == ("direct",)


def test_manual_only_profile_is_never_automatically_confirmed():
    profile = _profile().model_copy(update={"manual_only": True})
    profile.decision.minimum_score = 0.0
    profile.decision.require_direct_evidence = False

    decision = evaluate_profile(profile, [])

    assert decision.status is VerificationStatus.MANUAL_ONLY


@pytest.mark.parametrize("missing_evaluation", ["direct", "counter"])
def test_profile_is_inconclusive_when_a_required_rule_was_not_evaluated(missing_evaluation: str):
    profile = _profile()
    evaluations = [
        evaluate_assertion(profile.assertions[0], _result(value=20)),
        evaluate_assertion(profile.assertions[1], _result(source="loki", value=None, hits=1)),
        evaluate_assertion(profile.counter_evidence[0], _result(value=10)),
    ]
    evaluations = [evaluation for evaluation in evaluations if evaluation.assertion_id != missing_evaluation]
    if missing_evaluation == "direct":
        evaluations = [evaluation for evaluation in evaluations if evaluation.assertion_id != "direct"]
    else:
        evaluations = [evaluation for evaluation in evaluations if evaluation.assertion_id != "counter"]

    decision = evaluate_profile(profile, evaluations)

    assert decision.status is VerificationStatus.INCONCLUSIVE
    assert missing_evaluation in decision.unavailable_assertion_ids


def test_profile_is_inconclusive_when_counter_source_is_unavailable():
    profile = _profile()
    evaluations = [
        evaluate_assertion(profile.assertions[0], _result(value=20)),
        evaluate_assertion(profile.assertions[1], _result(source="loki", value=None, hits=1)),
        evaluate_assertion(profile.counter_evidence[0], _result(available=False, value=None, hits=0)),
    ]

    decision = evaluate_profile(profile, evaluations)

    assert decision.status is VerificationStatus.INCONCLUSIVE
    assert decision.unavailable_assertion_ids == ("counter",)


@pytest.mark.parametrize(
    "evaluations",
    [
        [],
        [evaluate_assertion(_profile().assertions[1], _result(source="loki", value=None, hits=1))],
        [evaluate_assertion(_profile().assertions[0], _result(available=False, value=None, hits=0))],
    ],
)
def test_profile_is_inconclusive_without_direct_evidence_or_available_data(evaluations: list):
    assert evaluate_profile(_profile(), evaluations).status is VerificationStatus.INCONCLUSIVE
