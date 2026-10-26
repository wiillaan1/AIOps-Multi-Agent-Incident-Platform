"""
Tests for the Reflection Layer (app.agents.reflection).
"""

from __future__ import annotations

from app.agents.reflection import (
    PERMANENT_ERRORS,
    REFLECTION_REASONS,
    ReflectionResult,
    VerifyEvidence,
    append_audit_trail,
    reflect_missing_info,
    reflect_on_tool_failure,
    reflect_on_verify,
)


# ============================ reflect_on_verify ============================


def test_verify_complete_when_confidence_high_and_symptoms_resolved():
    evidence = VerifyEvidence(
        confidence=0.85,
        symptoms_resolved=4,
        symptoms_total=5,
        tool_failures=[],
    )
    r = reflect_on_verify(evidence, retry_count=0)
    assert r.next_action == "complete"


def test_verify_retry_when_recovery_below_50_percent():
    evidence = VerifyEvidence(
        confidence=0.9,           # 置信度高
        symptoms_resolved=1,      # 但恢复不够
        symptoms_total=5,
    )
    r = reflect_on_verify(evidence, retry_count=0)
    assert r.next_action == "retry"
    assert r.reason == "incomplete_recovery"
    assert "20%" in r.reason_detail


def test_verify_escalate_when_low_confidence():
    evidence = VerifyEvidence(
        confidence=0.4,
        symptoms_resolved=5,
        symptoms_total=5,
    )
    r = reflect_on_verify(evidence, retry_count=0)
    assert r.next_action == "escalate"
    assert r.reason == "low_confidence"


def test_verify_escalate_when_max_retry_exceeded():
    evidence = VerifyEvidence(
        confidence=0.9,
        symptoms_resolved=2,
        symptoms_total=5,
    )
    r = reflect_on_verify(evidence, retry_count=2, max_retry=2)
    assert r.next_action == "escalate"
    assert r.reason == "max_retry_exceeded"


def test_verify_escalate_when_tool_failures_accumulate():
    evidence = VerifyEvidence(
        confidence=0.9,
        symptoms_resolved=5,
        symptoms_total=5,
        tool_failures=["err1", "err2", "err3"],
    )
    r = reflect_on_verify(evidence, retry_count=0)
    assert r.next_action == "escalate"
    assert r.reason == "tool_failure"


def test_verify_retry_before_max_with_low_recovery():
    evidence = VerifyEvidence(
        confidence=0.7,
        symptoms_resolved=2,
        symptoms_total=5,
    )
    r = reflect_on_verify(evidence, retry_count=0, max_retry=3)
    assert r.next_action == "retry"


def test_verify_handles_zero_symptoms_gracefully():
    """没有症状数据时不应当因为除零崩溃。"""
    evidence = VerifyEvidence(confidence=0.9, symptoms_resolved=0, symptoms_total=0)
    r = reflect_on_verify(evidence)
    # 没有 symptoms_total 数据时跳过"恢复比例"检查，按置信度判断
    assert r.next_action == "complete"


def test_reflection_result_includes_duration():
    evidence = VerifyEvidence(confidence=0.9, symptoms_resolved=5, symptoms_total=5)
    r = reflect_on_verify(evidence)
    assert r.duration_ms >= 0
    assert isinstance(r, ReflectionResult)


# ============================ reflect_on_tool_failure ============================


def test_tool_failure_retry_on_transient_error():
    r = reflect_on_tool_failure("query_logs", "ConnectionTimeout", attempt=0)
    assert r.next_action == "retry"
    assert "query_logs" in r.reason_detail


def test_tool_failure_escalate_on_permanent_error():
    r = reflect_on_tool_failure("query_logs", "permission_denied", attempt=0)
    assert r.next_action == "escalate"


def test_tool_failure_escalate_when_max_attempts_reached():
    r = reflect_on_tool_failure("query_logs", "timeout", attempt=3, max_attempts=3)
    assert r.next_action == "escalate"
    assert r.reason == "max_retry_exceeded"


# ============================ reflect_missing_info ============================


def test_missing_info_returns_questions_for_missing_fields():
    questions = reflect_missing_info(
        intent="fault_diagnosis",
        parsed_fields={"symptom": "high_latency"},  # 缺 service
        required_fields=["service", "symptom", "time_window"],
    )
    assert len(questions) == 2
    assert any("服务" in q for q in questions)
    assert any("什么时候" in q for q in questions)


def test_missing_info_empty_when_all_fields_present():
    questions = reflect_missing_info(
        intent="fault_diagnosis",
        parsed_fields={"service": "order", "symptom": "high_latency"},
        required_fields=["service", "symptom"],
    )
    assert questions == []


def test_missing_info_handles_unknown_field():
    questions = reflect_missing_info(
        intent="custom",
        parsed_fields={},
        required_fields=["unknown_field"],
    )
    assert questions == []  # 未知字段不追问


# ============================ append_audit_trail ============================


def test_audit_trail_appends_entry():
    ctx: dict = {}
    append_audit_trail(
        ctx, step="verify_reflection", trigger="low_confidence",
        actions_taken=["retry"], outcome="retry_count=1", duration_ms=15,
    )
    assert "audit_trail" in ctx
    assert len(ctx["audit_trail"]) == 1
    entry = ctx["audit_trail"][0]
    assert entry["step"] == "verify_reflection"
    assert entry["duration_ms"] == 15
    assert "timestamp" in entry


def test_audit_trail_caps_at_100():
    ctx: dict = {}
    for i in range(120):
        append_audit_trail(
            ctx, step=f"step-{i}", trigger="x",
            actions_taken=[], outcome="ok", duration_ms=0,
        )
    assert len(ctx["audit_trail"]) == 100
    # 应该是最近 100 条，丢掉前 20 条
    assert ctx["audit_trail"][0]["step"] == "step-20"
    assert ctx["audit_trail"][-1]["step"] == "step-119"


# ============================ Constants sanity ============================


def test_reflection_reasons_is_closed_set():
    assert REFLECTION_REASONS == {
        "low_confidence",
        "incomplete_recovery",
        "no_match",
        "tool_failure",
        "inconsistent_evidence",
        "missing_info",
        "max_retry_exceeded",
        "all_candidates_too_risky",
    }


def test_permanent_errors_contains_known_classes():
    for kw in ("permission_denied", "invalid_input", "service_not_found"):
        assert kw in PERMANENT_ERRORS
