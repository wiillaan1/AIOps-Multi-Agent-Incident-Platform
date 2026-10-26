"""W8.1 integration test — verify routes.py:verify_phase actually writes entity_snapshot.

模拟 verify_phase 调用，验证 W8.1 接入在主链路中能正确生成 entity_snapshot。
此测试不依赖 FastAPI client，直接调用 MainPathAuditEnvelope。
"""

from __future__ import annotations

import os

import pytest

from app.services.main_path_audit import (
    MainPathAuditEnvelope,
    is_w8_1_audit_enabled,
)


def test_feature_flag_off_no_snapshot_in_real_context(monkeypatch):
    """feature flag 关闭时，主链路 verify_phase 末尾不写 entity_snapshot。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "false")
    assert is_w8_1_audit_enabled() is False

    real_ctx = {
        "incident_id": "INC-REAL-001",
        "service": "checkout-api",
        "audit_trail": [
            {"step": "triage", "trigger": "x", "outcome": "confirmed"},
        ],
    }
    snapshot = MainPathAuditEnvelope().upgrade(real_ctx)
    assert snapshot is None
    assert "entity_snapshot" not in real_ctx


def test_feature_flag_on_writes_snapshot(monkeypatch):
    """feature flag 开启时，主链路 verify_phase 末尾写 entity_snapshot。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    real_ctx = {
        "incident_id": "INC-REAL-002",
        "service": "checkout-api",
        "alert_event": {
            "service": "checkout-api",
            "metric": "p99_latency_ms",
            "value": 2300,
        },
        "rca_event": {
            "root_cause": "database_issue",
            "confidence": 0.85,
            "impact_chain": ["payment-service"],
        },
        "audit_trail": [
            {"step": "triage", "trigger": "p99=2300", "actions_taken": ["3-sigma"], "outcome": "anomaly"},
            {"step": "rca_bayesian", "trigger": "symptoms=high_latency", "outcome": "database_issue=0.45"},
            {"step": "heal_dry_run", "trigger": "playbook=kill_long_query", "outcome": "expected_p99<500"},
            {"step": "verify_reflection", "trigger": "post_value=480", "actions_taken": ["next_action=complete"]},
        ],
        "verification": {
            "rca_match_status": "match",
            "rca_final_confidence": 0.85,
            "heal_recovered": True,
        },
    }

    snapshot = MainPathAuditEnvelope().upgrade(real_ctx)

    assert snapshot is not None
    assert "entity_snapshot" in real_ctx
    assert isinstance(snapshot["evidence_blocks"], list)
    assert len(snapshot["evidence_blocks"]) == 4
    assert snapshot["root_cause"] == "database_issue"
    assert "checkout-api" in snapshot["services"]


def test_audit_trail_format_unchanged_for_badcase_capture(monkeypatch):
    """W8.1 升级后 audit_trail 字段格式不变，BadCase 触发器仍可读。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")

    real_ctx = {
        "incident_id": "INC-REAL-003",
        "audit_trail": [
            {"step": "triage", "trigger": "x", "outcome": "y"},
            {"step": "verify_reflection", "actions_taken": ["next_action=escalate"], "outcome": "low_confidence"},
        ],
        "verification": {"heal_recovered": False},
    }

    snapshot = MainPathAuditEnvelope().upgrade(real_ctx)
    assert snapshot is not None

    # BadCase 触发器读 audit_trail 的格式不能变
    trail = real_ctx["audit_trail"]
    assert len(trail) == 2
    for entry in trail:
        assert "step" in entry
        assert "actions_taken" in entry or "trigger" in entry

    # 关键是：能从 audit_trail 找到 escalate（被 BadCase 触发器需要）
    has_escalate = any(
        "next_action=escalate" in str(a)
        for entry in trail
        for a in entry.get("actions_taken", [])
    )
    assert has_escalate
