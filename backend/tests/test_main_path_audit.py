"""Tests for W8.1 MainPathAuditEnvelope.

验证主链路 audit_trail → entity_snapshot 升级不影响原文，并产出 W8 风格
EvidenceBlock 5 字段。
"""

from __future__ import annotations

import os

import pytest

from app.services.main_path_audit import (
    MainPathAuditEnvelope,
    get_main_path_audit_envelope,
    is_w8_1_audit_enabled,
)


# ===== Feature Flag =====


def test_feature_flag_default_disabled():
    """未配置 AIOPS_USE_W8_1_AUDIT 时默认关闭。"""
    saved = os.environ.pop("AIOPS_USE_W8_1_AUDIT", None)
    try:
        assert is_w8_1_audit_enabled() is False
    finally:
        if saved is not None:
            os.environ["AIOPS_USE_W8_1_AUDIT"] = saved


def test_feature_flag_enable_variants(monkeypatch):
    """'true'/'1'/'yes'/'on' 都能启用。"""
    for val in ["true", "1", "yes", "on", "TRUE", "Yes"]:
        monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", val)
        assert is_w8_1_audit_enabled() is True, f"failed for {val}"


def test_feature_flag_disabled_variants(monkeypatch):
    """'false'/'0'/'no'/'off' 都是关闭。"""
    for val in ["false", "0", "no", "off", "garbage"]:
        monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", val)
        assert is_w8_1_audit_enabled() is False, f"failed for {val}"


# ===== 升级测试 =====


def _sample_context() -> dict:
    """构造一个典型的 incident.context。"""
    return {
        "incident_id": "INC-001",
        "service": "order-service",
        "alert_event": {
            "service": "order-service",
            "metric": "p99_latency_ms",
            "value": 2300,
            "severity": "high",
        },
        "rca_event": {
            "root_cause": "database_issue",
            "confidence": 0.85,
            "impact_chain": ["payment-service", "mysql-primary"],
        },
        "rca_result": {
            "root_cause": "database_issue",
            "confidence": 0.85,
            "impact_services": ["payment-service", "mysql-primary"],
        },
        "audit_trail": [
            {
                "step": "triage",
                "trigger": "pre_value=2200",
                "actions_taken": ["3-sigma", "ewma"],
                "outcome": "confirmed_anomaly",
                "duration_ms": 50,
                "timestamp": 1722000000.0,
            },
            {
                "step": "rca_bayesian",
                "trigger": "symptoms=high_latency,high_error_rate",
                "actions_taken": ["compute_posterior"],
                "outcome": "database_issue=0.45",
                "duration_ms": 120,
                "timestamp": 1722000001.0,
            },
            {
                "step": "heal_dry_run",
                "trigger": "playbook=kill_long_query",
                "actions_taken": ["dry_run"],
                "outcome": "expected_p99<500ms",
                "duration_ms": 80,
                "timestamp": 1722000002.0,
            },
            {
                "step": "verify_reflection",
                "trigger": "post_value=480",
                "actions_taken": ["next_action=complete"],
                "outcome": "metrics_normalized",
                "duration_ms": 30,
                "timestamp": 1722000003.0,
            },
        ],
        "verification": {
            "rca_match_status": "match",
            "rca_final_confidence": 0.85,
            "heal_recovered": True,
            "plan_b_triggered": False,
        },
    }


def test_upgrade_writes_entity_snapshot(monkeypatch):
    """升级后 entity_snapshot 字段出现在 incident.context。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert snapshot is not None
    assert "entity_snapshot" in ctx
    assert ctx["entity_snapshot"] is snapshot


def test_upgrade_preserves_audit_trail(monkeypatch):
    """升级不修改 audit_trail 原数据。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()
    original_trail = list(ctx["audit_trail"])

    env.upgrade(ctx)

    assert ctx["audit_trail"] == original_trail
    assert len(ctx["audit_trail"]) == len(original_trail)


def test_upgrade_extracts_services(monkeypatch):
    """services 字段包含从多个来源提取的服务。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert "order-service" in snapshot["services"]
    assert "payment-service" in snapshot["services"]
    assert "mysql-primary" in snapshot["services"]


def test_upgrade_extracts_root_cause(monkeypatch):
    """root_cause 优先从 rca_event 取。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert snapshot["root_cause"] == "database_issue"


def test_upgrade_confidence_from_verification(monkeypatch):
    """confidence 优先取 rca_final_confidence。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert snapshot["confidence"] == 0.85


def test_upgrade_confidence_heal_recovered(monkeypatch):
    """confidence 兜底：heal_recovered=True → 0.85。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()
    ctx["verification"] = {"heal_recovered": True, "rca_match_status": "unknown"}

    snapshot = env.upgrade(ctx)

    assert snapshot["confidence"] == 0.85


def test_upgrade_confidence_plan_b(monkeypatch):
    """confidence 兜底：plan_b 触发 → 0.65。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()
    ctx["verification"] = {"plan_b_triggered": True}

    snapshot = env.upgrade(ctx)

    assert snapshot["confidence"] == 0.65


def test_upgrade_confidence_mismatch(monkeypatch):
    """confidence 兜底：mismatch → 0.55。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()
    ctx["verification"] = {"rca_match_status": "mismatch"}

    snapshot = env.upgrade(ctx)

    assert snapshot["confidence"] == 0.55


def test_upgrade_evidence_blocks_5_fields(monkeypatch):
    """每个 EvidenceBlock 都有 5 字段 + 2 个扩展字段。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert len(snapshot["evidence_blocks"]) == 4
    for block in snapshot["evidence_blocks"]:
        # 5 字段必有
        assert "block_id" in block
        assert "object_ref" in block
        assert "time" in block
        assert "observation" in block
        assert "mechanism" in block
        assert "confidence" in block
        assert "is_evidence_against_hypothesis" in block
        # 2 扩展字段
        assert "source_step" in block
        assert "source_kind" in block


def test_upgrade_evidence_blocks_mechanism_mapping(monkeypatch):
    """不同 step 映射到不同 mechanism。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    mechanisms = [b["mechanism"] for b in snapshot["evidence_blocks"]]
    assert "3sigma_ewma" in mechanisms  # triage
    assert "statistical_inference" in mechanisms  # rca_bayesian
    assert "playbook_match" in mechanisms  # heal_dry_run
    assert "w7_reflection" in mechanisms  # verify_reflection


def test_upgrade_steps_deduplicated(monkeypatch):
    """steps 字段去重保序。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()
    ctx["audit_trail"].append({
        "step": "triage",  # 重复
        "trigger": "x",
        "actions_taken": [],
        "outcome": "y",
    })

    snapshot = env.upgrade(ctx)

    steps = snapshot["steps"]
    assert steps.count("triage") == 1
    assert len(steps) == 4  # 去重后 4 个


def test_upgrade_empty_audit_trail_returns_none(monkeypatch):
    """空 audit_trail → 返回 None，不写 entity_snapshot。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = {"incident_id": "INC-001", "audit_trail": []}

    snapshot = env.upgrade(ctx)

    assert snapshot is None
    assert "entity_snapshot" not in ctx


def test_upgrade_disabled_returns_none(monkeypatch):
    """feature flag 关闭时直接返回 None。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "false")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert snapshot is None
    assert "entity_snapshot" not in ctx


def test_upgrade_handles_missing_keys_gracefully(monkeypatch):
    """context 缺少 rca_event / verification 等字段时不报错。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = {
        "incident_id": "INC-001",
        "audit_trail": [
            {"step": "triage", "trigger": "x", "actions_taken": [], "outcome": "y"},
        ],
    }

    snapshot = env.upgrade(ctx)

    assert snapshot is not None
    assert snapshot["root_cause"] is None
    assert snapshot["services"] == []
    assert snapshot["confidence"] == 0.0


def test_upgrade_handles_bad_rca_event(monkeypatch):
    """rca_event 是字符串而非 dict 时不报错。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = {
        "incident_id": "INC-001",
        "rca_event": "not a dict",  # 故意给字符串
        "audit_trail": [{"step": "triage", "trigger": "x"}],
    }

    snapshot = env.upgrade(ctx)

    assert snapshot is not None
    assert snapshot["root_cause"] is None


def test_upgrade_badcase_captured_id(monkeypatch):
    """badcase_captured_id 字段透传。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()
    ctx["verification"]["badcase_captured_id"] = "bc-abc123"

    snapshot = env.upgrade(ctx)

    assert snapshot["badcase_captured_id"] == "bc-abc123"


def test_upgrade_includes_schema_version(monkeypatch):
    """snapshot 必须包含 schema_version 字段。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    env = MainPathAuditEnvelope()
    ctx = _sample_context()

    snapshot = env.upgrade(ctx)

    assert snapshot["schema_version"] == "1.0"
    assert "created_at" in snapshot


def test_factory_returns_instance():
    """工厂函数返回 MainPathAuditEnvelope 实例。"""
    env = get_main_path_audit_envelope()
    assert isinstance(env, MainPathAuditEnvelope)
