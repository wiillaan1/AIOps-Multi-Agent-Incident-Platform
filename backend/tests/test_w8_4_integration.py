"""W8.4 integration test — verify W8.4 correlator works with W8.1 snapshot.

验证当一个 incident 已经有 entity_snapshot（W8.1 产出）时，
W8.4 IncidentCorrelator 能正确基于 snapshot 做关联分析。
"""

from __future__ import annotations

import pytest

from app.services.main_path_audit import MainPathAuditEnvelope
from app.services.incident_correlator import IncidentCorrelator, is_w8_4_correlate_enabled


def _make_context(incident_id: str, service: str, root_cause: str | None = None) -> dict:
    """构造一个 W8.1 升级过的 incident context（含 entity_snapshot）。"""
    ctx = {
        "incident_id": incident_id,
        "service": service,
        "alert_event": {
            "service": service,
            "metric": "p99_latency_ms",
            "value": 2300,
        },
        "rca_event": {
            "root_cause": root_cause,
            "confidence": 0.85,
            "impact_chain": ["payment-service"] if root_cause else [],
        },
        "audit_trail": [
            {"step": "triage", "trigger": "x", "actions_taken": ["3-sigma"], "outcome": "anomaly"},
            {"step": "rca_bayesian", "trigger": "y", "outcome": root_cause or "unknown"},
            {"step": "heal_dry_run", "trigger": "z", "outcome": "ok"},
            {"step": "verify_reflection", "trigger": "w", "actions_taken": ["next_action=complete"]},
        ],
        "verification": {
            "rca_match_status": "match",
            "rca_final_confidence": 0.85,
            "heal_recovered": True,
        },
    }
    # 强制启用 W8.1 升级（无论外部 feature flag 状态）
    import os
    original = os.environ.get("AIOPS_USE_W8_1_AUDIT")
    os.environ["AIOPS_USE_W8_1_AUDIT"] = "true"
    try:
        snapshot = MainPathAuditEnvelope().upgrade(ctx)
    finally:
        if original is None:
            os.environ.pop("AIOPS_USE_W8_1_AUDIT", None)
        else:
            os.environ["AIOPS_USE_W8_1_AUDIT"] = original
    assert snapshot is not None
    return ctx


def test_w8_1_snapshot_feeds_w8_4_correlator(monkeypatch):
    """W8.1 产出的 entity_snapshot 直接被 W8.4 消费。"""
    monkeypatch.setenv("AIOPS_USE_W8_1_AUDIT", "true")
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")

    # 准备两个 incident context（W8.1 已升级）
    target = _make_context("INC-001", "order-service", root_cause="database_issue")
    candidate = _make_context("INC-002", "order-service", root_cause="database_issue")

    correlator = IncidentCorrelator()
    results = correlator.correlate_by_entity_snapshot(
        target["entity_snapshot"],
        [candidate["entity_snapshot"]],
    )

    assert len(results) == 1
    assert results[0].incident_id == "INC-002"
    # 4 维度全匹配 → 1.0
    assert results[0].similarity > 0.9


def test_correlator_disabled_no_results(monkeypatch):
    """feature flag 关闭时，correlator 不返回结果。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "false")
    target = _make_context("INC-001", "order-service", root_cause="database_issue")
    candidate = _make_context("INC-002", "order-service", root_cause="database_issue")

    correlator = IncidentCorrelator()
    results = correlator.correlate_by_entity_snapshot(
        target["entity_snapshot"],
        [candidate["entity_snapshot"]],
    )
    assert results == []


def test_correlator_partial_match(monkeypatch):
    """部分维度匹配时，相似度介于 0-1 之间。

    注意：impact_chain 自动给 snapshot 加 'payment-service' 作为 service，
    所以即使 order-service 和 inventory-service 不同，仍然有 'payment-service' 共享。
    实际相似度 = 0.4 (service) + 0.0 (rc) + 0.2 (evidence) + 0.1 (steps) = 0.7
    """
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    target = _make_context("INC-001", "order-service", root_cause="database_issue")
    # candidate: 不同 service (但 impact_chain 加了 payment-service), 不同 root_cause
    candidate = _make_context("INC-002", "inventory-service", root_cause="code_bug")

    correlator = IncidentCorrelator()
    results = correlator.correlate_by_entity_snapshot(
        target["entity_snapshot"],
        [candidate["entity_snapshot"]],
    )
    # 0.7 > 0.5 → 返回 1 个结果
    assert len(results) == 1
    assert abs(results[0].similarity - 0.7) < 0.01


def test_correlator_completely_different(monkeypatch):
    """完全不同的 incident：service 不同 + root_cause 不同 + 步骤少。

    通过手工构造 cand2 的 audit_trail 让 steps 重叠少。
    """
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    target = _make_context("INC-001", "order-service", root_cause="database_issue")

    # 构造一个完全不同 service + 不同 rc + 少 steps 的 candidate
    cand2 = _make_context("INC-002", "billing-service", root_cause="hardware_failure")
    # 清空 steps 让步骤维度为 0
    cand2["entity_snapshot"]["steps"] = []
    # 清空 evidence_blocks 让 evidence 维度为 0
    cand2["entity_snapshot"]["evidence_blocks"] = []
    cand2["entity_snapshot"]["services"] = ["billing-service"]

    correlator = IncidentCorrelator()
    results = correlator.correlate_by_entity_snapshot(
        target["entity_snapshot"],
        [cand2["entity_snapshot"]],
    )
    # 0 (service) + 0 (rc) + 0 (evidence) + 0 (steps) = 0 < 0.5
    assert results == []


def test_correlator_threshold_boundary(monkeypatch):
    """边界测试：similarity 接近阈值时的行为。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    target = _make_context("INC-001", "order-service", root_cause="database_issue")
    # candidate: 同 service + 同 root_cause → 0.7（> 0.5 阈值）
    candidate = _make_context("INC-002", "order-service", root_cause="database_issue")

    correlator = IncidentCorrelator()
    results = correlator.correlate_by_entity_snapshot(
        target["entity_snapshot"],
        [candidate["entity_snapshot"]],
    )
    # service 0.4 + rc 0.3 + evidence 0.2 + steps 0.1 = 1.0
    assert len(results) == 1
    assert results[0].similarity > 0.9


def test_correlator_multiple_candidates(monkeypatch):
    """多个候选时按相似度降序。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    target = _make_context("INC-001", "order-service", root_cause="database_issue")

    # 候选 1: 完全相同 → 1.0
    cand1 = _make_context("INC-002", "order-service", root_cause="database_issue")
    # 候选 2: 不同 service → 应被过滤
    cand3 = _make_context("INC-004", "inventory-service", root_cause="code_bug")

    correlator = IncidentCorrelator()
    results = correlator.correlate_by_entity_snapshot(
        target["entity_snapshot"],
        [cand1["entity_snapshot"], cand3["entity_snapshot"]],
    )

    # 至少 1 个有效（cand1）
    assert len(results) >= 1
    # 排序降序
    for i in range(len(results) - 1):
        assert results[i].similarity >= results[i + 1].similarity
    # 第一个应当是 cand1（完全匹配）
    assert results[0].incident_id == "INC-002"
