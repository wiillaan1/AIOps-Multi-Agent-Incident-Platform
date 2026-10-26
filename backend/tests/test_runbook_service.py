"""
Tests for RunbookService - 自动生成 Runbook 草案并引用 RCA 日志证据。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.models.events import AlertEvent, RCAEvent, SeverityLevel
from app.services.runbook_service import RunbookService, runbook_service


# ============================ Helpers ============================


def _alert() -> AlertEvent:
    return AlertEvent(
        event_id="evt-test",
        alert_name="high_cpu",
        severity=SeverityLevel.HIGH,
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        correlation_id="corr-test",
        annotations={},
        labels={},
    )


def _rca(
    *,
    root_cause: str = "resource_exhaustion",
    confidence: float = 0.78,
    impact_chain: list[str] | None = None,
    contributing_factors: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    recommended_actions: list[str] | None = None,
    incident_id: str = "INC-2026-001",
) -> RCAEvent:
    return RCAEvent(
        event_id="rca-evt",
        correlation_id="corr-test",
        source="rca_agent",
        incident_id=incident_id,
        root_cause=root_cause,
        confidence=confidence,
        impact_chain=impact_chain or ["order-service", "payment-service"],
        contributing_factors=contributing_factors or ["resource_exhaustion", "traffic_spike"],
        evidence=evidence or {},
        recommended_actions=recommended_actions or ["scale_up", "check_resource_limits"],
        time_range_start=None,
        time_range_end=None,
    )


# ============================ Section rendering ============================


def test_section_overview_includes_metric_value_threshold():
    rca = _rca()
    sec = RunbookService._section_overview(rca, _alert())
    assert sec["kind"] == "overview"
    assert "order-service" in sec["text"]
    assert "cpu_usage_percent" in sec["text"]
    assert "95.0" in sec["text"]
    assert "80.0" in sec["text"]


def test_section_root_cause_renders_bayesian_top():
    rca = _rca(
        evidence={
            "bayesian_top_causes": [
                {"cause": "resource_exhaustion", "posterior": 0.65},
                {"cause": "traffic_spike", "posterior": 0.20},
            ],
        },
    )
    sec = RunbookService._section_root_cause(rca, rca.evidence)
    assert "resource_exhaustion" in sec["text"]
    assert "0.65" in sec["text"]


def test_section_evidence_summary_includes_rag():
    evidence = {
        "rag_matches": [
            {"id": "kb-001", "category": "performance", "confidence_boost": 0.2},
        ],
        "bayesian_top_causes": [{"cause": "resource_exhaustion", "posterior": 0.5}],
        "log_evidence": {"available": True, "cause_hits": {"resource_exhaustion": 3}},
    }
    sec = RunbookService._section_evidence_summary(evidence)
    assert "kb-001" in sec["text"]
    assert "resource_exhaustion" in sec["text"]


def test_section_log_evidence_unavailable():
    sec = RunbookService._section_log_evidence({}, "order-service")
    assert sec["available"] is False
    assert "不可用" in sec["text"]


def test_section_log_evidence_renders_samples():
    log = {
        "available": True,
        "entries_count": 5,
        "cause_hits": {"resource_exhaustion": 5, "code_bug": 1},
        "confidence_boost": 0.10,
        "effective_boost": 0.10,
        "corroborates_bayesian": True,
        "diverges_from_bayesian": False,
        "samples": {
            "resource_exhaustion": [
                "ERROR OOM killed container",
                "ERROR java.lang.OutOfMemoryError",
            ],
        },
    }
    sec = RunbookService._section_log_evidence(log, "order-service")
    assert sec["available"] is True
    assert "佐证" in sec["text"]
    assert "OOM killed container" in sec["text"]
    assert "root_cause`resource_exhaustion`" in sec["text"] or "resource_exhaustion" in sec["text"]


def test_section_log_evidence_marks_divergence():
    log = {
        "available": True,
        "entries_count": 3,
        "cause_hits": {"code_bug": 3},
        "confidence_boost": 0.06,
        "effective_boost": 0.018,
        "corroborates_bayesian": False,
        "diverges_from_bayesian": True,
        "samples": {"code_bug": ["NullPointerException at line 42"]},
    }
    sec = RunbookService._section_log_evidence(log, "order-service")
    assert "分歧" in sec["text"]


def test_section_rollback_uses_root_cause_mapping():
    sec = RunbookService._section_rollback(_rca(root_cause="recent_deployment"), "order-service")
    assert "kubectl rollout undo" in sec["text"]
    assert "order-service" in sec["text"]


def test_section_rollback_falls_back_for_unknown_cause():
    sec = RunbookService._section_rollback(_rca(root_cause="exotic_unknown_cause_xyz"), "order-service")
    assert "保留现场" in sec["text"]


def test_section_recommended_actions_lists_all():
    rca = _rca(recommended_actions=["scale_up", "enable_rate_limiting", "notify_oncall"])
    sec = RunbookService._section_recommended_actions(rca)
    assert sec["actions"] == ["scale_up", "enable_rate_limiting", "notify_oncall"]
    assert "1." in sec["text"] and "2." in sec["text"] and "3." in sec["text"]


def test_section_verification_references_alert_metric():
    sec = RunbookService._section_verification(_alert())
    assert "cpu_usage_percent" in sec["text"]
    assert "order-service" in sec["text"]


# ============================ generate() end-to-end ============================


def test_generate_returns_dict_with_markdown():
    rca = _rca()
    draft = RunbookService().generate(rca, _alert())
    assert draft["title"].startswith("[Runbook]")
    assert draft["service"] == "order-service"
    assert draft["root_cause"] == "resource_exhaustion"
    assert draft["confidence"] == 0.78
    assert "## 概述" in draft["markdown"]
    assert "## 根因结论" in draft["markdown"]
    assert "## 回滚方案" in draft["markdown"]
    assert "## 验证步骤" in draft["markdown"]
    assert draft["source"]["type"] == "auto_generated"


def test_generate_includes_log_evidence_section_when_available():
    rca = _rca(
        evidence={
            "log_evidence": {
                "available": True,
                "entries_count": 4,
                "cause_hits": {"network_issue": 4},
                "confidence_boost": 0.08,
                "effective_boost": 0.08,
                "corroborates_bayesian": True,
                "diverges_from_bayesian": False,
                "samples": {"network_issue": ["Connection timeout"]},
            }
        },
    )
    draft = RunbookService().generate(rca, _alert())
    assert "## 日志证据" in draft["markdown"]
    assert "Connection timeout" in draft["markdown"]
    assert draft["source"]["log_evidence_available"] is True
    # find the log_evidence section
    log_secs = [s for s in draft["sections"] if s["kind"] == "log_evidence"]
    assert len(log_secs) == 1
    assert log_secs[0]["available"] is True
    assert log_secs[0]["samples_count"] == 1


def test_generate_handles_missing_evidence_gracefully():
    rca = _rca(evidence={})
    draft = RunbookService().generate(rca, _alert())
    assert "## 证据汇总" in draft["markdown"]
    assert "## 日志证据" in draft["markdown"]
    assert "不可用" in draft["markdown"]


def test_generate_works_without_alert():
    rca = _rca()
    draft = RunbookService().generate(rca, alert=None)
    assert draft["service"] == "unknown"
    assert "未知服务" in draft["markdown"]


def test_generate_uses_singleton():
    assert runbook_service is RunbookService() or isinstance(runbook_service, RunbookService)
    # 调用通过单例入口应与直接构造等价
    rca = _rca()
    a = runbook_service.generate(rca, _alert())
    b = RunbookService().generate(rca, _alert())
    assert a["title"] == b["title"]
    assert a["confidence"] == b["confidence"]


def test_render_markdown_includes_all_sections():
    md = RunbookService._render_markdown(
        title="[Runbook] test",
        sections=[
            {"heading": "A", "text": "aa"},
            {"heading": "B", "text": "bb"},
        ],
        metadata=["- m1", "- m2"],
    )
    assert md.startswith("# [Runbook] test")
    assert "## A" in md and "## B" in md
    assert "- m1" in md and "- m2" in md