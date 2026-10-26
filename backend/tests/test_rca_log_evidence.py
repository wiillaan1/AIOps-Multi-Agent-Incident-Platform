"""
Tests for RCA Agent's Loki log-evidence integration.

- _query_log_evidence is mocked (no real Loki required)
- _synthesize_analysis is verified to consume log_evidence and update confidence
- End-to-end process() must include log_evidence in evidence dict
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.rca_agent import (
    LOG_SIGNATURE_TO_CAUSE,
    RCAInput,
    _empty_log_evidence,
)
from app.agents.rca_agent import RCAAgent
from app.models.events import AlertEvent, SeverityLevel


# ============================ Helpers ============================


def _alert(service: str = "order-service", metric: str = "cpu_usage_percent") -> AlertEvent:
    return AlertEvent(
        event_id="evt-test",
        alert_id=1,
        alert_name="high_cpu_test",
        severity=SeverityLevel.HIGH,
        service=service,
        metric=metric,
        value=92.0,
        threshold=80.0,
        correlation_id="corr-test",
        annotations={},
        labels={},
    )


def _log_entries(*lines: str) -> list[dict[str, Any]]:
    return [
        {"timestamp": f"2026-07-17T10:0{i}:00Z", "labels": {"service": "x"}, "line": line}
        for i, line in enumerate(lines)
    ]


# ============================ Signature map ============================


def test_signature_map_covers_common_causes():
    """每条内置根因至少有一条签名覆盖。"""
    covered_causes = set(LOG_SIGNATURE_TO_CAUSE.values())
    assert "resource_exhaustion" in covered_causes
    assert "code_bug" in covered_causes
    assert "dependency_failure" in covered_causes
    assert "network_issue" in covered_causes
    assert "database_issue" in covered_causes
    assert "recent_deployment" in covered_causes


# ============================ _query_log_evidence ============================


@pytest.mark.asyncio
async def test_query_log_evidence_returns_empty_when_no_entries():
    """Loki 返回空时不报错，返回 available=True 但无票数。"""
    with patch(
        "app.infrastructure.log_client.LokiClient.get_service_errors",
        AsyncMock(return_value=[]),
    ):
        agent = RCAAgent()
        evidence = await agent._query_log_evidence(_alert(), lookback_minutes=30)

    assert evidence["available"] is True
    assert evidence["entries_count"] == 0
    assert evidence["boosted_cause"] is None
    assert evidence["confidence_boost"] == 0.0
    assert evidence["samples"] == {}
    assert evidence["source"] == "loki"


@pytest.mark.asyncio
async def test_query_log_evidence_aggregates_cause_votes():
    """多条同类日志应累加票数，并选最高票为 boosted_cause。"""
    entries = _log_entries(
        "ERROR OOM killed container",
        "ERROR OutOfMemoryError raised in worker",
        "WARN memory limit approaching",
    )
    with patch(
        "app.infrastructure.log_client.LokiClient.get_service_errors",
        AsyncMock(return_value=entries),
    ):
        agent = RCAAgent()
        evidence = await agent._query_log_evidence(_alert(), lookback_minutes=30)

    assert evidence["available"] is True
    assert evidence["entries_count"] == 3
    assert evidence["cause_hits"]["resource_exhaustion"] == 3
    assert evidence["boosted_cause"] == "resource_exhaustion"
    # 3 票 * 0.02 = 0.06
    assert evidence["confidence_boost"] == pytest.approx(0.06, abs=1e-4)
    # samples 限制每个 cause 最多 3 条
    assert len(evidence["samples"]["resource_exhaustion"]) == 3


@pytest.mark.asyncio
async def test_query_log_evidence_boost_capped_at_010():
    """票数很多时加成封顶 0.10。"""
    entries = _log_entries(*[f"ERROR oom event {i}" for i in range(20)])
    with patch(
        "app.infrastructure.log_client.LokiClient.get_service_errors",
        AsyncMock(return_value=entries),
    ):
        agent = RCAAgent()
        evidence = await agent._query_log_evidence(_alert(), lookback_minutes=30)

    assert evidence["confidence_boost"] <= 0.10


@pytest.mark.asyncio
async def test_query_log_evidence_handles_loki_failure():
    """Loki 不可达时返回 available=False，不抛异常。"""
    with patch(
        "app.infrastructure.log_client.LokiClient.get_service_errors",
        AsyncMock(side_effect=ConnectionError("loki down")),
    ):
        agent = RCAAgent()
        evidence = await agent._query_log_evidence(_alert(), lookback_minutes=30)

    assert evidence["available"] is False
    assert evidence["source"] == "unavailable"
    assert evidence["confidence_boost"] == 0.0


@pytest.mark.asyncio
async def test_query_log_evidence_sample_truncation():
    """长日志行被截断到 200 字符以内。"""
    long_line = "ERROR oom " + ("x" * 500)
    entries = _log_entries(long_line)
    with patch(
        "app.infrastructure.log_client.LokiClient.get_service_errors",
        AsyncMock(return_value=entries),
    ):
        agent = RCAAgent()
        evidence = await agent._query_log_evidence(_alert(), lookback_minutes=30)

    sample = evidence["samples"]["resource_exhaustion"][0]
    assert len(sample) <= 200


# ============================ _synthesize_analysis log integration ============================


@pytest.mark.asyncio
async def test_synthesize_corroboration_boosts_confidence():
    """当日志和贝叶斯都指向同一根因时，置信度应提升。"""
    agent = RCAAgent()

    # 构造一个简化的贝叶斯结果：top 是 resource_exhaustion
    from app.agents.rca_agent import BayesianNode, ServiceImpact

    bayesian = [
        BayesianNode(name="resource_exhaustion", prior=0.2, likelihood=0.9,
                     posterior=0.5, evidence_strength=0.8),
    ]
    rag: list[dict[str, Any]] = []
    impact = [ServiceImpact(service="order-service", hop_distance=0, tier="critical")]

    log_evidence = {
        "available": True,
        "entries_count": 5,
        "cause_hits": {"resource_exhaustion": 5},
        "boosted_cause": "resource_exhaustion",
        "confidence_boost": 0.10,
        "samples": {"resource_exhaustion": ["OOM killed"]},
        "source": "loki",
    }

    _, confidence_with_log, evidence_with = agent._synthesize_analysis(
        bayesian, rag, impact, _alert(),
        memory_results=[], log_evidence=log_evidence,
    )
    _, confidence_without_log, _ = agent._synthesize_analysis(
        bayesian, rag, impact, _alert(),
        memory_results=[], log_evidence=_empty_log_evidence("none"),
    )

    assert confidence_with_log > confidence_without_log
    assert evidence_with["log_evidence"]["corroborates_bayesian"] is True
    assert evidence_with["log_evidence"]["diverges_from_bayesian"] is False


@pytest.mark.asyncio
async def test_synthesize_divergence_reduces_effective_boost():
    """日志指向与贝叶斯不同的根因时，有效加成应小于一致场景。"""
    agent = RCAAgent()
    from app.agents.rca_agent import BayesianNode, ServiceImpact

    bayesian = [
        BayesianNode(name="resource_exhaustion", prior=0.2, likelihood=0.9,
                     posterior=0.5, evidence_strength=0.8),
    ]
    rag: list[dict[str, Any]] = []
    impact = [ServiceImpact(service="order-service", hop_distance=0, tier="critical")]

    log_diverging = {
        "available": True,
        "entries_count": 5,
        "cause_hits": {"code_bug": 5},
        "boosted_cause": "code_bug",
        "confidence_boost": 0.10,
        "samples": {"code_bug": ["NullPointerException at line 42"]},
        "source": "loki",
    }

    _, _, evidence = agent._synthesize_analysis(
        bayesian, rag, impact, _alert(),
        memory_results=[], log_evidence=log_diverging,
    )

    assert evidence["log_evidence"]["diverges_from_bayesian"] is True
    assert evidence["log_evidence"]["corroborates_bayesian"] is False
    # 分歧时 effective_boost = 0.10 * 0.3 = 0.03
    assert evidence["log_evidence"]["effective_boost"] == pytest.approx(0.03, abs=1e-4)


@pytest.mark.asyncio
async def test_synthesize_unavailable_logs_dont_change_confidence():
    """日志不可用时不应影响原有置信度。"""
    agent = RCAAgent()
    from app.agents.rca_agent import BayesianNode, ServiceImpact

    bayesian = [
        BayesianNode(name="resource_exhaustion", prior=0.2, likelihood=0.9,
                     posterior=0.5, evidence_strength=0.8),
    ]
    rag: list[dict[str, Any]] = []
    impact = [ServiceImpact(service="order-service", hop_distance=0, tier="critical")]

    _, conf_unavail, _ = agent._synthesize_analysis(
        bayesian, rag, impact, _alert(),
        memory_results=[], log_evidence=_empty_log_evidence("loki_error"),
    )
    _, conf_default, _ = agent._synthesize_analysis(
        bayesian, rag, impact, _alert(),
        memory_results=[], log_evidence=None,
    )

    assert conf_unavail == conf_default


# ============================ End-to-end process() ============================


@pytest.mark.asyncio
async def test_process_includes_log_evidence_in_rca_event():
    """process() 的最终 RCAEvent.evidence 应包含 log_evidence 字段。"""
    agent = RCAAgent()
    log_evidence_payload = {
        "available": True,
        "entries_count": 3,
        "cause_hits": {"network_issue": 3},
        "boosted_cause": "network_issue",
        "confidence_boost": 0.06,
        "samples": {"network_issue": ["Connection timeout"]},
        "source": "loki",
    }

    with patch.object(
        agent, "_query_log_evidence", AsyncMock(return_value=log_evidence_payload)
    ):
        result = await agent.process(
            RCAInput(alert=_alert(metric="error_rate_percent"), incident_id="INC-test"),
            context=None,
        )

    assert result.success is True
    rca_evidence = result.output_data["rca_event"]["evidence"]
    assert "log_evidence" in rca_evidence
    assert rca_evidence["log_evidence"]["entries_count"] == 3
    assert rca_evidence["log_evidence"]["boosted_cause"] == "network_issue"


@pytest.mark.asyncio
async def test_process_resilient_when_loki_unavailable():
    """Loki 不可达时 process() 仍应成功完成。"""
    agent = RCAAgent()
    with patch.object(
        agent, "_query_log_evidence",
        AsyncMock(return_value=_empty_log_evidence("loki_error")),
    ):
        result = await agent.process(
            RCAInput(alert=_alert(), incident_id="INC-test"),
            context=None,
        )

    assert result.success is True
    rca_evidence = result.output_data["rca_event"]["evidence"]
    assert rca_evidence["log_evidence"]["source"] == "unavailable"
    assert rca_evidence["log_evidence"]["entries_count"] == 0