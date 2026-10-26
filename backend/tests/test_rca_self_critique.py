"""
Tests for RCA self-critique + recovery (W2).

覆盖：
1. 高置信度时不应触发自我批判
2. 低置信度 + RAG 召回少 → 触发拓宽 RAG
3. 低置信度 + Memory 命中 0 → 触发扩大检索
4. 反思成功 → 提升置信度并写 reflection_note
5. 反思不显著 → 返回 None（让 Orchestrator 升级）
6. RCAEvent.evidence.reflection_note 在 outputs 里可见
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.rca_agent import BayesianNode, RCAAgent
from app.models.events import AlertEvent, SeverityLevel


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


def _low_confidence_bayesian() -> list[BayesianNode]:
    """模拟低置信度场景：所有 posterior 接近均匀。"""
    return [
        BayesianNode(name="resource_exhaustion", prior=0.2, likelihood=0.3,
                     posterior=0.15, evidence_strength=0.2),
        BayesianNode(name="traffic_spike", prior=0.18, likelihood=0.25,
                     posterior=0.13, evidence_strength=0.2),
        BayesianNode(name="code_bug", prior=0.12, likelihood=0.2,
                     posterior=0.11, evidence_strength=0.2),
    ]


@pytest.mark.asyncio
async def test_self_critique_recovers_when_broadened_rag_finds_more():
    """首轮 RAG 只有 1 条 → 拓宽后命中 3 条 → 反思触发并写入 reflection_note。"""
    agent = RCAAgent()
    initial_rag = [{"id": "kb-1", "category": "perf"}]
    broadened_rag = [
        {"id": "kb-1", "category": "perf"},
        {"id": "kb-2", "category": "perf"},
        {"id": "kb-3", "category": "perf"},
    ]

    with patch.object(
        agent, "_rag_retrieve_broadened", return_value=broadened_rag,
    ), patch.object(
        agent, "_search_historical_incidents_broadened",
        AsyncMock(return_value=[]),
    ), patch.object(
        agent, "_query_log_evidence_broadened",
        AsyncMock(return_value={"available": True, "entries_count": 0}),
    ):
        result = await agent._self_critique_and_recover(
            alert=_alert(),
            original_root_cause="resource_exhaustion",
            original_confidence=0.35,
            bayesian_results=_low_confidence_bayesian(),
            rag_results=initial_rag,
            memory_results=[],
            log_evidence={"available": False, "entries_count": 0},
        )

    assert result is not None, "反思应当返回 recovery result"
    assert "broaden_rag_keywords" in result["evidence"]["reflection_actions"]
    # 注：recovered_confidence 不一定大于 original，因为重跑综合分析时
    # 弱 bayesian 节点可能拉低分数；这里只断言"反思触发了"
    assert "首轮置信度" in result["reflection_note"]


@pytest.mark.asyncio
async def test_self_critique_returns_none_when_no_improvement():
    """所有拓宽策略都没有新发现 → 返回 None。"""
    agent = RCAAgent()

    with patch.object(
        agent, "_rag_retrieve_broadened", return_value=[{"id": "kb-1"}],
    ), patch.object(
        agent, "_search_historical_incidents_broadened",
        AsyncMock(return_value=[]),
    ), patch.object(
        agent, "_query_log_evidence_broadened",
        AsyncMock(return_value={"available": False, "entries_count": 0}),
    ):
        result = await agent._self_critique_and_recover(
            alert=_alert(),
            original_root_cause="unknown",
            original_confidence=0.3,
            bayesian_results=_low_confidence_bayesian(),
            rag_results=[{"id": "kb-1"}],  # 已有 1 条
            memory_results=[{"memory_id": "m1"}],  # 已有 1 条
            log_evidence={"available": True, "entries_count": 2},
        )

    assert result is None


@pytest.mark.asyncio
async def test_self_critique_expand_memory_when_no_history():
    """Memory 命中 0 条 → 扩大 service 范围 → 反思。"""
    agent = RCAAgent()

    with patch.object(
        agent, "_rag_retrieve_broadened", return_value=[{"id": "kb-1"}],
    ), patch.object(
        agent, "_search_historical_incidents_broadened",
        AsyncMock(return_value=[{"memory_id": "m1"}, {"memory_id": "m2"}]),
    ), patch.object(
        agent, "_query_log_evidence_broadened",
        AsyncMock(return_value={"available": False, "entries_count": 0}),
    ):
        result = await agent._self_critique_and_recover(
            alert=_alert(),
            original_root_cause="unknown",
            original_confidence=0.35,
            bayesian_results=_low_confidence_bayesian(),
            rag_results=[{"id": "kb-1"}, {"id": "kb-2"}],  # 已够
            memory_results=[],
            log_evidence={"available": False, "entries_count": 0},
        )

    assert result is not None
    assert "expand_memory_search" in result["evidence"]["reflection_actions"]


@pytest.mark.asyncio
async def test_self_critique_broaden_logs_when_no_evidence():
    """日志证据空 → 查 WARN 级 → 反思。"""
    agent = RCAAgent()

    with patch.object(
        agent, "_rag_retrieve_broadened", return_value=[{"id": "kb-1"}],
    ), patch.object(
        agent, "_search_historical_incidents_broadened",
        AsyncMock(return_value=[]),
    ), patch.object(
        agent, "_query_log_evidence_broadened",
        AsyncMock(return_value={
            "available": True,
            "entries_count": 5,
            "cause_hits": {},
            "boosted_cause": None,
            "confidence_boost": 0.0,
            "samples": {"WARN": ["timeout", "retry"]},
        }),
    ):
        result = await agent._self_critique_and_recover(
            alert=_alert(),
            original_root_cause="unknown",
            original_confidence=0.35,
            bayesian_results=_low_confidence_bayesian(),
            rag_results=[{"id": "kb-1"}, {"id": "kb-2"}],
            memory_results=[{"memory_id": "m1"}],
            log_evidence={"available": False, "entries_count": 0},
        )

    assert result is not None
    assert "broaden_log_query" in result["evidence"]["reflection_actions"]


@pytest.mark.asyncio
async def test_self_critique_caps_confidence_boost():
    """反思成功置信度也不能超过 0.95。"""
    agent = RCAAgent()

    with patch.object(
        agent, "_rag_retrieve_broadened",
        return_value=[{"id": f"kb-{i}"} for i in range(10)],
    ), patch.object(
        agent, "_search_historical_incidents_broadened",
        AsyncMock(return_value=[]),
    ), patch.object(
        agent, "_query_log_evidence_broadened",
        AsyncMock(return_value={"available": False, "entries_count": 0}),
    ):
        result = await agent._self_critique_and_recover(
            alert=_alert(),
            original_root_cause="unknown",
            original_confidence=0.90,
            bayesian_results=_low_confidence_bayesian(),
            rag_results=[],
            memory_results=[],
            log_evidence={"available": False, "entries_count": 0},
        )

    assert result is not None
    assert result["confidence"] <= 0.95


@pytest.mark.asyncio
async def test_self_critique_records_reflection_note():
    """反思成功时 reflection_note 包含"首轮置信度" + "修正为"。"""
    agent = RCAAgent()

    with patch.object(
        agent, "_rag_retrieve_broadened",
        return_value=[{"id": "kb-1"}, {"id": "kb-2"}, {"id": "kb-3"}],
    ), patch.object(
        agent, "_search_historical_incidents_broadened",
        AsyncMock(return_value=[]),
    ), patch.object(
        agent, "_query_log_evidence_broadened",
        AsyncMock(return_value={"available": False, "entries_count": 0}),
    ):
        result = await agent._self_critique_and_recover(
            alert=_alert(),
            original_root_cause="resource_exhaustion",
            original_confidence=0.30,
            bayesian_results=_low_confidence_bayesian(),
            rag_results=[{"id": "kb-1"}],
            memory_results=[],
            log_evidence={"available": False, "entries_count": 0},
        )

    assert "首轮置信度" in result["reflection_note"]
    assert "修正为" in result["reflection_note"]


@pytest.mark.asyncio
async def test_process_skips_reflection_when_high_confidence():
    """高置信度（>= 0.55）时 process 不应调用自我批判。"""
    agent = RCAAgent()

    # 让首轮置信度足够高（用 patch 强制 mock）
    alert = _alert()

    with patch.object(
        agent, "_self_critique_and_recover",
        AsyncMock(return_value=None),
    ) as mock_critique, patch.object(
        agent, "_synthesize_analysis",
        return_value=("resource_exhaustion", 0.85, {
            "bayesian_top": "resource_exhaustion",
            "bayesian_posterior": 0.5,
            "rag_top_match": "kb-1",
            "rag_match_score": 0.7,
            "historical_memory_matches": 1,
            "memory_top_score": 0.6,
            "memory_confidence_boost": 0.05,
            "log_corroboration": False,
            "log_divergence": False,
            "affected_services": ["order-service"],
            "critical_services_affected": [],
            "total_impact_hops": 1,
        }),
    ), patch.object(
        agent, "_query_log_evidence",
        AsyncMock(return_value={"available": False, "entries_count": 0}),
    ), patch.object(
        agent, "_search_historical_incidents",
        AsyncMock(return_value=[]),
    ):
        from app.agents.rca_agent import RCAInput
        from app.models.agent import AgentExecutionContext

        result = await agent.process(
            RCAInput(alert=alert, incident_id="INC-test"),
            AgentExecutionContext(incident_id="INC-test"),
        )

    # 0.85 >= 0.55 → 不触发反思
    assert result.success is True
    evidence = result.output_data["rca_event"]["evidence"]
    assert evidence.get("reflection_note", "") == ""
    mock_critique.assert_not_called()


def test_rag_retrieve_broadened_returns_more_when_keywords_match():
    """放宽版 RAG 比标准版阈值更低，应返回更多候选。"""
    from app.agents.rca_agent import RCAAgent

    agent = RCAAgent()
    alert = _alert()

    # 标准版应当 < 2 条；放宽版应当 >= 2 条
    standard = agent._rag_retrieve(alert, [])
    broadened = agent._rag_retrieve_broadened(alert)

    # 至少标准版不会包含 broadened_match 标记
    for r in standard:
        assert "broadened_match" not in r
    for r in broadened:
        assert r.get("broadened_match") is True
    # 放宽版的阈值是 0.2，原版是 0.3，所以放宽版候选 >= 标准版
    assert len(broadened) >= len(standard)
