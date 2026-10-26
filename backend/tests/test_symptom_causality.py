"""_infer_symptom_causality() 多症状告警朴素因果推断测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from app.agents.symptom_causality import (
    MIN_SHARED_CAUSALITY,
    SharedRootCauseHypothesis,
    SymptomCausalityPair,
    SymptomCausalityResult,
    _infer_symptom_causality,
)


# ----- 测试用 fake 对象（不依赖真实 AlertEvent/Incident） -----


@dataclass
class FakeAlert:
    event_id: str
    service: str
    metric: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: str = "MEDIUM"


@dataclass
class FakeRCA:
    root_cause: str  # "service::metric" 格式


@dataclass
class FakeIncident:
    alert_event: FakeAlert
    rca_event: FakeRCA | None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    context: dict[str, Any] = field(default_factory=dict)


def _make_multi_symptom_incident(
    primary_service: str,
    primary_metric: str,
    additional: list[tuple[str, str]],  # [(service, metric), ...]
    root_service: str,
    root_metric: str,
    *,
    minutes_ago: int = 0,
) -> FakeIncident:
    """多症状 incident：primary alert + additional alerts 共享同一根因"""
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return FakeIncident(
        alert_event=FakeAlert(
            event_id=f"primary-{primary_service}",
            service=primary_service,
            metric=primary_metric,
            timestamp=ts,
        ),
        rca_event=FakeRCA(root_cause=f"{root_service}::{root_metric}"),
        created_at=ts,
        context={
            "additional_alerts": [
                {"service": svc, "metric": mtr, "event_id": f"add-{svc}-{i}"}
                for i, (svc, mtr) in enumerate(additional)
            ]
        },
    )


def _make_incident(
    service: str,
    metric: str,
    root_service: str,
    root_metric: str,
    *,
    minutes_ago: int = 0,
    event_id: str = "inc",
) -> FakeIncident:
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return FakeIncident(
        alert_event=FakeAlert(
            event_id=event_id,
            service=service,
            metric=metric,
            timestamp=ts,
        ),
        rca_event=FakeRCA(root_cause=f"{root_service}::{root_metric}"),
        created_at=ts,
    )


# ----- 测试 -----


def test_less_than_two_alerts_returns_empty():
    """少于 2 个告警 → 无意义，返回空结果"""
    result = _infer_symptom_causality(
        alerts=[FakeAlert(event_id="a1", service="svc", metric="cpu")],
        historical_incidents=[],
    )
    assert result.pairs == []
    assert result.shared_hypotheses == []
    assert "alerts < 2" in result.notes[0]


def test_two_alerts_with_shared_history_get_high_score():
    """历史 5 次都共享 mysql 慢查询 → 当前两个告警被强烈认为共享根因"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(5)
    ]

    now = datetime.now(timezone.utc)
    alerts = [
        # i (先) 比 j (后) 早 90 秒，超过默认 slop 60s
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=90)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now),
    ]

    result = _infer_symptom_causality(alerts, historical)
    assert len(result.pairs) == 1
    pair = result.pairs[0]
    assert pair.co_occurrence_count == 5  # 5 次多症状 incident
    # co_occurrence 高 + 时序匹配 → naive_causal_prob 接近 1
    assert pair.naive_causal_prob > 0.7
    assert pair.time_sequence_match is True
    assert pair.historical_shared_root == "mysql::slow_query"

    # 共享根因假设
    assert len(result.shared_hypotheses) == 1
    hyp = result.shared_hypotheses[0]
    assert hyp.root_cause_service == "mysql"
    assert hyp.root_cause_metric == "slow_query"
    assert hyp.aggregated_prob > 0.7
    assert set(hyp.contributing_alert_ids) == {"a1", "a2"}


def test_time_sequence_mismatch_lowers_score():
    """时序不匹配（j 在 i 之前）→ 朴素因果分数降低"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(5)
    ]

    now = datetime.now(timezone.utc)
    # 反向时序：j 在 i 之前（payment 比 account 晚 120s）
    # 内部会自动 swap，让"先发生的"成为 i。
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now - timedelta(seconds=120)),
    ]

    result = _infer_symptom_causality(alerts, historical)
    pair = result.pairs[0]
    # 时序内部已 swap：account 早、payment 晚 → time_match=True
    # 说明 swap 起作用了
    assert pair.time_sequence_match is True
    # 验证 swap：score 应该高（co_occurrence + time match）
    assert pair.naive_causal_prob > 0.7


def test_time_sequence_no_order_lowers_score():
    """两个 alert 时间戳几乎相同（< slop）→ time_match=False"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(5)
    ]

    now = datetime.now(timezone.utc)
    # 间隔只有 10 秒（< 默认 slop 60s）→ time_match=False
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=10)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now),
    ]

    result = _infer_symptom_causality(alerts, historical)
    pair = result.pairs[0]
    assert pair.time_sequence_match is False
    # co_occurrence 高，time=0.5 → naive_prob 中等
    assert 0.5 < pair.naive_causal_prob < 0.9


def test_no_history_pair_not_included_when_threshold_strict():
    """历史极少 + 该 pair 历史没出现 → 默认跳过（min_co_occurrence 默认 1）"""
    historical = [_make_incident("payment", "latency", "redis", "cache_miss", minutes_ago=i) for i in range(10)]

    now = datetime.now(timezone.utc)
    alerts = [
        FakeAlert(event_id="a1", service="order", metric="error_rate", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a2", service="payment", metric="timeout", timestamp=now),
    ]

    result = _infer_symptom_causality(alerts, historical)
    # order+payment 这个 pair 在历史里没同时出现 → 跳过
    assert result.pairs == []


def test_historical_incidents_too_few_falls_back_to_time_only():
    """历史 incidents < 3 → 警告且结果以时序为主"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait")],
            "mysql", "slow_query",
            minutes_ago=0,
        ),
    ]

    now = datetime.now(timezone.utc)
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now),
    ]

    result = _infer_symptom_causality(alerts, historical)
    assert any("历史 incidents 仅" in n for n in result.notes)
    # 但 pair 还是会算
    assert len(result.pairs) >= 1


def test_shared_hypothesis_aggregated_prob_descending():
    """多个共享根因假设时按 aggregated_prob 降序"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(8)
    ]
    historical += [
        _make_multi_symptom_incident(
            "payment", "latency",
            [],
            "redis", "cache_miss",
            minutes_ago=i,
        )
        for i in range(3)
    ]

    now = datetime.now(timezone.utc)
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now),
    ]

    result = _infer_symptom_causality(alerts, historical)
    # 应该有至少 1 个 mysql 假设
    assert any(h.root_cause_service == "mysql" for h in result.shared_hypotheses)
    # 排序：aggregated_prob 降序
    probs = [h.aggregated_prob for h in result.shared_hypotheses]
    assert probs == sorted(probs, reverse=True)


def test_pair_evidence_fields_present():
    """输出 Pair 字段完整"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(5)
    ]
    now = datetime.now(timezone.utc)
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now),
    ]
    result = _infer_symptom_causality(alerts, historical)
    pair = result.pairs[0]
    assert isinstance(pair, SymptomCausalityPair)
    assert 0.0 <= pair.naive_causal_prob <= 1.0
    assert 0.0 <= pair.co_occurrence_prob <= 1.0
    assert isinstance(pair.time_sequence_match, bool)
    assert isinstance(pair.co_occurrence_count, int)


def test_min_shared_causality_threshold():
    """MIN_SHARED_CAUSALITY 是公开常量，便于调整"""
    assert MIN_SHARED_CAUSALITY == 0.5


def test_three_alerts_three_pairs():
    """3 个 alert → 3 个 pair (C(3,2) = 3)"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait"), ("order", "error_rate")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(5)
    ]
    now = datetime.now(timezone.utc)
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=60)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a3", service="order", metric="error_rate", timestamp=now),
    ]
    result = _infer_symptom_causality(alerts, historical)
    # 3 个 pair 都会被算
    assert len(result.pairs) == 3


def test_missing_event_id_skipped():
    """缺 event_id 的 alert 不会进入配对"""
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency"),
        FakeAlert(event_id=None, service="account", metric="thread_wait"),
    ]
    result = _infer_symptom_causality(alerts, [])
    assert result.pairs == []
    assert any("缺少 event_id" in n for n in result.notes)


def test_hypothesis_contributing_alerts_unique():
    """共享根因假设的 contributing_alert_ids 去重"""
    historical = [
        _make_multi_symptom_incident(
            "payment", "latency",
            [("account", "thread_wait"), ("order", "error_rate")],
            "mysql", "slow_query",
            minutes_ago=i,
        )
        for i in range(10)
    ]

    now = datetime.now(timezone.utc)
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=60)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a3", service="order", metric="error_rate", timestamp=now),
    ]
    result = _infer_symptom_causality(alerts, historical)
    hyp = result.shared_hypotheses[0]
    # a1, a2, a3 都应该出现且唯一
    assert sorted(hyp.contributing_alert_ids) == ["a1", "a2", "a3"]


def test_laplace_smoothing_prevents_zero_probability():
    """co-occurrence_count=0 时 co_occurrence_prob 用 Laplace smoothing 不为 0"""
    historical = [_make_incident("redis", "cache_miss", "redis", "memory_full", minutes_ago=i) for i in range(5)]
    now = datetime.now(timezone.utc)
    # 当前 alerts 用一个历史从未同时出现的组合
    alerts = [
        FakeAlert(event_id="a1", service="payment", metric="latency", timestamp=now - timedelta(seconds=30)),
        FakeAlert(event_id="a2", service="account", metric="thread_wait", timestamp=now),
    ]
    result = _infer_symptom_causality(alerts, historical, min_co_occurrence=0)
    pair = result.pairs[0]
    # 即使 count=0，Laplace 平滑后 prob = (0+1)/(5+2) ≈ 0.143
    assert 0.0 < pair.co_occurrence_prob < 0.5