"""
Tests for Monitor adaptive threshold + context reflection (W3b).
"""

from __future__ import annotations

from app.agents.monitor_agent import MonitorAgent
from app.models.events import AlertEvent  # 复用 MetricInput 风格


def _run_with_votes(monitor: MonitorAgent, value: float, history: list[float]):
    """直接调用 detect_anomaly 但不传 MetricInput：走更轻量路径。"""
    # 构造 fake MetricInput-like 对象（dataclass duck-typing）
    from types import SimpleNamespace

    return monitor.detect_anomaly(SimpleNamespace(
        metric_name="cpu_usage_percent",
        service_name="order-service",
        metric_value=value,
        history_values=history,
        labels={"tier": "critical"},
    ))


def test_get_historical_weights_returns_none_when_no_data():
    monitor = MonitorAgent()
    assert monitor._get_historical_weights("svc:metric") is None


def test_get_historical_weights_returns_none_when_too_few_samples():
    monitor = MonitorAgent()
    monitor._algo_accuracy = {}
    monitor._algo_accuracy["svc:m"] = {"3-Sigma": 3}
    assert monitor._get_historical_weights("svc:m") is None


def test_get_historical_weights_normalizes_to_unit_total():
    monitor = MonitorAgent()
    monitor._algo_accuracy = {"svc:m": {"3-Sigma": 8, "EWMA": 5, "IsolationForest": 7}}
    weights = monitor._get_historical_weights("svc:m")
    assert weights is not None
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["3-Sigma"] > weights["EWMA"]


def test_compute_adaptive_score_returns_none_without_weights():
    monitor = MonitorAgent()
    # fake votes
    from types import SimpleNamespace
    votes = [SimpleNamespace(algorithm_name="3-Sigma", score=0.8)]
    assert monitor._compute_adaptive_score(votes, None) is None


def test_compute_adaptive_score_weights_by_historical_accuracy():
    monitor = MonitorAgent()
    weights = {"3-Sigma": 0.9, "EWMA": 0.1}
    from types import SimpleNamespace
    votes = [
        SimpleNamespace(algorithm_name="3-Sigma", score=0.8),
        SimpleNamespace(algorithm_name="EWMA", score=0.3),
    ]
    score = monitor._compute_adaptive_score(votes, weights)
    # 0.8*0.9 + 0.3*0.1 = 0.72 + 0.03 = 0.75
    assert abs(score - 0.75) < 1e-9


def test_record_algorithm_feedback_enables_adaptive():
    monitor = MonitorAgent()
    key = "order:cpu"
    for _ in range(15):
        monitor.record_algorithm_feedback(key, "3-Sigma", True)
        monitor.record_algorithm_feedback(key, "EWMA", False)

    weights = monitor._get_historical_weights(key)
    assert weights is not None
    assert weights["3-Sigma"] == 1.0  # 全部 True


def test_reflect_with_context_returns_none_for_short_history():
    monitor = MonitorAgent()
    from types import SimpleNamespace
    result = monitor._reflect_with_context(
        SimpleNamespace(metric_value=100.0),
        [],
        history=[1.0, 2.0],
    )
    assert result is None


def test_reflect_with_context_lowers_confidence_when_within_trend():
    """当前值只是近期均值 1.3 倍 → 视为正常波动 → 降低置信度。"""
    monitor = MonitorAgent()
    from types import SimpleNamespace
    result = monitor._reflect_with_context(
        SimpleNamespace(metric_value=130.0),
        [],
        history=[90.0, 95.0, 100.0, 105.0, 100.0],  # 近期均值 98
    )
    assert result is not None
    assert result < 0.6  # 应当降为非异常


def test_reflect_with_context_raises_confidence_for_spike():
    """当前值远高于近期均值 → 真实异常 → 提升置信度。"""
    monitor = MonitorAgent()
    from types import SimpleNamespace
    result = monitor._reflect_with_context(
        SimpleNamespace(metric_value=300.0),  # 3 倍于近期均值
        [],
        history=[90.0, 95.0, 100.0, 105.0, 100.0],
    )
    assert result is not None
    assert result > 0.6


def test_reflect_with_context_caps_confidence_at_0_85():
    monitor = MonitorAgent()
    from types import SimpleNamespace
    result = monitor._reflect_with_context(
        SimpleNamespace(metric_value=1_000_000.0),
        [],
        history=[90.0, 95.0, 100.0, 105.0, 100.0],
    )
    assert result is not None
    assert result <= 0.85
