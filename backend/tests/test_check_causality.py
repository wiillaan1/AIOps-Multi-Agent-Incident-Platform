"""
测试 _check_causality() 因果校验（审查报告四、V1+V2）

覆盖：
1. 时间相关性：告警在变更窗口内 → time_correlation=1.0；否则 0.0
2. 健康实例对照：差异 > 30% → healthy_baseline_diff > 0.30
3. 反事实前置打分：综合时间 + 基线差
4. 综合判定 is_causal / confidence
5. 缺数据时不抛异常，给出推荐
6. 与 _find_recent_changes 联动（order-service / payment-service 模拟数据）
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from app.agents.rca_agent import AlertEvent, RCAAgent


@pytest.fixture
def agent():
    return RCAAgent()


def _make_alert(timestamp: datetime | None = None) -> AlertEvent:
    """构造一个最小 AlertEvent（mock 避免 Pydantic 字段校验失败）"""
    alert = MagicMock(spec=AlertEvent)
    alert.timestamp = timestamp or datetime.now(timezone.utc)
    alert.service = "order-service"
    return alert


# ============================================================
# 1. 时间相关性
# ============================================================

class TestTimeCorrelation:
    def test_alert_within_change_window_for_order_service(self, agent):
        # _find_recent_changes 里 order-service 的变更时间是 2024-01-15T10:30:00Z
        alert_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        alert = _make_alert(alert_time)
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
        ))
        assert "time_correlation" in result["checks_performed"]
        assert result["signals"]["time_correlation"] == 1.0
        assert result["is_causal"] is True
        # 至少有一条 evidence_ref 指向变更
        assert any("change:" in ref for ref in result["evidence_refs"])

    def test_alert_outside_change_window(self, agent):
        # 告警在变更前 30 天
        alert_time = datetime(2024, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        alert = _make_alert(alert_time)
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
        ))
        assert result["signals"]["time_correlation"] == 0.0

    def test_unknown_service_returns_zero_correlation(self, agent):
        alert = _make_alert()
        result = asyncio.run(agent._check_causality(
            service="unknown-service",
            alert=alert,
        ))
        assert result["signals"]["time_correlation"] == 0.0

    def test_no_alert_skips_time_correlation(self, agent):
        # 没传 alert → 跳过时间相关性，不报错
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=None,
        ))
        assert "time_correlation" not in result["checks_performed"]
        assert result["signals"]["time_correlation"] == 0.0


# ============================================================
# 2. 健康实例对照
# ============================================================

class TestHealthyBaselineComparison:
    def test_significant_diff_triggers_recommendation(self, agent):
        # baseline cpu=30, current cpu=80 → 相对差异 (80-30)/30 = 1.67 → 截断到 1.0
        result = asyncio.run(agent._check_causality(
            service="order-service",
            baseline_metrics={"cpu": 30, "latency": 50},
            current_metrics={"cpu": 80, "latency": 50},
        ))
        assert "healthy_baseline_comparison" in result["checks_performed"]
        # 平均：(80-30)/30=1.0, (50-50)/50=0.0 → 0.5
        assert 0.4 < result["signals"]["healthy_baseline_diff"] < 0.6

    def test_minimal_diff_returns_low_score(self, agent):
        # baseline ≈ current → 差异 < 30%
        result = asyncio.run(agent._check_causality(
            service="order-service",
            baseline_metrics={"cpu": 50, "latency": 50},
            current_metrics={"cpu": 52, "latency": 51},
        ))
        # 差异都很小
        assert result["signals"]["healthy_baseline_diff"] < 0.1

    def test_no_overlap_returns_zero(self, agent):
        result = asyncio.run(agent._check_causality(
            service="order-service",
            baseline_metrics={"cpu": 30},
            current_metrics={"memory": 90},  # 没有共同 key
        ))
        # 没有共同 key → 不会触发 healthy_baseline_comparison
        assert "healthy_baseline_comparison" not in result["checks_performed"]
        assert result["signals"]["healthy_baseline_diff"] == 0.0

    def test_baseline_zero_with_nonzero_current(self, agent):
        # baseline=0, current=10 → 完全偏离 → 1.0
        result = asyncio.run(agent._check_causality(
            service="order-service",
            baseline_metrics={"error_rate": 0},
            current_metrics={"error_rate": 10},
        ))
        assert result["signals"]["healthy_baseline_diff"] == 1.0

    def test_baseline_zero_zero_current(self, agent):
        # baseline=0, current=0 → 0 差异
        result = asyncio.run(agent._check_causality(
            service="order-service",
            baseline_metrics={"error_rate": 0},
            current_metrics={"error_rate": 0},
        ))
        assert result["signals"]["healthy_baseline_diff"] == 0.0


# ============================================================
# 3. 反事实前置打分 + 综合判定
# ============================================================

class TestCounterfactualAndDecision:
    def test_high_signal_passes_causality(self, agent):
        # 时间相关性 = 1.0 + 健康差异 = 0.5 → counterfactual = 0.4*1 + 0.6*0.5 = 0.7
        alert_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        alert = _make_alert(alert_time)
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
            baseline_metrics={"cpu": 30},
            current_metrics={"cpu": 80},
        ))
        assert result["signals"]["counterfactual_score"] >= 0.5
        assert result["is_causal"] is True
        assert result["confidence"] >= 0.5

    def test_low_signal_fails_causality(self, agent):
        # 无变更窗口 + 健康实例无差异 → counterfactual < 0.5
        result = asyncio.run(agent._check_causality(
            service="unknown-service",
            alert=None,
            baseline_metrics={"cpu": 50},
            current_metrics={"cpu": 50},
        ))
        assert result["is_causal"] is False
        assert "因果证据不足" in result["recommendations"][-1]

    def test_evidence_refs_populated(self, agent):
        alert_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        alert = _make_alert(alert_time)
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
            baseline_metrics={"cpu": 30},
            current_metrics={"cpu": 80},
        ))
        assert len(result["evidence_refs"]) >= 2  # change + baseline_diff

    def test_recommendations_included(self, agent):
        alert_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        alert = _make_alert(alert_time)
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
            baseline_metrics={"cpu": 30},
            current_metrics={"cpu": 80},
        ))
        assert len(result["recommendations"]) > 0


# ============================================================
# 4. 边界场景
# ============================================================

class TestEdgeCases:
    def test_no_inputs_returns_low_confidence(self, agent):
        result = asyncio.run(agent._check_causality(
            service="unknown-service",
            alert=None,
        ))
        assert result["is_causal"] is False
        assert result["confidence"] == 0.0
        assert result["checks_performed"] == []

    def test_custom_change_window(self, agent):
        # 默认 24h 窗口外，2h 窗口内 → time_correlation=1.0
        alert_time = datetime(2024, 1, 15, 11, 0, 0, tzinfo=timezone.utc)  # 变更后 30 分钟
        alert = _make_alert(alert_time)
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
            change_window_hours=2,
        ))
        assert result["signals"]["time_correlation"] == 1.0

    def test_string_timestamp_parsed(self, agent):
        # alert.timestamp 是 ISO 字符串
        alert = MagicMock()
        alert.timestamp = "2024-01-15T11:00:00+00:00"
        alert.service = "order-service"
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
        ))
        assert result["signals"]["time_correlation"] == 1.0

    def test_invalid_string_timestamp_returns_zero(self, agent):
        # 损坏的字符串 → 不抛异常，time_correlation=0
        alert = MagicMock()
        alert.timestamp = "not a date"
        alert.service = "order-service"
        result = asyncio.run(agent._check_causality(
            service="order-service",
            alert=alert,
        ))
        assert result["signals"]["time_correlation"] == 0.0

    def test_returns_valid_structure(self, agent):
        result = asyncio.run(agent._check_causality(service="order-service"))
        assert "is_causal" in result
        assert "confidence" in result
        assert "signals" in result
        assert "evidence_refs" in result
        assert "recommendations" in result
        assert "checks_performed" in result
        # confidence ∈ [0, 1]
        assert 0.0 <= result["confidence"] <= 1.0
        # 三层信号都在
        assert "time_correlation" in result["signals"]
        assert "healthy_baseline_diff" in result["signals"]
        assert "counterfactual_score" in result["signals"]
