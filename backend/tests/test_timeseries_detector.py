"""
TimeSeriesDetector 测试

覆盖：
1. Z-Score（数据足/不足/无波动）
2. IQR（数据足/不足/单边异常）
3. 趋势（线性回归）
4. 静态阈值（above/below）
5. 综合判定（多算法触发 + severity）
6. 真实场景测试
"""
from __future__ import annotations

import pytest

from app.detection import TimeSeriesDetector, DetectionResult, Severity, AlgorithmTrigger


# =====================================================================
# Z-Score 测试
# =====================================================================

class TestZScore:
    def test_normal_value_not_triggered(self):
        det = TimeSeriesDetector()
        trigger = det._z_score_check(current=82.0, history=[80, 81, 82, 83, 80, 81])
        assert trigger.triggered is False
        assert trigger.algorithm == "z_score"

    def test_spike_triggered(self):
        det = TimeSeriesDetector(z_score_threshold=3.0)
        history = [80, 81, 82, 83, 80, 81, 82, 80, 81, 82]
        trigger = det._z_score_check(current=99.0, history=history)
        assert trigger.triggered is True
        assert "Z=" in trigger.description

    def test_insufficient_data(self):
        det = TimeSeriesDetector()
        trigger = det._z_score_check(current=99.0, history=[80, 81])
        assert trigger.triggered is False
        assert "数据不足" in trigger.description

    def test_zero_variance(self):
        det = TimeSeriesDetector()
        # current 偏离 50% → 触发；偏离 < 10% → 不触发
        trigger = det._z_score_check(current=85.0, history=[80, 80, 80, 80, 80])
        # 偏离 5/80 = 6.25% < 10%，不触发
        assert trigger.triggered is False
        assert "σ≈0" in trigger.description

    def test_zero_variance_large_deviation(self):
        det = TimeSeriesDetector()
        trigger = det._z_score_check(current=120.0, history=[80, 80, 80, 80, 80])
        # 偏离 40/80 = 50% > 10%，触发
        assert trigger.triggered is True


# =====================================================================
# IQR 测试
# =====================================================================

class TestIQR:
    def test_normal_value(self):
        det = TimeSeriesDetector()
        trigger = det._iqr_check(current=82.0, history=[80, 81, 82, 83, 84, 85, 86, 87])
        assert trigger.triggered is False

    def test_outlier_above(self):
        det = TimeSeriesDetector()
        history = [80, 81, 82, 83, 84, 85, 86, 87]  # Q1=81.25, Q3=85.75, IQR=4.5
        trigger = det._iqr_check(current=99.0, history=history)
        assert trigger.triggered is True

    def test_outlier_below(self):
        det = TimeSeriesDetector()
        history = [80, 81, 82, 83, 84, 85, 86, 87]
        trigger = det._iqr_check(current=70.0, history=history)
        assert trigger.triggered is True

    def test_insufficient_data(self):
        det = TimeSeriesDetector()
        trigger = det._iqr_check(current=99.0, history=[80, 81, 82])
        assert trigger.triggered is False


# =====================================================================
# 趋势测试
# =====================================================================

class TestTrend:
    def test_stable_trend_not_triggered(self):
        det = TimeSeriesDetector()
        # 完全恒定 → 斜率 = 0
        history = [80] * 10
        trigger = det._trend_check(current=80.0, history=history)
        assert trigger.triggered is False

    def test_trend_predicted_check(self):
        """趋势算法要求 current > predicted 才触发"""
        det = TimeSeriesDetector(trend_slope_threshold=0.1)
        # 上升斜率但 current 远低于预测 → 不应触发
        history = [80, 82, 84, 86, 88, 90]
        trigger = det._trend_check(current=70.0, history=history)
        assert trigger.triggered is False  # current < predicted

    def test_upward_trend(self):
        det = TimeSeriesDetector(trend_slope_threshold=0.1)
        history = [80, 82, 84, 86, 88, 90]  # 斜率 ≈ 2.0
        trigger = det._trend_check(current=95.0, history=history)
        assert trigger.triggered is True
        assert "slope=" in trigger.description

    def test_downward_trend_not_triggered(self):
        """下降趋势且 current < predicted → 不应触发"""
        det = TimeSeriesDetector(trend_slope_threshold=0.1)
        history = [80, 78, 76, 74, 72]  # 斜率 ≈ -2
        trigger = det._trend_check(current=70.0, history=history)
        assert trigger.triggered is False  # current < predicted

    def test_insufficient_data(self):
        det = TimeSeriesDetector()
        trigger = det._trend_check(current=99.0, history=[80, 81, 82])
        assert trigger.triggered is False


# =====================================================================
# 静态阈值测试
# =====================================================================

class TestStaticThreshold:
    def test_above_threshold_triggered(self):
        det = TimeSeriesDetector()
        trigger = det._static_check(current=95.0, threshold=80.0, direction="above")
        assert trigger.triggered is True

    def test_below_threshold_not_triggered_for_above(self):
        det = TimeSeriesDetector()
        trigger = det._static_check(current=70.0, threshold=80.0, direction="above")
        assert trigger.triggered is False

    def test_below_direction(self):
        det = TimeSeriesDetector()
        trigger = det._static_check(current=10.0, threshold=80.0, direction="below")
        assert trigger.triggered is True

    def test_score_proportional_to_deviation(self):
        det = TimeSeriesDetector()
        small = det._static_check(current=82.0, threshold=80.0, direction="above")
        large = det._static_check(current=120.0, threshold=80.0, direction="above")
        assert large.score > small.score


# =====================================================================
# 综合判定
# =====================================================================

class TestAggregate:
    def test_no_trigger_normal(self):
        det = TimeSeriesDetector()
        result = det.detect(
            metric_name="cpu_usage_percent",
            current=82.0,
            history_1h=[80, 81, 82, 83, 80],
            history_7d=[78, 79, 80, 81, 82, 83],
            history_24h=[80, 81, 82, 80, 81],
            static_threshold=90.0,
        )
        assert result.is_anomaly is False
        assert result.severity in (Severity.NORMAL.value, Severity.LOW.value)
        assert len(result.triggers) == 4  # 4 个算法都跑了

    def test_single_trigger_low(self):
        det = TimeSeriesDetector()
        result = det.detect(
            metric_name="cpu_usage_percent",
            current=88.0,  # 略超阈值
            history_1h=[80, 81, 82, 80, 81],
            history_7d=[80, 81, 82, 83, 84, 85, 86, 87],
            history_24h=[80, 81, 82, 81, 80],
            static_threshold=85.0,
        )
        assert result.is_anomaly is True
        # 静态阈值会触发，置信度应该中等

    def test_multi_trigger_higher_severity(self):
        """3 个算法同时触发 → severity 应升级"""
        det = TimeSeriesDetector()
        # 历史平稳，current 突然飙到 99（z_score + static 都触发）
        # IQR 历史也设到会让 99 是异常值
        result = det.detect(
            metric_name="cpu_usage_percent",
            current=99.0,
            history_1h=[80, 81, 82, 83, 80, 81, 82, 80, 81, 82],  # z_score 触发
            history_7d=[80, 81, 82, 83, 84, 85, 86, 87],  # IQR 触发
            history_24h=[80, 82, 84, 86, 88, 90],  # trend 触发
            static_threshold=80.0,
        )
        triggered = [t for t in result.triggers if t.triggered]
        assert len(triggered) >= 2
        # 多算法触发 → 严重度升级
        assert result.severity in (Severity.HIGH.value, Severity.CRITICAL.value)

    def test_summary_includes_trigger_count(self):
        det = TimeSeriesDetector()
        result = det.detect(
            metric_name="cpu",
            current=99.0,
            history_1h=[80, 81, 82],
            static_threshold=80.0,
        )
        if result.is_anomaly:
            assert "异常" in result.summary
            assert "/" in result.summary  # "2/4 算法触发"

    def test_recommended_window_priority(self):
        """z_score > trend > iqr > static"""
        det = TimeSeriesDetector()
        # 让 z_score 和 static 都触发
        result = det.detect(
            metric_name="cpu",
            current=99.0,
            history_1h=[80, 81, 82, 80, 81],  # z_score 会触发
            history_7d=[80, 81, 82, 83, 84, 85, 86, 87],  # IQR 不触发
            history_24h=[80, 82, 84, 86, 88],  # trend 会触发
            static_threshold=80.0,
        )
        # z_score 优先级最高 → 1h
        if any(t.algorithm == "z_score" and t.triggered for t in result.triggers):
            assert result.recommended_window == "1h"


# =====================================================================
# 真实场景
# =====================================================================

class TestRealScenarios:
    """真实运维场景测试"""

    def test_cpu_spike_after_deploy(self):
        """部署后 CPU 飙升至 95%"""
        det = TimeSeriesDetector()
        # 1h 内：前 50 分钟 80% 平稳，部署后 5 分钟飙到 95
        history_1h = [80] * 8 + [82, 85, 88, 92, 95]
        # 7d 同期：1 周内都是 75-85 范围
        history_7d = [78, 80, 82, 81, 79, 80, 82, 83, 84, 81, 80, 82, 83, 82]

        result = det.detect(
            metric_name="cpu_usage_percent",
            current=95.0,
            history_1h=history_1h,
            history_7d=history_7d,
            history_24h=history_1h + [80] * 4,  # 24h 内大部分平稳
            static_threshold=80.0,
        )
        assert result.is_anomaly is True
        assert result.severity in (Severity.HIGH.value, Severity.CRITICAL.value)

    def test_slow_disk_full(self):
        """磁盘缓慢填满（24h 趋势）"""
        det = TimeSeriesDetector(trend_slope_threshold=0.1)
        history_24h = [60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82]
        result = det.detect(
            metric_name="disk_usage_percent",
            current=85.0,
            history_24h=history_24h,
            history_1h=history_24h[-5:],
            history_7d=[60, 65, 70, 75, 80],
            static_threshold=90.0,  # 静态阈值还没超
        )
        # 趋势应该触发（虽然静态没超）
        assert result.is_anomaly is True

    def test_no_issue_normal_usage(self):
        """正常波动不算异常"""
        det = TimeSeriesDetector()
        result = det.detect(
            metric_name="latency_ms",
            current=120.0,
            history_1h=[100, 105, 110, 115, 120, 118, 110, 105, 100, 95],
            history_7d=[90, 100, 110, 105, 115, 120, 110, 100, 95, 105],
            history_24h=[100, 105, 110, 115, 120],
            static_threshold=500.0,
        )
        assert result.is_anomaly is False

    def test_result_to_dict(self):
        det = TimeSeriesDetector()
        result = det.detect(
            metric_name="cpu",
            current=99.0,
            history_1h=[80, 81, 82, 80, 81],
            static_threshold=80.0,
        )
        d = result.to_dict()
        assert "is_anomaly" in d
        assert "severity" in d
        assert "confidence" in d
        assert "triggers" in d
        assert "summary" in d
        assert "recommended_window" in d
