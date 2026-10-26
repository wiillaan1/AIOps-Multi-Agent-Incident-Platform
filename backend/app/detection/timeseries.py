"""
TimeSeriesDetector - 多窗口时序异常检测

4 个独立算法 + 综合判定：
1. Z-Score（μ + nσ）：检测实时尖峰（1h 窗口）
2. IQR（Q3 + 1.5×IQR）：检测同期异常值（7d 窗口）
3. 线性回归斜率：检测趋势恶化（24h/48h 窗口）
4. 静态阈值：兜底，无历史数据也能用

输出 DetectionResult 含：
- is_anomaly: 是否异常
- severity: critical/high/medium/low/normal
- confidence: 0-1
- triggers: 触发的算法列表（哪个算法 + 数值）
- summary: 人类可读的描述
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """异常严重度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NORMAL = "normal"


@dataclass
class AlgorithmTrigger:
    """单个算法的触发记录"""
    algorithm: str          # "z_score" / "iqr" / "trend" / "static"
    window: str             # "1h" / "24h" / "7d" / "static"
    triggered: bool
    metric_value: float     # 实际值
    threshold_value: float  # 触发阈值
    score: float            # 算法得分（0-1）
    description: str        # 算法描述

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "window": self.window,
            "triggered": self.triggered,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "score": round(self.score, 3),
            "description": self.description,
        }


@dataclass
class DetectionResult:
    """综合异常检测结果"""
    is_anomaly: bool
    severity: str             # Severity.value
    confidence: float         # 0-1
    triggers: list[AlgorithmTrigger] = field(default_factory=list)
    summary: str = ""         # 人类可读描述
    recommended_window: str = "1h"  # 优先关注窗口

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_anomaly": self.is_anomaly,
            "severity": self.severity,
            "confidence": round(self.confidence, 3),
            "triggers": [t.to_dict() for t in self.triggers],
            "summary": self.summary,
            "recommended_window": self.recommended_window,
        }


class TimeSeriesDetector:
    """
    多窗口时序异常检测器。

    用法：
        detector = TimeSeriesDetector()
        result = detector.detect(
            metric_name="cpu_usage_percent",
            current=95.0,
            history_1h=[80, 82, 85, 83, 81, 88, 92, 90, 95],  # 最近 1h
            history_7d=[75, 78, 80, 79, 81, 80, 82, 85],      # 过去 7d
            static_threshold=80.0,
        )
        if result.is_anomaly:
            print(f"[{result.severity}] {result.summary}")
    """

    def __init__(
        self,
        z_score_threshold: float = 3.0,        # Z-Score > 3 触发
        iqr_multiplier: float = 1.5,           # IQR > 1.5×IQR 触发
        trend_slope_threshold: float = 0.1,    # 24h 斜率 > 0.1/h 触发（标准化）
        trend_min_points: int = 5,             # 趋势检测最少数据点
        z_score_min_points: int = 5,           # Z-Score 最少数据点
        iqr_min_points: int = 4,               # IQR 最少数据点
    ):
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.trend_slope_threshold = trend_slope_threshold
        self.trend_min_points = trend_min_points
        self.z_score_min_points = z_score_min_points
        self.iqr_min_points = iqr_min_points

    def detect(
        self,
        metric_name: str,
        current: float,
        history_1h: list[float] | None = None,
        history_24h: list[float] | None = None,
        history_7d: list[float] | None = None,
        static_threshold: float | None = None,
        static_direction: str = "above",  # "above" / "below"
    ) -> DetectionResult:
        """
        多窗口综合检测。

        参数：
            metric_name: 指标名（用于摘要）
            current: 当前值
            history_1h: 最近 1 小时数据点（~60 个，每分钟）
            history_24h: 最近 24h 数据点（~96 个，每 15 分钟）
            history_7d: 最近 7d 数据点（~28 个，每 6 小时）
            static_threshold: 静态阈值（兜底）
            static_direction: "above" 表示 current > threshold 触发
        """
        triggers: list[AlgorithmTrigger] = []

        # 1. Z-Score（1h 实时尖峰）
        z_trigger = self._z_score_check(current, history_1h or [])
        triggers.append(z_trigger)

        # 2. IQR（7d 同期基线）
        iqr_trigger = self._iqr_check(current, history_7d or [])
        triggers.append(iqr_trigger)

        # 3. 趋势（24h 线性回归）
        trend_trigger = self._trend_check(current, history_24h or [])
        triggers.append(trend_trigger)

        # 4. 静态阈值（兜底）
        if static_threshold is not None:
            triggers.append(self._static_check(current, static_threshold, static_direction))

        # 综合判定
        return self._aggregate(triggers, metric_name)

    # ---------- 4 个算法 ----------
    def _z_score_check(self, current: float, history: list[float]) -> AlgorithmTrigger:
        """Z-Score: current 与历史均值的偏离倍数"""
        if len(history) < self.z_score_min_points:
            return AlgorithmTrigger(
                algorithm="z_score", window="1h",
                triggered=False, metric_value=current,
                threshold_value=self.z_score_threshold, score=0.0,
                description=f"数据不足（{len(history)} < {self.z_score_min_points}）",
            )

        mu = statistics.mean(history)
        sigma = statistics.stdev(history) if len(history) > 1 else 0.0

        if sigma < 1e-9:
            # 几乎无波动，比较 current vs mu
            deviation = abs(current - mu)
            triggered = deviation > (abs(mu) * 0.1)  # 偏离 10%
            score = min(1.0, deviation / max(abs(mu), 1.0))
            return AlgorithmTrigger(
                algorithm="z_score", window="1h",
                triggered=triggered, metric_value=current,
                threshold_value=self.z_score_threshold, score=score,
                description=f"σ≈0（数据无波动），deviation={deviation:.2f}",
            )

        z = abs(current - mu) / sigma
        triggered = z > self.z_score_threshold
        score = min(1.0, z / (self.z_score_threshold * 2))  # 归一化到 0-1
        return AlgorithmTrigger(
            algorithm="z_score", window="1h",
            triggered=triggered, metric_value=current,
            threshold_value=self.z_score_threshold, score=score,
            description=f"μ={mu:.2f}, σ={sigma:.2f}, Z={z:.2f}",
        )

    def _iqr_check(self, current: float, history: list[float]) -> AlgorithmTrigger:
        """IQR: 四分位距法"""
        if len(history) < self.iqr_min_points:
            return AlgorithmTrigger(
                algorithm="iqr", window="7d",
                triggered=False, metric_value=current,
                threshold_value=self.iqr_multiplier, score=0.0,
                description=f"数据不足（{len(history)} < {self.iqr_min_points}）",
            )

        sorted_h = sorted(history)
        n = len(sorted_h)
        q1 = sorted_h[n // 4]
        q3 = sorted_h[(3 * n) // 4]
        iqr = q3 - q1

        upper_fence = q3 + self.iqr_multiplier * iqr
        lower_fence = q1 - self.iqr_multiplier * iqr

        triggered = current > upper_fence or current < lower_fence
        if iqr < 1e-9:
            score = 0.5 if triggered else 0.0
        else:
            distance = max(current - upper_fence, lower_fence - current, 0)
            score = min(1.0, distance / (iqr * 2))
        return AlgorithmTrigger(
            algorithm="iqr", window="7d",
            triggered=triggered, metric_value=current,
            threshold_value=upper_fence, score=score,
            description=f"Q1={q1:.2f}, Q3={q3:.2f}, upper_fence={upper_fence:.2f}",
        )

    def _trend_check(self, current: float, history: list[float]) -> AlgorithmTrigger:
        """线性回归：斜率"""
        if len(history) < self.trend_min_points:
            return AlgorithmTrigger(
                algorithm="trend", window="24h",
                triggered=False, metric_value=current,
                threshold_value=self.trend_slope_threshold, score=0.0,
                description=f"数据不足（{len(history)} < {self.trend_min_points}）",
            )

        # 简单线性回归：y = slope * x + intercept
        # x = 0, 1, 2, ... ; y = history
        n = len(history)
        xs = list(range(n))
        ys = history
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_xx = sum(x * x for x in xs)

        denom = n * sum_xx - sum_x ** 2
        if abs(denom) < 1e-9:
            slope = 0.0
            intercept = sum_y / n if n else 0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n

        # 当前点 (n) 与回归线的偏离
        predicted = slope * n + intercept
        deviation = abs(current - predicted)

        # 触发：斜率绝对值 > 阈值 AND 当前值 > 回归预测
        triggered = (
            abs(slope) > self.trend_slope_threshold
            and current > predicted
        )
        score = min(1.0, abs(slope) / (self.trend_slope_threshold * 3))

        return AlgorithmTrigger(
            algorithm="trend", window="24h",
            triggered=triggered, metric_value=current,
            threshold_value=self.trend_slope_threshold, score=score,
            description=f"slope={slope:.3f}/step, predicted={predicted:.2f}, dev={deviation:.2f}",
        )

    def _static_check(
        self,
        current: float,
        threshold: float,
        direction: str = "above",
    ) -> AlgorithmTrigger:
        """静态阈值"""
        if direction == "above":
            triggered = current > threshold
            deviation = current - threshold
        else:
            triggered = current < threshold
            deviation = threshold - current

        # 偏离度归一化（按阈值百分比）
        pct = deviation / abs(threshold) if abs(threshold) > 1e-9 else 0.0
        score = min(1.0, pct / 0.5)  # 偏离 50% 得满分

        return AlgorithmTrigger(
            algorithm="static", window="static",
            triggered=triggered, metric_value=current,
            threshold_value=threshold, score=score,
            description=f"threshold={threshold}, dev={deviation:.2f} ({pct*100:.1f}%)",
        )

    # ---------- 综合判定 ----------
    def _aggregate(
        self,
        triggers: list[AlgorithmTrigger],
        metric_name: str,
    ) -> DetectionResult:
        """综合多个算法的结果"""
        triggered = [t for t in triggers if t.triggered]
        is_anomaly = len(triggered) > 0

        # confidence：仅看触发的算法的最高 score（不触发 = 0）
        if triggered:
            confidence = max(t.score for t in triggered)
        else:
            confidence = 0.0  # 未触发就是 0（哪怕有些算法 score > 0 也没用）

        # severity 映射（基于 confidence + 触发数）
        severity = self._severity(confidence, len(triggered))

        # 推荐窗口：取最严重的触发的窗口
        recommended = "static"
        if triggered:
            # 优先级：z_score (1h) > trend (24h) > iqr (7d) > static
            priority = {"z_score": 1, "trend": 2, "iqr": 3, "static": 4}
            triggered.sort(key=lambda t: priority.get(t.algorithm, 99))
            recommended = triggered[0].window

        # summary
        if is_anomaly:
            descs = [t.description for t in triggered]
            summary = f"{metric_name} 异常（{len(triggered)}/{len(triggers)} 算法触发）: {'; '.join(descs)}"
        else:
            summary = f"{metric_name} 正常（{len(triggers)} 算法均未触发）"

        return DetectionResult(
            is_anomaly=is_anomaly,
            severity=severity,
            confidence=confidence,
            triggers=triggers,
            summary=summary,
            recommended_window=recommended,
        )

    @staticmethod
    def _severity(confidence: float, trigger_count: int) -> str:
        """根据 confidence 和触发数判定 severity"""
        # 多算法同时触发 → 升级严重度
        base_score = confidence
        if trigger_count >= 3:
            base_score = min(1.0, base_score + 0.2)
        elif trigger_count == 2:
            base_score = min(1.0, base_score + 0.1)

        if base_score >= 0.85:
            return Severity.CRITICAL.value
        elif base_score >= 0.65:
            return Severity.HIGH.value
        elif base_score >= 0.4:
            return Severity.MEDIUM.value
        elif base_score > 0:
            return Severity.LOW.value
        else:
            return Severity.NORMAL.value
