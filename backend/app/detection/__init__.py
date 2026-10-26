"""
Detection 模块 - 多窗口时序异常检测（Zenjoy 模式）

设计原则（来自 Zenjoy AWS AIOps Agent）：
1. 多窗口时序分析：1h 实时尖峰 + 24h/48h 趋势 + 7d 同期基线
2. 所有数学计算由 Python 程序完成
3. LLM 只在异常判定后做"自然语言解释"（Phase C2）
4. 静态阈值兜底，防止历史数据稀疏时算法失效

算法选择：
- Z-Score: 适合 1h 实时尖峰（需要 ~30 个数据点）
- IQR: 适合 7d 同期基线（需要 ~14 个数据点）
- 线性回归: 适合 24h/48h 趋势检测
- 静态阈值: 兜底，不需要历史数据
"""
from app.detection.timeseries import (
    TimeSeriesDetector,
    DetectionResult,
    Severity,
    AlgorithmTrigger,
)

__all__ = [
    "TimeSeriesDetector",
    "DetectionResult",
    "Severity",
    "AlgorithmTrigger",
]
