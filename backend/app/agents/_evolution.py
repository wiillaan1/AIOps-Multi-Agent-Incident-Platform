"""
M1 - 三路径自进化 Implementation (GREEN 阶段)

借鉴 Devix §6.3 - 三条自进化路径：
1. RCA→Playbook 沉淀（高置信度 RCA 自动转为 Playbook）
2. Badcase 捕获（失败 case 累计，触发 alert）
3. 反馈调优（人工评分 → 调整 prompt 权重）

只做测试要求的功能。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# 路径 1：RCA → Playbook
@dataclass
class Playbook:
    """从 RCA 沉淀出的 Playbook"""
    name: str
    root_cause: str
    confidence: float
    actions: list[str]
    evidence: list[str]
    source_incident_id: str


class RCAToPlaybookConverter:
    """RCA → Playbook 转换器"""

    # 高于此置信度才沉淀
    CONFIDENCE_THRESHOLD = 0.85

    def convert(self, rca_result: dict[str, Any]) -> Playbook | None:
        confidence = rca_result.get("confidence", 0.0)
        if confidence < self.CONFIDENCE_THRESHOLD:
            return None

        return Playbook(
            name=f"PB-{rca_result['incident_id']}",
            root_cause=rca_result.get("root_cause", ""),
            confidence=confidence,
            actions=[rca_result.get("heal_action", "")],
            evidence=rca_result.get("evidence", []),
            source_incident_id=rca_result.get("incident_id", ""),
        )


# 路径 2：Badcase 捕获
@dataclass
class BadcaseEntry:
    """失败 case 条目"""
    incident_id: str
    root_cause: str
    confidence: float
    failure_reason: str


class BadcaseCapture:
    """Badcase 捕获器"""

    # 同一根因出现此次数触发 alert
    ALERT_THRESHOLD = 3

    def __init__(self) -> None:
        self._entries: list[BadcaseEntry] = []

    def capture_failed(
        self,
        incident_id: str,
        root_cause: str,
        confidence: float,
        failure_reason: str,
    ) -> BadcaseEntry:
        entry = BadcaseEntry(
            incident_id=incident_id,
            root_cause=root_cause,
            confidence=confidence,
            failure_reason=failure_reason,
        )
        self._entries.append(entry)
        return entry

    def group_by_root_cause(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for entry in self._entries:
            groups[entry.root_cause] = groups.get(entry.root_cause, 0) + 1
        return groups

    def should_alert(self, root_cause: str) -> bool:
        count = sum(1 for e in self._entries if e.root_cause == root_cause)
        return count >= self.ALERT_THRESHOLD


# 路径 3：反馈 → Prompt 调优
@dataclass
class Feedback:
    """人工反馈"""
    incident_id: str
    prompt_section: str
    rating: int  # 1-5
    comment: str = ""


class PromptOptimizer:
    """Prompt 权重优化器"""

    # 权重调整步长
    POSITIVE_DELTA = 0.10
    NEGATIVE_DELTA = 0.10

    def __init__(self) -> None:
        self._weights: dict[str, float] = {}
        self._history: list[Feedback] = []

    def get_weight(self, section: str) -> float:
        return self._weights.get(section, 1.0)

    def apply_feedback(self, feedback: Feedback) -> None:
        self._history.append(feedback)
        current = self.get_weight(feedback.prompt_section)
        if feedback.rating >= 4:
            new_weight = current + self.POSITIVE_DELTA
        elif feedback.rating <= 2:
            new_weight = max(0.1, current - self.NEGATIVE_DELTA)
        else:
            new_weight = current  # 中性反馈不调整
        self._weights[feedback.prompt_section] = new_weight

    def history(self) -> list[Feedback]:
        return list(self._history)


# Pipeline
@dataclass
class EvolutionResult:
    path1_executed: bool  # RCA→Playbook
    path2_executed: bool  # Badcase
    path3_executed: bool  # Feedback
    playbook: Playbook | None = None
    badcase: BadcaseEntry | None = None


class EvolutionPipeline:
    """3 路径自进化 Pipeline"""

    def __init__(self) -> None:
        self.converter = RCAToPlaybookConverter()
        self.badcase_capture = BadcaseCapture()
        self.prompt_optimizer = PromptOptimizer()

    def evolve(
        self,
        rca_result: dict[str, Any],
        feedback: Feedback | None = None,
        failed: bool = False,
    ) -> EvolutionResult:
        result = EvolutionResult(
            path1_executed=False,
            path2_executed=False,
            path3_executed=False,
        )

        # 路径 1：高置信度 RCA → Playbook
        playbook = self.converter.convert(rca_result)
        if playbook is not None:
            result.path1_executed = True
            result.playbook = playbook

        # 路径 2：失败 → Badcase
        if failed:
            result.badcase = self.badcase_capture.capture_failed(
                incident_id=rca_result.get("incident_id", "unknown"),
                root_cause=rca_result.get("root_cause", ""),
                confidence=rca_result.get("confidence", 0.0),
                failure_reason="execution_failed",
            )
            result.path2_executed = True

        # 路径 3：有反馈 → 调优
        if feedback is not None:
            self.prompt_optimizer.apply_feedback(feedback)
            result.path3_executed = True

        return result