"""W8.4 — IncidentCorrelator (跨 incident 关联分析).

识别同一时间窗口内多个 incident 的关联性。例：
- "今天 5 个 incident 是否其实是同一个根因"
- "order-service 这 10 分钟的 3 个告警是否是级联故障"

设计原则：
- 基于 W8.1 的 `entity_snapshot` 做相似度（扁平化对比）
- 启发式评分（4 维度），不需要 InvestigationGraph
- 默认关闭（feature flag），关闭 = 端点返回 404 / 空结果
- 不修改 incident 数据，只读分析

相似度算法（4 维度加权）：
- service 相同：+0.4
- root_cause 相同：+0.3
- 5 路 evidence 至少 3 路相同：+0.2
- steps 序列相似：+0.1

阈值：similarity >= 0.5 算"关联"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ===== Feature Flag =====


def is_w8_4_correlate_enabled() -> bool:
    """判断 W8.4 跨 incident 关联是否启用。默认关闭。"""
    val = os.getenv("AIOPS_USE_W8_4_CORRELATE", "false").lower()
    return val in ("true", "1", "yes", "on")


# ===== 关联结果 =====


@dataclass
class CorrelationResult:
    """两个 incident 之间的关联度。"""

    incident_id: str
    similarity: float
    common_services: list[str] = field(default_factory=list)
    shared_root_cause: str | None = None
    matched_evidence_kinds: list[str] = field(default_factory=list)
    step_overlap_count: int = 0
    service: str = ""
    created_at: str = ""
    note: str = ""


# ===== 关联分析器 =====


class IncidentCorrelator:
    """跨 incident 关联分析。"""

    # 4 维度权重
    WEIGHT_SERVICE = 0.4
    WEIGHT_ROOT_CAUSE = 0.3
    WEIGHT_EVIDENCE = 0.2
    WEIGHT_STEPS = 0.1

    # 关联阈值
    SIMILARITY_THRESHOLD = 0.5

    # 时间窗口（分钟）
    DEFAULT_WINDOW_MINUTES = 60

    def __init__(self) -> None:
        self._enabled = is_w8_4_correlate_enabled()

    def correlate(
        self,
        target_incident: dict,
        candidate_incidents: list[dict],
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
    ) -> list[CorrelationResult]:
        """对 target incident 与 candidate 列表算关联度。

        Args:
            target_incident: 目标 incident 字典（包含 entity_snapshot 字段）
            candidate_incidents: 候选 incident 列表
            window_minutes: 时间窗口（默认 60 分钟）

        Returns:
            按 similarity 降序排序的 CorrelationResult 列表
        """
        if not self._enabled:
            return []

        try:
            target_snapshot = self._extract_snapshot(target_incident)
            if not target_snapshot:
                return []

            target_time = self._extract_time(target_incident)
            if target_time is None:
                return []

            results: list[CorrelationResult] = []
            for candidate in candidate_incidents:
                # 排除自己
                if self._extract_id(candidate) == self._extract_id(target_incident):
                    continue

                # 时间窗口过滤
                cand_time = self._extract_time(candidate)
                if cand_time is None:
                    continue
                if abs((target_time - cand_time).total_seconds()) > window_minutes * 60:
                    continue

                # 算相似度
                result = self._compute_similarity(target_snapshot, candidate)
                if result and result.similarity >= self.SIMILARITY_THRESHOLD:
                    results.append(result)

            # 按 similarity 降序
            results.sort(key=lambda r: r.similarity, reverse=True)
            return results

        except Exception as exc:
            logger.warning("W8.4 correlate failed (non-fatal)", error=str(exc))
            return []

    def correlate_by_entity_snapshot(
        self,
        target_snapshot: dict,
        candidate_snapshots: list[dict],
    ) -> list[CorrelationResult]:
        """直接对 entity_snapshot 列表算关联（无需完整 incident dict）。

        用途：测试 / e2e 评测场景。
        """
        if not self._enabled:
            return []

        results: list[CorrelationResult] = []
        for cand_snapshot in candidate_snapshots:
            if cand_snapshot.get("schema_version") != "1.0":
                continue

            similarity = self._compute_snapshot_similarity(target_snapshot, cand_snapshot)
            if similarity < self.SIMILARITY_THRESHOLD:
                continue

            result = CorrelationResult(
                incident_id=cand_snapshot.get("incident_id", "unknown"),
                similarity=similarity,
                common_services=self._common_services(target_snapshot, cand_snapshot),
                shared_root_cause=self._shared_root_cause(target_snapshot, cand_snapshot),
                matched_evidence_kinds=self._common_evidence_kinds(target_snapshot, cand_snapshot),
                step_overlap_count=self._step_overlap(target_snapshot, cand_snapshot),
                service=cand_snapshot.get("services", [""])[0] if cand_snapshot.get("services") else "",
                created_at=cand_snapshot.get("created_at", ""),
            )
            results.append(result)

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    # ----- 内部方法 -----

    def _extract_snapshot(self, incident: dict) -> dict | None:
        """从 incident 提取 entity_snapshot。"""
        return incident.get("entity_snapshot")

    def _extract_id(self, incident: dict) -> str:
        """提取 incident id。"""
        return incident.get("incident_id") or incident.get("id") or ""

    def _extract_time(self, incident: dict) -> datetime | None:
        """提取 incident 时间。"""
        ts = (
            incident.get("created_at")
            or incident.get("timestamp")
            or incident.get("alert_time")
        )
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            try:
                # 处理 ISO 格式
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                return datetime.fromisoformat(ts)
            except (ValueError, TypeError):
                return None
        return None

    def _compute_similarity(
        self,
        target_snapshot: dict,
        candidate: dict,
    ) -> CorrelationResult | None:
        """计算 target 与 candidate 的相似度。"""
        cand_snapshot = candidate.get("entity_snapshot")
        if not cand_snapshot:
            return None

        similarity = self._compute_snapshot_similarity(target_snapshot, cand_snapshot)
        if similarity < self.SIMILARITY_THRESHOLD:
            return None

        return CorrelationResult(
            incident_id=self._extract_id(candidate),
            similarity=similarity,
            common_services=self._common_services(target_snapshot, cand_snapshot),
            shared_root_cause=self._shared_root_cause(target_snapshot, cand_snapshot),
            matched_evidence_kinds=self._common_evidence_kinds(target_snapshot, cand_snapshot),
            step_overlap_count=self._step_overlap(target_snapshot, cand_snapshot),
            service=cand_snapshot.get("services", [""])[0] if cand_snapshot.get("services") else "",
            created_at=self._extract_time(candidate).isoformat() if self._extract_time(candidate) else "",
        )

    def _compute_snapshot_similarity(
        self,
        s1: dict,
        s2: dict,
    ) -> float:
        """两个 snapshot 之间的相似度（4 维度加权）。"""
        # 1. service 相同
        services1 = set(s1.get("services", []))
        services2 = set(s2.get("services", []))
        service_score = self.WEIGHT_SERVICE if services1 & services2 else 0.0

        # 2. root_cause 相同
        rc1 = s1.get("root_cause")
        rc2 = s2.get("root_cause")
        rc_score = self.WEIGHT_ROOT_CAUSE if (rc1 and rc1 == rc2) else 0.0

        # 3. evidence 至少 3 路相同
        kinds1 = set(b.get("source_kind", "") for b in s1.get("evidence_blocks", []))
        kinds2 = set(b.get("source_kind", "") for b in s2.get("evidence_blocks", []))
        common_kinds = kinds1 & kinds2
        # 排除空 kind
        common_kinds.discard("")
        evidence_score = self.WEIGHT_EVIDENCE if len(common_kinds) >= 3 else 0.0

        # 4. steps 序列相似（jaccard 相似度）
        steps1 = set(s1.get("steps", []))
        steps2 = set(s2.get("steps", []))
        if steps1 or steps2:
            jaccard = len(steps1 & steps2) / len(steps1 | steps2)
            steps_score = self.WEIGHT_STEPS * jaccard
        else:
            steps_score = 0.0

        return service_score + rc_score + evidence_score + steps_score

    def _common_services(self, s1: dict, s2: dict) -> list[str]:
        return sorted(set(s1.get("services", [])) & set(s2.get("services", [])))

    def _shared_root_cause(self, s1: dict, s2: dict) -> str | None:
        rc1 = s1.get("root_cause")
        rc2 = s2.get("root_cause")
        if rc1 and rc1 == rc2:
            return rc1
        return None

    def _common_evidence_kinds(self, s1: dict, s2: dict) -> list[str]:
        kinds1 = set(b.get("source_kind", "") for b in s1.get("evidence_blocks", []))
        kinds2 = set(b.get("source_kind", "") for b in s2.get("evidence_blocks", []))
        common = (kinds1 & kinds2) - {""}
        return sorted(common)

    def _step_overlap(self, s1: dict, s2: dict) -> int:
        return len(set(s1.get("steps", [])) & set(s2.get("steps", [])))


# ===== 工厂函数 =====


def get_incident_correlator() -> IncidentCorrelator:
    """获取 IncidentCorrelator 单例。"""
    return IncidentCorrelator()
