"""
TriggerDedupService — 告警去重 + 跨 incident in-flight 上限

修复审查报告（中期补 #10）：
- 5 分钟内相同 fingerprint 的告警 → dedup（返回 fingerprint + 现有 incident_id）
- 跨 incident in-flight 上限（默认 10）→ 超出拒绝新触发
- 内存版实现（启动丢失）；生产环境可对接 Redis（接口稳定）

API：
- compute_alert_fingerprint(alert) → stable hash 字符串
- TriggerDedupService.should_dedupe(fingerprint, ...) → (is_dup, existing_incident_id)
- TriggerDedupService.mark_in_flight(fingerprint, incident_id)
- TriggerDedupService.mark_done(incident_id)
- TriggerDedupService.in_flight_count() → int
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any


DEFAULT_DEDUP_WINDOW_SECONDS = 300  # 5 分钟（审查报告：相同 alert 反复 trigger）
DEFAULT_MAX_IN_FLIGHT = 10  # 跨 incident 在处理中的上限


def compute_alert_fingerprint(
    service: str,
    metric: str,
    severity: str,
    scope: str | None = None,
) -> str:
    """
    生成告警 fingerprint（稳定的 dedup key）。

    字段选择：service + metric + severity + scope（无 scope 时用 "default"）。
    这些都是"告警身份"的核心字段；alert_id/timestamp 不参与 fingerprint
    （timestamp 变化但身份不变 → 仍应被 dedup）。
    """
    parts = [
        str(service or "").strip().lower(),
        str(metric or "").strip().lower(),
        str(severity or "").strip().lower(),
        str(scope or "default").strip().lower(),
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class _DedupRecord:
    """fingerprint → incident_id + last_seen。"""

    fingerprint: str
    incident_id: str
    last_seen: datetime
    hit_count: int = 1


@dataclass
class _InFlightRecord:
    incident_id: str
    started_at: datetime
    fingerprint: str


class TriggerDedupService:
    """告警去重 + in-flight 上限（线程安全）。"""

    def __init__(
        self,
        dedup_window_seconds: int = DEFAULT_DEDUP_WINDOW_SECONDS,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    ) -> None:
        self.dedup_window = timedelta(seconds=dedup_window_seconds)
        self.max_in_flight = max_in_flight
        self._lock = threading.Lock()
        self._dedup_records: dict[str, _DedupRecord] = {}  # fingerprint → record
        self._in_flight: dict[str, _InFlightRecord] = {}  # incident_id → record

    # ----- 去重 -----

    def should_dedupe(
        self,
        fingerprint: str,
        now: datetime | None = None,
    ) -> tuple[bool, str | None]:
        """查询 fingerprint 是否在去重窗口内。

        Returns:
            (是否去重, 现有 incident_id 或 None)
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            self._cleanup_expired(now)
            record = self._dedup_records.get(fingerprint)
            if record is None:
                return False, None
            # 在窗口内 → dedup
            if now - record.last_seen <= self.dedup_window:
                record.hit_count += 1
                record.last_seen = now
                return True, record.incident_id
            return False, None

    def mark_in_flight(
        self,
        fingerprint: str,
        incident_id: str,
        now: datetime | None = None,
    ) -> None:
        """记录 fingerprint → incident_id 映射。"""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            self._dedup_records[fingerprint] = _DedupRecord(
                fingerprint=fingerprint,
                incident_id=incident_id,
                last_seen=now,
            )
            self._in_flight[incident_id] = _InFlightRecord(
                incident_id=incident_id,
                started_at=now,
                fingerprint=fingerprint,
            )

    def mark_done(self, incident_id: str) -> None:
        """标记 incident 处理完成（释放 in-flight 槽）。"""
        with self._lock:
            self._in_flight.pop(incident_id, None)

    def get_record(self, fingerprint: str) -> _DedupRecord | None:
        """查询 fingerprint 当前记录（用于测试/审计）。"""
        with self._lock:
            return self._dedup_records.get(fingerprint)

    # ----- in-flight 上限 -----

    def in_flight_count(self) -> int:
        return len(self._in_flight)

    def can_accept_new(self) -> bool:
        """是否还能接收新 incident（in-flight 未达上限）。"""
        return self.in_flight_count() < self.max_in_flight

    def list_in_flight(self) -> list[_InFlightRecord]:
        with self._lock:
            return list(self._in_flight.values())

    # ----- 维护 -----

    def _cleanup_expired(self, now: datetime) -> None:
        """清理过期的 dedup 记录（保留 in-flight 由 mark_done 主动释放）。"""
        cutoff = now - self.dedup_window
        expired_fps = [
            fp for fp, rec in self._dedup_records.items()
            if rec.last_seen < cutoff
        ]
        for fp in expired_fps:
            # 只清理 dedup，不动 in-flight（in-flight 可能更长寿）
            del self._dedup_records[fp]

    def clear(self) -> None:
        """清空所有记录（仅测试用）。"""
        with self._lock:
            self._dedup_records.clear()
            self._in_flight.clear()


# ===== 单例（带线程安全的懒加载） =====


_default_service: TriggerDedupService | None = None
_default_lock = threading.Lock()


def get_trigger_dedup_service() -> TriggerDedupService:
    """获取全局单例的 TriggerDedupService。"""
    global _default_service
    if _default_service is None:
        with _default_lock:
            if _default_service is None:
                _default_service = TriggerDedupService()
    return _default_service


def reset_trigger_dedup_service() -> None:
    """重置全局单例（仅测试用）。"""
    global _default_service
    with _default_lock:
        _default_service = None
