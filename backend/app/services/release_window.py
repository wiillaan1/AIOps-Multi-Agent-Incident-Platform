"""
ReleaseWindow — 变更窗口追踪 + 自愈冻结

修复审查报告（中期补 #8）：发布期/变更窗口自动冻结自愈。

设计：
- ReleaseWindow：服务某次发布的开始/结束时间窗口
- ReleaseWindowStore：JSONL append-only，类似 audit_log 风格（线程安全 + 启动恢复）
- is_in_release_window(service_name) → (是否在窗口, 窗口元数据)
- 在 release 窗口内，HealAgent 应强制 dry-run 并记录 reason="release_freeze"

未来接入：可对接 Jenkins/Spinnaker/Argo CD webhook 自动开窗。
当前接口已稳定，外部对接只需替换 store 后端。
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class ReleaseWindow:
    """单次发布/变更窗口。

    Attributes:
        id: UUID
        service_name: 服务名（如 "order-service"）
        version: 发布版本（如 "v2.3.1"）
        started_at: 窗口开始时间（UTC）
        ended_at: 窗口结束时间（UTC，None 表示进行中）
        release_type: "deployment" / "config_change" / "maintenance" / "rollback"
        actor: 发起人/系统（如 "user:alice" / "system:jenkins"）
        notes: 备注
    """

    service_name: str
    version: str
    started_at: datetime
    release_type: str = "deployment"
    actor: str = "system:unknown"
    notes: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ended_at: datetime | None = None

    def is_active(self, now: datetime | None = None) -> bool:
        """窗口是否在进行中（started_at ≤ now ≤ ended_at）。"""
        now = now or datetime.now(timezone.utc)
        if now < self.started_at:
            return False
        if self.ended_at is None:
            return True
        return now <= self.ended_at

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        d["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReleaseWindow":
        return cls(
            id=d["id"],
            service_name=d["service_name"],
            version=d["version"],
            started_at=datetime.fromisoformat(d["started_at"]),
            ended_at=datetime.fromisoformat(d["ended_at"]) if d["ended_at"] else None,
            release_type=d.get("release_type", "deployment"),
            actor=d.get("actor", "system:unknown"),
            notes=d.get("notes", ""),
        )


DEFAULT_WINDOW_HOURS = 4  # 默认变更窗口长度


class ReleaseWindowStore:
    """JSONL append-only 持久化的 ReleaseWindow 存储。

    与 audit_log 风格一致：startup 加载全部窗口到内存（轻量，通常 <1k 行），
    append 时加线程锁，写盘失败抛异常不静默（与 audit_log 不同：这里允许外部
    决定是否重试）。
    """

    def __init__(self, store_path: str = "backend/data/release_windows.jsonl") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._windows: dict[str, ReleaseWindow] = {}  # id → ReleaseWindow
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        try:
            with self.store_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        w = ReleaseWindow.from_dict(d)
                        self._windows[w.id] = w
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # 单行损坏跳过（append-only 不修改历史）
                        continue
        except OSError:
            pass

    def _append(self, window: ReleaseWindow) -> None:
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(window.to_dict(), ensure_ascii=False) + "\n")

    def open_window(
        self,
        service_name: str,
        version: str,
        release_type: str = "deployment",
        actor: str = "system:unknown",
        notes: str = "",
        now: datetime | None = None,
    ) -> ReleaseWindow:
        """开一个新的变更窗口。

        如果同一 service_name + 未结束 → 自动结束上一个再开新的（避免重叠）。
        """
        now = now or datetime.now(timezone.utc)
        with self._lock:
            # 关闭同服务所有未结束的窗口
            for w in list(self._windows.values()):
                if (
                    w.service_name == service_name
                    and w.ended_at is None
                    and w.is_active(now=now)
                ):
                    w.ended_at = now
                    self._append(w)
            new_window = ReleaseWindow(
                service_name=service_name,
                version=version,
                started_at=now,
                release_type=release_type,
                actor=actor,
                notes=notes,
            )
            self._windows[new_window.id] = new_window
            self._append(new_window)
            return new_window

    def close_window(
        self,
        window_id: str,
        now: datetime | None = None,
    ) -> ReleaseWindow | None:
        """结束一个变更窗口。"""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            w = self._windows.get(window_id)
            if w is None:
                return None
            if w.ended_at is None:
                w.ended_at = now
                self._append(w)
            return w

    def close_service_window(
        self,
        service_name: str,
        now: datetime | None = None,
    ) -> ReleaseWindow | None:
        """关闭某服务当前活动的窗口（如果有）。"""
        now = now or datetime.now(timezone.utc)
        with self._lock:
            for w in list(self._windows.values()):
                if (
                    w.service_name == service_name
                    and w.ended_at is None
                    and w.is_active(now=now)
                ):
                    w.ended_at = now
                    self._append(w)
                    return w
            return None

    def is_in_release_window(
        self,
        service_name: str,
        now: datetime | None = None,
    ) -> tuple[bool, ReleaseWindow | None]:
        """查询某服务是否在变更窗口内。

        Returns:
            (是否在窗口, 活动窗口对象或 None)
        """
        now = now or datetime.now(timezone.utc)
        for w in self._windows.values():
            if w.service_name == service_name and w.is_active(now=now):
                return True, w
        return False, None

    def list_active_windows(
        self,
        now: datetime | None = None,
    ) -> list[ReleaseWindow]:
        """列出所有进行中的窗口。"""
        now = now or datetime.now(timezone.utc)
        return [w for w in self._windows.values() if w.is_active(now=now)]

    def get_window(self, window_id: str) -> ReleaseWindow | None:
        return self._windows.get(window_id)
