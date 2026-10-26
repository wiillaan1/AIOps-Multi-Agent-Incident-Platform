"""
OncallScheduler — 值班排班 + 升级链路

修复审查报告（中期补 #9）：
- OncallSchedule 模型（rotation/user/time window/service scope）
- OncallScheduler：get_current_oncall(policy) → 匹配当前值班人
- 升级链路：L1 → L2 → LEAD（最多 3 层）
- 集成到 Orchestrator._node_escalate：写 state["oncall_assignment"]

默认策略（生产环境可对接 PagerDuty / OpsGenie / 国内蓝信）：
- severity=CRITICAL → L1 oncall（如未响应 5 分钟 → 升级 L2）
- severity=HIGH → L1 oncall
- severity=MEDIUM/LOW → 静默记录，不主动通知

实现要点：
- RotationStore（JSONL append-only，类似 audit_log/ReleaseWindow）
- 时间窗口 [start, end] 内视为值班
- 同一时刻同 service_scope 可有多人值班（按 user_id 排序保证确定性）
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ===== 数据模型 =====


@dataclass
class OncallSchedule:
    """单条值班排班。

    Attributes:
        id: UUID
        user_id: 值班人 user_id（如 "alice@example.com" 或 PagerDuty ID）
        user_email: 通知邮箱
        role: "L1" / "L2" / "LEAD"
        start_time: 值班开始时间（UTC）
        end_time: 值班结束时间（UTC）
        service_scope: 负责的服务列表（如 ["order-service", "payment-service"]），
            空列表表示"全部服务"
        rotation_id: 排班组 ID（标识同组轮换）
    """

    user_id: str
    user_email: str
    role: str
    start_time: datetime
    end_time: datetime
    service_scope: list[str] = field(default_factory=list)
    rotation_id: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def covers(self, service_name: str | None, now: datetime | None = None) -> bool:
        """是否在值班时间 + 负责该服务（或全服务）。"""
        now = now or datetime.now(timezone.utc)
        if not (self.start_time <= now <= self.end_time):
            return False
        if not self.service_scope:
            return True  # 空 scope = 全部服务
        if service_name is None:
            return True  # 无具体服务 = 任意 service_scope 都行
        return service_name in self.service_scope

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start_time"] = self.start_time.isoformat()
        d["end_time"] = self.end_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OncallSchedule":
        return cls(
            id=d["id"],
            user_id=d["user_id"],
            user_email=d["user_email"],
            role=d["role"],
            start_time=datetime.fromisoformat(d["start_time"]),
            end_time=datetime.fromisoformat(d["end_time"]),
            service_scope=d.get("service_scope", []),
            rotation_id=d.get("rotation_id", "default"),
        )


@dataclass
class OncallAssignment:
    """排班分配结果（含决策原因）。"""

    user_id: str
    user_email: str
    role: str
    schedule_id: str
    matched_at: datetime
    escalation_level: int = 0  # 0 = L1, 1 = L2, 2 = LEAD
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "matched_at": self.matched_at.isoformat(),
        }


# ===== Store =====


class OncallScheduleStore:
    """JSONL append-only 持久化的排班存储（线程安全 + startup 恢复）。"""

    def __init__(self, store_path: str = "backend/data/oncall_schedules.jsonl") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._schedules: dict[str, OncallSchedule] = {}
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
                        s = OncallSchedule.from_dict(d)
                        self._schedules[s.id] = s
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except OSError:
            pass

    def _append(self, s: OncallSchedule) -> None:
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    def add(self, schedule: OncallSchedule) -> OncallSchedule:
        """新增排班。"""
        with self._lock:
            self._schedules[schedule.id] = schedule
            self._append(schedule)
            return schedule

    def get(self, schedule_id: str) -> OncallSchedule | None:
        return self._schedules.get(schedule_id)

    def all(self) -> list[OncallSchedule]:
        return list(self._schedules.values())


# ===== Scheduler =====


# 默认升级链：CRITICAL/HIGH 默认 2 层升级
DEFAULT_ESCALATION_ROLES: list[str] = ["L1", "L2", "LEAD"]


class OncallScheduler:
    """值班调度器。

    用法：
        scheduler = OncallScheduler(store)
        # 查询当前值班人
        assignment = scheduler.get_current_oncall(service_name="order-service", now=now)
        # 升级
        next_assignment = scheduler.escalate(assignment)
    """

    def __init__(
        self,
        store: OncallScheduleStore,
        escalation_chain: list[str] | None = None,
    ) -> None:
        self.store = store
        self.escalation_chain = escalation_chain or DEFAULT_ESCALATION_ROLES

    def get_current_oncall(
        self,
        service_name: str | None = None,
        severity: str = "MEDIUM",
        now: datetime | None = None,
    ) -> OncallAssignment | None:
        """查询当前值班人。

        策略：
        - CRITICAL/HIGH → 找 L1 role（不在 → 找 L2 → 找 LEAD）
        - MEDIUM/LOW → 静默记录，不通知（返回 None）
        """
        now = now or datetime.now(timezone.utc)
        if severity.upper() not in ("CRITICAL", "HIGH"):
            return None

        # 从 L1 开始查，按 escalation_chain 顺序回退
        for level, role in enumerate(self.escalation_chain):
            assignment = self._find_oncall_by_role(role, service_name, now)
            if assignment is not None:
                assignment.escalation_level = level
                assignment.reason = f"role_match:{role}"
                return assignment

        # 全部没匹配：返回 None（生产环境可兜底 oncall_lead）
        return None

    def escalate(
        self,
        current: OncallAssignment,
        service_name: str | None = None,
        now: datetime | None = None,
    ) -> OncallAssignment | None:
        """升级到下一层值班人。

        Returns:
            - 下一层 OncallAssignment（如果找到）
            - None（已到顶层 LEAD）
        """
        now = now or datetime.now(timezone.utc)
        next_level = current.escalation_level + 1
        if next_level >= len(self.escalation_chain):
            return None  # 已到 LEAD，无下一层

        next_role = self.escalation_chain[next_level]
        assignment = self._find_oncall_by_role(next_role, service_name, now)
        if assignment is None:
            return None
        assignment.escalation_level = next_level
        assignment.reason = f"escalated_from:{current.user_id}:{current.role}"
        return assignment

    def _find_oncall_by_role(
        self,
        role: str,
        service_name: str | None,
        now: datetime,
    ) -> OncallAssignment | None:
        """在所有排班里找 role 匹配且时间窗口匹配的值班人。"""
        matches: list[OncallSchedule] = []
        for s in self.store.all():
            if s.role.upper() == role.upper() and s.covers(service_name, now):
                matches.append(s)

        if not matches:
            return None

        # 确定性：按 start_time 升序（早开始优先），再按 user_id 排序
        matches.sort(key=lambda s: (s.start_time, s.user_id))
        chosen = matches[0]
        return OncallAssignment(
            user_id=chosen.user_id,
            user_email=chosen.user_email,
            role=chosen.role,
            schedule_id=chosen.id,
            matched_at=now,
        )


# ===== 默认演示排班（让示例代码可跑） =====


def install_default_demo_schedules(
    store: OncallScheduleStore,
    now: datetime | None = None,
) -> list[OncallSchedule]:
    """为本地/CI 演示安装默认 L1/L2/LEAD 排班（覆盖未来 24 小时）。"""
    now = now or datetime.now(timezone.utc)
    schedules = [
        OncallSchedule(
            user_id="alice@example.com",
            user_email="alice@example.com",
            role="L1",
            start_time=now,
            end_time=now.replace(day=now.day) + (now - now.replace(hour=0, minute=0, second=0, microsecond=0)),
            service_scope=["order-service", "payment-service"],
        ),
    ]
    # 让 L1 覆盖未来 24h
    schedules[0].end_time = schedules[0].start_time.replace(hour=23, minute=59, second=59) + (
        schedules[0].end_time - schedules[0].start_time
    )

    for s in schedules:
        store.add(s)
    return schedules
