"""
AIOps Agent Platform - Runbook Version Store

每个 incident 的 Runbook 草案在发布时都会生成一份不可变快照，
便于审计、回滚与跨事故复用。

生命周期：
    draft  ──publish──▶  published  ──archive──▶  archived
                              │
                              └──publish again──▶  published (v+1)

当前实现是进程内 dict 存储（与本 v1.0 backend 一致）；
后续要持久化可以替换为 SQLite/PostgreSQL 表 + Alembic migration。
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunbookVersion:
    """Runbook 不可变版本快照。"""

    version_id: str
    incident_id: str
    version_number: int
    status: str  # draft / published / archived
    title: str
    root_cause: str
    confidence: float
    service: str
    markdown: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    change_note: str = ""
    published_by: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_summary(self) -> dict[str, Any]:
        """列表展示用：省略大字段（markdown/sections）。"""
        return {
            "version_id": self.version_id,
            "incident_id": self.incident_id,
            "version_number": self.version_number,
            "status": self.status,
            "title": self.title,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "service": self.service,
            "change_note": self.change_note,
            "published_by": self.published_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunbookVersionStore:
    """按 incident_id 维护版本列表的进程内存储。"""

    def __init__(self) -> None:
        self._versions_by_incident: dict[str, list[RunbookVersion]] = {}

    # ==================== Write ====================

    def publish(
        self,
        incident_id: str,
        draft: dict[str, Any],
        change_note: str = "",
        published_by: str | None = None,
    ) -> RunbookVersion:
        """从当前 draft 发布一份新版本；每次 publish 都会 +1。"""
        now = datetime.now(timezone.utc).isoformat()
        versions = self._versions_by_incident.setdefault(incident_id, [])

        # 找当前最高 version_number
        max_v = max((v.version_number for v in versions), default=0)
        new_v = max_v + 1

        # 之前 published 的版本保持 published（不自动覆盖）
        version = RunbookVersion(
            version_id=f"rbv-{uuid.uuid4().hex[:12]}",
            incident_id=incident_id,
            version_number=new_v,
            status="published",
            title=draft.get("title", ""),
            root_cause=draft.get("root_cause", ""),
            confidence=float(draft.get("confidence", 0.0)),
            service=draft.get("service", ""),
            markdown=draft.get("markdown", ""),
            sections=list(draft.get("sections", []) or []),
            change_note=change_note,
            published_by=published_by,
            created_at=now,
            updated_at=now,
        )
        versions.append(version)
        logger.info(
            "Runbook published",
            incident_id=incident_id,
            version=new_v,
            change_note=change_note or "(none)",
        )
        return version

    def archive(
        self,
        incident_id: str,
        version_number: int,
        change_note: str = "",
    ) -> RunbookVersion:
        """归档指定版本号。"""
        versions = self._versions_by_incident.get(incident_id, [])
        target = next(
            (v for v in versions if v.version_number == version_number), None
        )
        if target is None:
            raise KeyError(
                f"Version {version_number} not found for incident {incident_id}"
            )
        target.status = "archived"
        target.change_note = (
            f"{target.change_note}; archive: {change_note}"
            if target.change_note
            else f"archive: {change_note}"
        )
        target.updated_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Runbook version archived",
            incident_id=incident_id,
            version=version_number,
        )
        return target

    # ==================== Read ====================

    def list_versions(self, incident_id: str) -> list[dict[str, Any]]:
        """按 version_number 升序列出版本摘要。"""
        versions = self._versions_by_incident.get(incident_id, [])
        return [v.to_summary() for v in sorted(versions, key=lambda v: v.version_number)]

    def get_version(
        self, incident_id: str, version_number: int
    ) -> RunbookVersion | None:
        versions = self._versions_by_incident.get(incident_id, [])
        return next(
            (v for v in versions if v.version_number == version_number), None
        )

    def latest_published(self, incident_id: str) -> RunbookVersion | None:
        """返回最新一份 published 版本（不包含 archived）。"""
        versions = self._versions_by_incident.get(incident_id, [])
        published = [v for v in versions if v.status == "published"]
        if not published:
            return None
        return max(published, key=lambda v: v.version_number)


# 单例
runbook_version_store = RunbookVersionStore()
