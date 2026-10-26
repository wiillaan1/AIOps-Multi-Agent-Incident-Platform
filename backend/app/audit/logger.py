"""
AuditLogger - 持久化审计日志（append-only）

数据库表：
    audit_log(
        id, timestamp, actor, action, target_type, target_id,
        input_hash, output_hash, details, prev_hash
    )

设计：
- 单独 SQLite 数据库文件（./data/audit.db），不与业务库混用
- append-only：只提供 INSERT / SELECT，不暴露 UPDATE / DELETE
- 线程安全 + 进程安全（每条记录独立事务）
- 90 天保留策略（生产可配置）

接口：
    logger = get_audit_logger()
    logger.record(actor=..., action=..., target_type=..., target_id=..., input_data=..., output_data=..., details=...)
    entries = logger.query(actor="system:rca_agent", since="2026-01-01")
    csv_text = logger.export_csv(since=..., until=...)
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app.audit.models import AuditEntry, compute_hash


logger = logging.getLogger(__name__)


DEFAULT_AUDIT_DB_PATH = os.getenv(
    "AUDIT_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "audit.db"),
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    input_hash TEXT,
    output_hash TEXT,
    details TEXT,
    prev_hash TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_target ON audit_log(target_type, target_id);
"""


class AuditLogger:
    """持久化审计日志（append-only）"""

    def __init__(self, db_path: str = DEFAULT_AUDIT_DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """初始化表结构"""
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ---------- 写入 ----------
    def record(
        self,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        input_data: Any = None,
        output_data: Any = None,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """
        记录一条审计日志（append-only）。

        参数：
            actor: 操作主体（"system:rca_agent" / "user:zhangsan" / "tool:xxx"）
            action: AuditAction 值（如 "llm_called"）
            target_type: 目标类型（"incident" / "change" / "prompt" / "agent" / "llm"）
            target_id: 目标 ID
            input_data: 输入数据（任意可序列化对象，会计算 hash）
            output_data: 输出数据（同上）
            details: 附加元数据

        返回：
            AuditEntry（包含数据库自增 ID）
        """
        entry = AuditEntry(
            actor=actor,
            action=action,
            target_type=target_type,
            target_id=target_id,
            input_hash=compute_hash(input_data) if input_data is not None else "",
            output_hash=compute_hash(output_data) if output_data is not None else "",
            details=details or {},
        )

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO audit_log
                        (timestamp, actor, action, target_type, target_id,
                         input_hash, output_hash, details, prev_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.timestamp,
                        entry.actor,
                        entry.action,
                        entry.target_type,
                        entry.target_id,
                        entry.input_hash,
                        entry.output_hash,
                        json.dumps(entry.details, ensure_ascii=False, default=str),
                        entry.prev_hash,
                    ),
                )
                entry.id = cursor.lastrowid

        logger.debug(
            "[audit] recorded id=%s action=%s target=%s/%s actor=%s",
            entry.id, entry.action, entry.target_type, entry.target_id, entry.actor,
        )
        return entry

    # ---------- 查询 ----------
    def query(
        self,
        actor: str | None = None,
        action: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        since: str | None = None,  # ISO 8601
        until: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """
        查询审计日志。

        所有过滤条件都是可选的 AND 组合。
        """
        sql = "SELECT * FROM audit_log WHERE 1=1"
        params: list[Any] = []

        if actor:
            sql += " AND actor = ?"
            params.append(actor)
        if action:
            sql += " AND action = ?"
            params.append(action)
        if target_type:
            sql += " AND target_type = ?"
            params.append(target_type)
        if target_id:
            sql += " AND target_id = ?"
            params.append(target_id)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        if until:
            sql += " AND timestamp <= ?"
            params.append(until)

        sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_entry(r) for r in rows]

    def count(self, since: str | None = None) -> int:
        """统计日志条数（用于可观测性）"""
        with self._connect() as conn:
            if since:
                return conn.execute(
                    "SELECT COUNT(*) FROM audit_log WHERE timestamp >= ?", (since,)
                ).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    @staticmethod
    def _row_to_entry(row: tuple) -> AuditEntry:
        details_raw = row[8]
        try:
            details = json.loads(details_raw) if details_raw else {}
        except (json.JSONDecodeError, TypeError):
            details = {"_raw": details_raw}
        return AuditEntry(
            id=row[0],
            timestamp=row[1],
            actor=row[2],
            action=row[3],
            target_type=row[4],
            target_id=row[5],
            input_hash=row[6] or "",
            output_hash=row[7] or "",
            details=details,
            prev_hash=row[9],
        )

    # ---------- 导出 ----------
    def export_csv(
        self,
        since: str | None = None,
        until: str | None = None,
        actor: str | None = None,
        action: str | None = None,
    ) -> str:
        """导出为 CSV 字符串（合规审计用）"""
        entries = self.query(
            actor=actor, action=action, since=since, until=until, limit=100_000
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "timestamp", "actor", "action",
            "target_type", "target_id", "input_hash", "output_hash", "details",
        ])
        for e in entries:
            writer.writerow([
                e.id, e.timestamp, e.actor, e.action,
                e.target_type, e.target_id, e.input_hash, e.output_hash,
                json.dumps(e.details, ensure_ascii=False),
            ])
        return buf.getvalue()

    def export_json(
        self,
        since: str | None = None,
        until: str | None = None,
        actor: str | None = None,
        action: str | None = None,
    ) -> str:
        """导出为 JSON 字符串"""
        entries = self.query(
            actor=actor, action=action, since=since, until=until, limit=100_000
        )
        return json.dumps(
            [e.to_dict() for e in entries],
            ensure_ascii=False, indent=2,
        )

    # ---------- 维护 ----------
    def cleanup_older_than(self, days: int = 90) -> int:
        """
        删除超过 N 天的审计日志（仅清理接口，不暴露通用 DELETE）。

        生产环境应该按合规要求保留，本接口默认 90 天。
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_log WHERE timestamp < ?", (cutoff,)
                )
                deleted = cursor.rowcount
                conn.commit()
                logger.info("[audit] cleanup removed %s entries older than %s days", deleted, days)
                return deleted


# ---------- 全局单例 ----------
_logger: AuditLogger | None = None
_logger_lock = threading.Lock()


def get_audit_logger(db_path: str | None = None) -> AuditLogger:
    """获取全局审计日志实例"""
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = AuditLogger(db_path or DEFAULT_AUDIT_DB_PATH)
    return _logger


def reset_audit_logger() -> None:
    """重置全局实例（用于测试）"""
    global _logger
    _logger = None
