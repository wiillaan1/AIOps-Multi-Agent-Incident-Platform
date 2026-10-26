"""
PromptStore - Prompt 版本仓库

数据库路径：./data/evolution.db（与 audit / reviews / 业务库隔离）

表：
- prompt_versions: 每次 prompt 改动一行
- evolution_cycles: 每次自进化循环一行
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.evolution.models import EvolutionCycle, PromptVersion


logger = logging.getLogger(__name__)


DEFAULT_EVOLUTION_DB_PATH = os.getenv(
    "EVOLUTION_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "evolution.db"),
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    parent_version TEXT,
    prompt_segment TEXT NOT NULL,
    target_dimension TEXT,
    badcase_ids TEXT,
    cycle_id INTEGER,
    timestamp TEXT NOT NULL,
    ab_test_results TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_prompt_version ON prompt_versions(version);
CREATE INDEX IF NOT EXISTS idx_prompt_parent ON prompt_versions(parent_version);
CREATE INDEX IF NOT EXISTS idx_prompt_cycle ON prompt_versions(cycle_id);

CREATE TABLE IF NOT EXISTS evolution_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    badcase_ids TEXT,
    pattern_analysis TEXT,
    generated_prompt TEXT,
    ab_test_summary TEXT,
    promotion_decision TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT DEFAULT '',
    review_notes TEXT DEFAULT '',
    current_stage TEXT NOT NULL DEFAULT 'badcase_collected',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cycle_stage ON evolution_cycles(current_stage);
CREATE INDEX IF NOT EXISTS idx_cycle_decision ON evolution_cycles(promotion_decision);
"""


class PromptStore:
    """Prompt 版本仓库（线程安全）"""

    def __init__(self, db_path: str = DEFAULT_EVOLUTION_DB_PATH):
        self.db_path = db_path
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    # ---------- PromptVersion ----------
    def save_prompt_version(self, prompt: PromptVersion) -> PromptVersion:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO prompt_versions
                        (version, parent_version, prompt_segment, target_dimension,
                         badcase_ids, cycle_id, timestamp, ab_test_results)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prompt.version,
                        prompt.parent_version or None,
                        prompt.prompt_segment,
                        prompt.target_dimension,
                        json.dumps(prompt.badcase_ids, ensure_ascii=False),
                        prompt.cycle_id,
                        prompt.timestamp,
                        json.dumps(prompt.ab_test_results, ensure_ascii=False, default=str),
                    ),
                )
                prompt.id = cursor.lastrowid
        logger.info(
            "[evolution] saved prompt version=%s id=%s cycle=%s",
            prompt.version, prompt.id, prompt.cycle_id,
        )
        return prompt

    def get_latest_prompt(self) -> PromptVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return self._row_to_prompt(row) if row else None

    def get_prompt_by_version(self, version: str) -> PromptVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_versions WHERE version = ?", (version,)
            ).fetchone()
        return self._row_to_prompt(row) if row else None

    # ---------- EvolutionCycle ----------
    def save_cycle(self, cycle: EvolutionCycle) -> EvolutionCycle:
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO evolution_cycles
                        (started_at, completed_at, badcase_ids, pattern_analysis,
                         generated_prompt, ab_test_summary, promotion_decision,
                         reviewer, review_notes, current_stage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cycle.started_at,
                        cycle.completed_at,
                        json.dumps(cycle.badcase_ids, ensure_ascii=False),
                        json.dumps(cycle.pattern_analysis, ensure_ascii=False, default=str),
                        cycle.generated_prompt,
                        json.dumps(cycle.ab_test_summary, ensure_ascii=False, default=str),
                        cycle.promotion_decision,
                        cycle.reviewer,
                        cycle.review_notes,
                        cycle.current_stage,
                    ),
                )
                cycle.id = cursor.lastrowid
        logger.info("[evolution] saved cycle id=%s stage=%s", cycle.id, cycle.current_stage)
        return cycle

    def update_cycle(self, cycle: EvolutionCycle) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE evolution_cycles SET
                        completed_at = ?, pattern_analysis = ?, generated_prompt = ?,
                        ab_test_summary = ?, promotion_decision = ?, reviewer = ?,
                        review_notes = ?, current_stage = ?
                    WHERE id = ?
                    """,
                    (
                        cycle.completed_at,
                        json.dumps(cycle.pattern_analysis, ensure_ascii=False, default=str),
                        cycle.generated_prompt,
                        json.dumps(cycle.ab_test_summary, ensure_ascii=False, default=str),
                        cycle.promotion_decision,
                        cycle.reviewer,
                        cycle.review_notes,
                        cycle.current_stage,
                        cycle.id,
                    ),
                )
        logger.info("[evolution] updated cycle id=%s stage=%s", cycle.id, cycle.current_stage)

    def get_cycle(self, cycle_id: int) -> EvolutionCycle | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM evolution_cycles WHERE id = ?", (cycle_id,)
            ).fetchone()
        return self._row_to_cycle(row) if row else None

    def list_cycles(self, limit: int = 20, decision: str | None = None) -> list[EvolutionCycle]:
        sql = "SELECT * FROM evolution_cycles"
        params: tuple = ()
        if decision:
            sql += " WHERE promotion_decision = ?"
            params = (decision,)
        sql += " ORDER BY id DESC LIMIT ?"
        params = (limit,) if not decision else (decision, limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_cycle(r) for r in rows]

    # ---------- 辅助 ----------
    @staticmethod
    def _row_to_prompt(row: tuple) -> PromptVersion:
        badcase_raw = row[5]
        try:
            badcase_ids = json.loads(badcase_raw) if badcase_raw else []
        except (json.JSONDecodeError, TypeError):
            badcase_ids = []
        ab_raw = row[8]
        try:
            ab_results = json.loads(ab_raw) if ab_raw else {}
        except (json.JSONDecodeError, TypeError):
            ab_results = {}
        return PromptVersion(
            id=row[0],
            version=row[1],
            parent_version=row[2] or "",
            prompt_segment=row[3],
            target_dimension=row[4] or "",
            badcase_ids=badcase_ids,
            cycle_id=row[6],
            timestamp=row[7],
            ab_test_results=ab_results,
        )

    @staticmethod
    def _row_to_cycle(row: tuple) -> EvolutionCycle:
        def _loads(raw: str | None) -> Any:
            try:
                return json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                return {}

        return EvolutionCycle(
            id=row[0],
            started_at=row[1],
            completed_at=row[2],
            badcase_ids=_loads(row[3]) if isinstance(_loads(row[3]), list) else [],
            pattern_analysis=_loads(row[4]),
            generated_prompt=row[5] or "",
            ab_test_summary=_loads(row[6]),
            promotion_decision=row[7],
            reviewer=row[8] or "",
            review_notes=row[9] or "",
            current_stage=row[10],
        )


# ---------- 全局单例 ----------
_store: PromptStore | None = None
_store_lock = threading.Lock()


def get_prompt_store(db_path: str | None = None) -> PromptStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = PromptStore(db_path or DEFAULT_EVOLUTION_DB_PATH)
    return _store


def reset_prompt_store() -> None:
    global _store
    _store = None
