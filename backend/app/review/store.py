"""
ReviewStore - 人评数据持久化（SQLite）

数据库路径：./data/reviews.db（与 audit.db / 业务库隔离）
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import statistics
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.review.models import Review, ReviewDimension


logger = logging.getLogger(__name__)


DEFAULT_REVIEW_DB_PATH = os.getenv(
    "REVIEW_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "reviews.db"),
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS human_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    incident_id TEXT NOT NULL,
    badcase_id TEXT,
    reviewer TEXT NOT NULL,
    decision TEXT NOT NULL,
    scores TEXT NOT NULL,
    comments TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_review_incident ON human_reviews(incident_id);
CREATE INDEX IF NOT EXISTS idx_review_reviewer ON human_reviews(reviewer);
CREATE INDEX IF NOT EXISTS idx_review_decision ON human_reviews(decision);
CREATE INDEX IF NOT EXISTS idx_review_timestamp ON human_reviews(timestamp);
"""


class ReviewStore:
    """人评 SQLite 存储（线程安全）"""

    def __init__(self, db_path: str = DEFAULT_REVIEW_DB_PATH):
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

    # ---------- CRUD ----------
    def submit(self, review: Review) -> Review:
        """提交一条人评（写完后 review.id 自动填充）"""
        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO human_reviews
                        (timestamp, incident_id, badcase_id, reviewer, decision, scores, comments)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review.timestamp,
                        review.incident_id,
                        review.badcase_id or None,
                        review.reviewer,
                        review.decision,
                        json.dumps(review.scores, ensure_ascii=False),
                        review.comments,
                    ),
                )
                review.id = cursor.lastrowid
        logger.info(
            "[review] id=%s incident=%s reviewer=%s decision=%s avg=%.2f",
            review.id, review.incident_id, review.reviewer,
            review.decision, review.avg_score(),
        )
        return review

    def get(self, review_id: int) -> Review | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM human_reviews WHERE id = ?", (review_id,)
            ).fetchone()
        return self._row_to_review(row) if row else None

    def list_recent(self, limit: int = 50) -> list[Review]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM human_reviews ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_review(r) for r in rows]

    def list_pending_incidents(self, all_incident_ids: list[str]) -> list[str]:
        """返回待评分的 incident_id（在 all_incident_ids 中但没评分过的）"""
        if not all_incident_ids:
            return []
        placeholders = ",".join("?" * len(all_incident_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT incident_id FROM human_reviews WHERE incident_id IN ({placeholders})",
                all_incident_ids,
            ).fetchall()
        reviewed = {r[0] for r in rows}
        return [iid for iid in all_incident_ids if iid not in reviewed]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM human_reviews").fetchone()[0]

    # ---------- 统计 ----------
    def stats(self, since: str | None = None) -> dict[str, Any]:
        """评分汇总统计"""
        sql = "SELECT reviewer, decision, scores FROM human_reviews"
        params: tuple = ()
        if since:
            sql += " WHERE timestamp >= ?"
            params = (since,)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        if not rows:
            return {
                "total": 0,
                "by_decision": {},
                "by_reviewer": {},
                "avg_overall": 0.0,
                "pass_rate": 0.0,
                "since": since,
            }

        by_decision: Counter = Counter()
        by_reviewer: Counter = Counter()
        overall_scores: list[float] = []

        for reviewer, decision, scores_raw in rows:
            by_decision[decision] += 1
            by_reviewer[reviewer] += 1
            try:
                scores = json.loads(scores_raw) if scores_raw else {}
            except (json.JSONDecodeError, TypeError):
                scores = {}
            overall = scores.get(ReviewDimension.OVERALL.value)
            if isinstance(overall, (int, float)):
                overall_scores.append(float(overall))

        accepted = by_decision.get("accept", 0)
        rejected = by_decision.get("reject", 0)
        pass_rate = accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0.0

        return {
            "total": len(rows),
            "by_decision": dict(by_decision),
            "by_reviewer": dict(by_reviewer),
            "avg_overall": round(statistics.mean(overall_scores), 2) if overall_scores else 0.0,
            "pass_rate": round(pass_rate, 3),
            "since": since,
        }

    # ---------- 辅助 ----------
    @staticmethod
    def _row_to_review(row: tuple) -> Review:
        # 列顺序: id, timestamp, incident_id, badcase_id, reviewer, decision, scores, comments, created_at
        scores_raw = row[6]
        try:
            scores = json.loads(scores_raw) if scores_raw else {}
        except (json.JSONDecodeError, TypeError):
            scores = {"_raw": scores_raw}
        return Review(
            id=row[0],
            timestamp=row[1],
            incident_id=row[2],
            badcase_id=row[3] or "",
            reviewer=row[4],
            decision=row[5],
            scores=scores,
            comments=row[7] or "",
        )


# ---------- 全局单例 ----------
_store: ReviewStore | None = None
_store_lock = threading.Lock()


def get_review_store(db_path: str | None = None) -> ReviewStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = ReviewStore(db_path or DEFAULT_REVIEW_DB_PATH)
    return _store


def reset_review_store() -> None:
    global _store
    _store = None
