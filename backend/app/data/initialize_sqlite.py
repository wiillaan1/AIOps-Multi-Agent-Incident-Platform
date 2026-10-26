"""
backend/app/data/initialize_sqlite.py

初始化 data/aiops.db：建审计日志表、LLM 调用记录表、评测反馈表，
并建一个 audit_logs 视图，兼容 eval_tools.py 里对复数表名的查询。

运行方式（在 backend/ 目录下）：
    python -m app.data.initialize_sqlite
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.audit.logger import SCHEMA_SQL as AUDIT_SCHEMA_SQL
from app.core.llm_cost import SCHEMA_SQL as COST_SCHEMA_SQL

DB_PATH = Path("data/aiops.db")

EVALUATION_FEEDBACK_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_id VARCHAR(64) NOT NULL,
    feedback TEXT,
    rating FLOAT,
    metadata TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

# 兼容 eval_tools.py 里 `SELECT * FROM audit_logs`（复数）——
# 实际表名是 audit_log（单数），用视图做别名，不改动已有代码
COMPAT_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS audit_logs AS SELECT * FROM audit_log;
"""


def initialize_sqlite() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), isolation_level=None, timeout=10.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(AUDIT_SCHEMA_SQL)
        conn.executescript(COST_SCHEMA_SQL)
        conn.executescript(EVALUATION_FEEDBACK_SQL)
        conn.executescript(COMPAT_VIEW_SQL)
        conn.commit()
    finally:
        conn.close()

    print(f"SQLite initialized at {DB_PATH}")


if __name__ == "__main__":
    initialize_sqlite()