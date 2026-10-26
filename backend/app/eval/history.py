"""评测历史 + baseline 管理。

设计目的：
- 每次 run_evaluation() 把结果存到 SQLite（自动记录 trend）
- baseline 文件存"业内标杆 / 历史基线"用于 CI 门禁和 A/B 对比
- CLI 提供 history / baseline / diff 三个子命令

V3 改造（E5）：
- 之前评测结果只 log，不存盘 → 改 prompt 前后没法对比
- 现在所有跑过的 eval 都进 SQLite，可以查 trend
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

# 默认历史库路径
_DEFAULT_HISTORY_DIR = Path(__file__).resolve().parents[3] / "data" / "eval_history"
_DEFAULT_DB_PATH = _DEFAULT_HISTORY_DIR / "history.db"
_DEFAULT_BASELINES_DIR = _DEFAULT_HISTORY_DIR / "baselines"

_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_history_db_path() -> Path:
    """获取历史库路径（可被 EVOLUTION_DB_PATH 类 envvar 覆盖）"""
    env = os.getenv("EVAL_HISTORY_DB_PATH")
    if env:
        return Path(env)
    return _DEFAULT_DB_PATH


def get_baselines_dir() -> Path:
    """获取 baseline 文件目录"""
    env = os.getenv("EVAL_BASELINES_DIR")
    if env:
        return Path(env)
    return _DEFAULT_BASELINES_DIR


@contextmanager
def _connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """打开 SQLite 连接，自动建表"""
    if db_path is None:
        db_path = get_history_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        with _LOCK:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    eval_id TEXT PRIMARY KEY,
                    run_at TEXT NOT NULL,
                    split TEXT NOT NULL,
                    model TEXT,
                    prompt_version TEXT,
                    analyzer TEXT,
                    total INTEGER NOT NULL,
                    passed INTEGER NOT NULL,
                    pass_rate REAL NOT NULL,
                    avg_confidence REAL,
                    by_dimension_json TEXT,
                    by_category_json TEXT,
                    failing_ids_json TEXT,
                    config_json TEXT,
                    note TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_eval_runs_run_at ON eval_runs(run_at DESC);
                CREATE INDEX IF NOT EXISTS idx_eval_runs_split ON eval_runs(split);
                """
            )
            conn.commit()
        yield conn
    finally:
        conn.close()


def record_eval_run(
    *,
    eval_id: str | None = None,
    split: str,
    model: str | None = None,
    prompt_version: str | None = None,
    analyzer: str | None = None,
    summary: Any,
    config: dict | None = None,
    note: str | None = None,
    db_path: Path | None = None,
) -> str:
    """记录一次评测结果到 SQLite。

    Args:
        eval_id: 自定义 ID（默认生成 uuid）
        split: 'dev' | 'test' | 'holdout' | 'all'
        model: LLM 模型名（如 'deepseek-chat'）
        prompt_version: 自进化 prompt 版本 ID
        analyzer: 'rules_engine' | 'llm_v1' 等
        summary: RCAEvaluationSummary 对象
        config: 任意额外配置（dict）
        note: 备注
        db_path: 覆盖路径（默认从 env 读）

    Returns:
        eval_id（字符串）
    """
    if eval_id is None:
        eval_id = f"eval-{uuid.uuid4().hex[:12]}"

    summary_dict = summary.to_dict()

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs (
                eval_id, run_at, split, model, prompt_version, analyzer,
                total, passed, pass_rate, avg_confidence,
                by_dimension_json, by_category_json, failing_ids_json,
                config_json, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                _now_iso(),
                split,
                model,
                prompt_version,
                analyzer,
                summary_dict["total"],
                summary_dict["passed"],
                summary_dict["pass_rate"],
                summary_dict.get("avg_confidence"),
                json.dumps(summary_dict.get("by_dimension", {}), ensure_ascii=False),
                json.dumps(summary_dict.get("by_category", {}), ensure_ascii=False),
                json.dumps(summary_dict.get("failing_ids", []), ensure_ascii=False),
                json.dumps(config or {}, ensure_ascii=False),
                note,
            ),
        )
        conn.commit()

    return eval_id


def list_runs(
    *,
    split: str | None = None,
    limit: int = 30,
    db_path: Path | None = None,
) -> list[dict]:
    """列出最近的评测运行"""
    query = "SELECT * FROM eval_runs"
    params: list = []
    if split:
        query += " WHERE split = ?"
        params.append(split)
    query += " ORDER BY run_at DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_trend(
    *,
    split: str,
    limit: int = 30,
    db_path: Path | None = None,
) -> list[dict]:
    """获取某 split 的 pass_rate 趋势（按时间顺序）"""
    rows = list_runs(split=split, limit=limit, db_path=db_path)
    return list(reversed(rows))


def get_latest_run(
    split: str | None = None, db_path: Path | None = None
) -> dict | None:
    """获取最近一次评测"""
    rows = list_runs(split=split, limit=1, db_path=db_path)
    return rows[0] if rows else None


# ============ Baseline 文件管理 ============

def save_baseline(
    name: str,
    *,
    pass_rate: float,
    total: int,
    passed: int,
    description: str = "",
    split: str = "test",
    source: str = "manual",
    notes: str = "",
    baselines_dir: Path | None = None,
) -> Path:
    """保存 baseline 到 JSON 文件。

    baseline 用于 CI 门禁和 A/B 对比，存放在 ./data/eval_history/baselines/{name}.json
    """
    if baselines_dir is None:
        baselines_dir = get_baselines_dir()
    baselines_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": name,
        "pass_rate": pass_rate,
        "total": total,
        "passed": passed,
        "split": split,
        "source": source,
        "description": description,
        "notes": notes,
        "created_at": _now_iso(),
    }
    path = baselines_dir / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_baseline(
    name: str, baselines_dir: Path | None = None
) -> dict | None:
    """加载 baseline，文件不存在返回 None"""
    if baselines_dir is None:
        baselines_dir = get_baselines_dir()
    path = baselines_dir / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_baselines(baselines_dir: Path | None = None) -> list[dict]:
    """列出所有 baseline"""
    if baselines_dir is None:
        baselines_dir = get_baselines_dir()
    if not baselines_dir.exists():
        return []
    out = []
    for p in sorted(baselines_dir.glob("*.json")):
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def check_against_baseline(
    summary: Any,
    baseline_name: str,
    *,
    min_pass_rate: float | None = None,
    baselines_dir: Path | None = None,
) -> dict:
    """拿当前 summary 对比 baseline，返回 {passed, baseline_pass_rate, current_pass_rate, delta, ...}"""
    baseline = load_baseline(baseline_name, baselines_dir)
    summary_dict = summary.to_dict()
    current_rate = summary_dict["pass_rate"]

    if baseline is None:
        return {
            "passed": True,
            "reason": f"baseline '{baseline_name}' 不存在，跳过门禁",
            "current_pass_rate": current_rate,
        }

    baseline_rate = baseline["pass_rate"]
    threshold = min_pass_rate if min_pass_rate is not None else baseline_rate
    delta = current_rate - baseline_rate
    passed = current_rate >= threshold

    return {
        "passed": passed,
        "reason": (
            f"current={current_rate:.3f} >= baseline={baseline_rate:.3f}"
            if passed else
            f"current={current_rate:.3f} < baseline={baseline_rate:.3f}"
        ),
        "current_pass_rate": current_rate,
        "baseline_pass_rate": baseline_rate,
        "delta": delta,
        "min_required": threshold,
        "baseline_name": baseline_name,
    }