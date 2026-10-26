"""
LLM 使用与成本核算（审查报告七、中期补 #7）

为什么需要这个模块：
1. LLMCallResult.token_in / token_out 字段已存在，但 `_sync_chat` 一直没把
   response.usage_metadata 透传出来 —— token 永远是 0。
2. 真实的 LLM E2E 跑下来没法看每次 incident 花了多少钱、占多少 token。
3. 学生讲项目时如果只说"接入了 LLM"，面试官一问"每次 incident 多少钱"就破。

设计：
- LLMUsageRecord：单次调用的不可变记录（token + cost + 元数据）
- LLMUsageRecorder：append-only SQLite 写入，复用 audit.db 同一文件（多张表）
- cost_usd = (token_in/1000) * input_cost_per_1k + (token_out/1000) * output_cost_per_1k
- 如果 input_cost_per_1k == output_cost_per_1k == 0，cost_usd 记 0，但 token 仍然记录

数据库表：
    aiops_model_invocations(
        id, timestamp, incident_id, model, provider,
        token_in, token_out, cost_usd, currency,
        latency_ms, via_fallback, caller_agent,
        prompt_preview, response_preview, details
    )
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)


# 默认复用 audit.db 文件，多张表共存
DEFAULT_DB_PATH = os.getenv(
    "LLM_COST_DB_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "audit.db"),
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS aiops_model_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    incident_id TEXT,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    token_in INTEGER NOT NULL DEFAULT 0,
    token_out INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    currency TEXT NOT NULL DEFAULT 'USD',
    latency_ms INTEGER NOT NULL DEFAULT 0,
    via_fallback INTEGER NOT NULL DEFAULT 0,
    caller_agent TEXT,
    prompt_preview TEXT,
    response_preview TEXT,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_ami_incident ON aiops_model_invocations(incident_id);
CREATE INDEX IF NOT EXISTS idx_ami_timestamp ON aiops_model_invocations(timestamp);
CREATE INDEX IF NOT EXISTS idx_ami_provider ON aiops_model_invocations(provider, model);
CREATE INDEX IF NOT EXISTS idx_ami_caller ON aiops_model_invocations(caller_agent);
"""


@dataclass(frozen=True)
class LLMUsageRecord:
    """单次 LLM 调用的不可变记录"""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    incident_id: str | None = None
    model: str = ""
    provider: str = ""
    token_in: int = 0
    token_out: int = 0
    cost_usd: float = 0.0
    currency: str = "USD"
    latency_ms: int = 0
    via_fallback: bool = False
    caller_agent: str | None = None
    prompt_preview: str | None = None
    response_preview: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compute_cost_usd(
    token_in: int,
    token_out: int,
    input_cost_per_1k: float,
    output_cost_per_1k: float,
) -> float:
    """
    计算单次调用的 USD 成本。

    公式：(token_in / 1000) * input_cost_per_1k + (token_out / 1000) * output_cost_per_1k

    注意：
    - 输入必须是 int（token 数），不能传 float（小数会被丢弃）
    - 当 input_cost_per_1k 和 output_cost_per_1k 都为 0 时返回 0.0
      （意味着用户没配置单价，但 token 数仍然会被记录）
    - 返回值保留 6 位小数（足够微支付场景），避免浮点精度问题
    """
    if input_cost_per_1k < 0 or output_cost_per_1k < 0:
        raise ValueError(
            f"cost_per_1k 必须非负，得到 input={input_cost_per_1k}, output={output_cost_per_1k}"
        )
    cost = (token_in / 1000.0) * input_cost_per_1k + (token_out / 1000.0) * output_cost_per_1k
    return round(cost, 6)


def extract_usage_metadata(response: Any) -> dict[str, int]:
    """
    从 langchain AIMessage 响应里安全地提取 usage_metadata。

    LangChain 在不同版本下字段名不同：
    - 新版（>=0.1）：response.usage_metadata = {"input_tokens": N, "output_tokens": M, "total_tokens": T}
    - 旧版：response.response_metadata.token_usage / response_metadata.usage
    - ChatOpenAI 新版：response.usage_metadata 优先

    返回：
        {"token_in": int, "token_out": int} —— 找不到时全 0
    """
    if response is None:
        return {"token_in": 0, "token_out": 0}

    # 1. 新版 usage_metadata（langchain >=0.1.0）
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        in_t = usage.get("input_tokens", 0) or 0
        out_t = usage.get("output_tokens", 0) or 0
        if in_t or out_t:
            return {"token_in": int(in_t), "token_out": int(out_t)}

    # 2. response_metadata.token_usage（旧版 openai）
    resp_meta = getattr(response, "response_metadata", None)
    if isinstance(resp_meta, dict):
        tu = resp_meta.get("token_usage") or resp_meta.get("usage")
        if isinstance(tu, dict):
            in_t = tu.get("prompt_tokens", 0) or 0
            out_t = tu.get("completion_tokens", 0) or 0
            if in_t or out_t:
                return {"token_in": int(in_t), "token_out": int(out_t)}

    # 3. 顶层 dict 形态（部分 provider）
    if isinstance(response, dict):
        in_t = response.get("token_in", 0) or 0
        out_t = response.get("token_out", 0) or 0
        return {"token_in": int(in_t), "token_out": int(out_t)}

    return {"token_in": 0, "token_out": 0}


class LLMUsageRecorder:
    """LLM 使用量 + 成本持久化（append-only，与 audit_log 同库多表）"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
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

    def record(self, usage: LLMUsageRecord) -> int | None:
        """
        写入一条 LLMUsageRecord，返回自增 id（写入失败返回 None）。

        设计：与 audit_log 一样 fire-and-forget，写入失败仅 warning 不抛异常，
        避免成本记录故障影响主链路。
        """
        try:
            with self._lock:
                with self._connect() as conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO aiops_model_invocations (
                            timestamp, incident_id, model, provider,
                            token_in, token_out, cost_usd, currency,
                            latency_ms, via_fallback, caller_agent,
                            prompt_preview, response_preview, details
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            usage.timestamp,
                            usage.incident_id,
                            usage.model,
                            usage.provider,
                            usage.token_in,
                            usage.token_out,
                            usage.cost_usd,
                            usage.currency,
                            usage.latency_ms,
                            1 if usage.via_fallback else 0,
                            usage.caller_agent,
                            usage.prompt_preview,
                            usage.response_preview,
                            json.dumps(usage.details or {}, ensure_ascii=False, default=str),
                        ),
                    )
                    return cursor.lastrowid
        except Exception as e:
            # fire-and-forget：成本记录失败不阻断主链路
            logger.warning(f"[LLMUsageRecorder] 写入失败（不影响主链路）: {e}")
            return None

    def aggregate(
        self,
        since: str | None = None,
        until: str | None = None,
        incident_id: str | None = None,
    ) -> dict[str, Any]:
        """
        按 incident_id / 时间窗聚合 cost + token 用量。

        返回：
            {
                "total_calls": int,
                "total_token_in": int,
                "total_token_out": int,
                "total_cost_usd": float,
                "currency": str,
                "by_model": [{"model": str, "calls": int, "token_in": int, "token_out": int, "cost_usd": float}, ...]
            }
        """
        clauses = []
        params: list[Any] = []
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if incident_id:
            clauses.append("incident_id = ?")
            params.append(incident_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_calls,
                    COALESCE(SUM(token_in), 0) AS total_token_in,
                    COALESCE(SUM(token_out), 0) AS total_token_out,
                    COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd,
                    COALESCE(MIN(currency), 'USD') AS currency
                FROM aiops_model_invocations
                {where}
                """,
                params,
            ).fetchone()

            by_model_rows = conn.execute(
                f"""
                SELECT model, provider,
                       COUNT(*) AS calls,
                       COALESCE(SUM(token_in), 0) AS token_in,
                       COALESCE(SUM(token_out), 0) AS token_out,
                       COALESCE(SUM(cost_usd), 0.0) AS cost_usd
                FROM aiops_model_invocations
                {where}
                GROUP BY model, provider
                ORDER BY cost_usd DESC
                """,
                params,
            ).fetchall()

        return {
            "total_calls": row[0] or 0,
            "total_token_in": row[1] or 0,
            "total_token_out": row[2] or 0,
            "total_cost_usd": round(row[3] or 0.0, 6),
            "currency": row[4] or "USD",
            "by_model": [
                {
                    "model": r[0],
                    "provider": r[1],
                    "calls": r[2],
                    "token_in": r[3],
                    "token_out": r[4],
                    "cost_usd": round(r[5] or 0.0, 6),
                }
                for r in by_model_rows
            ],
        }


# ---------- 全局单例 ----------
_recorder: LLMUsageRecorder | None = None


def get_llm_usage_recorder(db_path: str | None = None) -> LLMUsageRecorder:
    """获取全局 LLMUsageRecorder（懒加载）"""
    global _recorder
    if _recorder is None:
        _recorder = LLMUsageRecorder(db_path or DEFAULT_DB_PATH)
    return _recorder


def reset_llm_usage_recorder() -> None:
    """重置（用于测试或配置热更新）"""
    global _recorder
    _recorder = None
