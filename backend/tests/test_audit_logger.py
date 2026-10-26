"""
Audit 模块测试

覆盖：
1. AuditLogger.record / query / count
2. AuditLogger.export_csv / export_json
3. AuditLogger.cleanup_older_than
4. @audit 装饰器（同步 + 异步）
5. LLMClient 集成审计（成功/失败/熔断 OPEN 都记录）
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import tempfile
from pathlib import Path

import pytest

from app.audit import AuditAction, AuditEntry, AuditLogger, audit, get_audit_logger, reset_audit_logger
from app.audit.models import compute_hash
from app.config import LLMConfig
from app.core.circuit_breaker import CircuitBreaker
from app.core.llm_client import LLMClient


@pytest.fixture
def tmp_audit_db(tmp_path):
    """每个测试一个临时审计库"""
    db_path = tmp_path / "audit.db"
    reset_audit_logger()
    yield str(db_path)
    reset_audit_logger()


@pytest.fixture
def logger(tmp_audit_db) -> AuditLogger:
    return AuditLogger(tmp_audit_db)


class TestAuditLoggerRecord:
    def test_record_basic(self, logger: AuditLogger):
        e = logger.record(
            actor="system:rca_agent",
            action=AuditAction.RCA_COMPLETED,
            target_type="incident",
            target_id="inc-1",
            input_data={"alert": "high_cpu"},
            output_data={"root_cause": "resource_saturation", "confidence": 0.85},
            details={"latency_ms": 120},
        )
        assert e.id is not None
        assert e.actor == "system:rca_agent"
        # compute_hash 返回 64 字符 hex（不带前缀）
        assert len(e.input_hash) == 64
        assert all(c in "0123456789abcdef" for c in e.input_hash)

    def test_record_with_no_data(self, logger: AuditLogger):
        e = logger.record(
            actor="system:test",
            action="custom_action",
            target_type="test",
            target_id="t-1",
        )
        assert e.id is not None
        assert e.input_hash == ""

    def test_compute_hash_consistent(self):
        h1 = compute_hash({"a": 1, "b": 2})
        h2 = compute_hash({"b": 2, "a": 1})  # key 顺序不同
        assert h1 == h2  # sort_keys=True 应保证一致性

        h3 = compute_hash({"a": 1, "b": 3})
        assert h1 != h3


class TestAuditLoggerQuery:
    def test_query_all(self, logger: AuditLogger):
        for i in range(5):
            logger.record(
                actor="system:x", action="evt", target_type="t", target_id=f"id-{i}",
            )
        entries = logger.query()
        assert len(entries) == 5

    def test_query_by_actor(self, logger: AuditLogger):
        logger.record(actor="a1", action="x", target_type="t", target_id="1")
        logger.record(actor="a2", action="x", target_type="t", target_id="2")
        logger.record(actor="a1", action="x", target_type="t", target_id="3")
        entries = logger.query(actor="a1")
        assert len(entries) == 2
        assert all(e.actor == "a1" for e in entries)

    def test_query_by_action(self, logger: AuditLogger):
        logger.record(actor="s", action="llm_called", target_type="llm", target_id="m1")
        logger.record(actor="s", action="rca_completed", target_type="inc", target_id="i1")
        entries = logger.query(action="llm_called")
        assert len(entries) == 1
        assert entries[0].action == "llm_called"

    def test_query_by_target(self, logger: AuditLogger):
        logger.record(actor="s", action="x", target_type="incident", target_id="inc-1")
        logger.record(actor="s", action="x", target_type="incident", target_id="inc-2")
        entries = logger.query(target_id="inc-1")
        assert len(entries) == 1

    def test_query_pagination(self, logger: AuditLogger):
        for i in range(10):
            logger.record(actor="s", action="x", target_type="t", target_id=str(i))
        first_page = logger.query(limit=3, offset=0)
        second_page = logger.query(limit=3, offset=3)
        assert len(first_page) == 3
        assert len(second_page) == 3
        assert first_page[0].id != second_page[0].id

    def test_count(self, logger: AuditLogger):
        assert logger.count() == 0
        for _ in range(7):
            logger.record(actor="s", action="x", target_type="t", target_id="1")
        assert logger.count() == 7


class TestAuditLoggerExport:
    def test_export_csv(self, logger: AuditLogger):
        logger.record(
            actor="s", action="evt", target_type="inc", target_id="i1",
            input_data={"a": 1}, output_data={"b": 2},
        )
        csv_text = logger.export_csv()
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["actor"] == "s"
        assert rows[0]["action"] == "evt"

    def test_export_json(self, logger: AuditLogger):
        logger.record(actor="s", action="evt", target_type="t", target_id="1")
        json_text = logger.export_json()
        data = json.loads(json_text)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["actor"] == "s"

    def test_export_with_filter(self, logger: AuditLogger):
        logger.record(actor="a1", action="x", target_type="t", target_id="1")
        logger.record(actor="a2", action="x", target_type="t", target_id="2")
        csv_text = logger.export_csv(actor="a1")
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        assert len(rows) == 1
        assert rows[0]["actor"] == "a1"


class TestAuditLoggerCleanup:
    def test_cleanup_older_than(self, logger: AuditLogger):
        # 插一条"很老"的记录
        logger.record(actor="old", action="x", target_type="t", target_id="1")
        # 直接 SQL 改 timestamp 到 100 天前
        import sqlite3
        with sqlite3.connect(logger.db_path) as conn:
            conn.execute(
                "UPDATE audit_log SET timestamp = datetime('now', '-100 days')"
            )
            conn.commit()
        # 再插一条新的
        logger.record(actor="new", action="x", target_type="t", target_id="2")

        deleted = logger.cleanup_older_than(days=90)
        assert deleted == 1

        entries = logger.query()
        assert len(entries) == 1
        assert entries[0].actor == "new"


# =====================================================================
# @audit 装饰器测试
# =====================================================================

class TestAuditDecorator:
    def test_sync_function_success(self, logger: AuditLogger):
        # 替换全局 logger
        import app.audit.decorator as dec
        dec.get_audit_logger = lambda: logger

        @audit(action=AuditAction.LLM_CALLED, target_type="llm")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

        entries = logger.query(action=AuditAction.LLM_CALLED.value)
        assert len(entries) == 1
        assert entries[0].details["status"] == "success"

    def test_sync_function_failure(self, logger: AuditLogger):
        import app.audit.decorator as dec
        dec.get_audit_logger = lambda: logger

        @audit(action=AuditAction.RCA_COMPLETED, target_type="inc")
        def will_fail():
            raise ValueError("simulated")

        with pytest.raises(ValueError):
            will_fail()

        entries = logger.query()
        assert len(entries) == 1
        assert entries[0].details["status"] == "failed"
        assert entries[0].details["exception_type"] == "ValueError"

    @pytest.mark.asyncio
    async def test_async_function(self, logger: AuditLogger):
        import app.audit.decorator as dec
        dec.get_audit_logger = lambda: logger

        @audit(action=AuditAction.LLM_CALLED, target_type="llm", include_output=True)
        async def async_func(x):
            return x * 2

        result = await async_func(5)
        assert result == 10

        entries = logger.query()
        assert len(entries) == 1
        assert "result" in entries[0].details or entries[0].output_hash


# =====================================================================
# LLMClient 集成审计测试
# =====================================================================

class TestLLMClientAudit:
    @pytest.mark.asyncio
    async def test_chat_records_llm_circuit_open(self, logger: AuditLogger):
        import app.core.llm_client as lc
        lc.get_audit_logger = lambda: logger

        cfg = LLMConfig(provider="openai", api_key="sk-test12345678901234567890")
        breaker = CircuitBreaker("test", failure_threshold=1, timeout_seconds=60)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

        client = LLMClient(cfg, breaker=breaker)
        with pytest.raises(Exception):
            await client.chat([{"role": "user", "content": "hi"}], incident_id="inc-1")

        entries = logger.query(action=AuditAction.LLM_CIRCUIT_OPEN.value)
        assert len(entries) == 1
        assert entries[0].target_id == cfg.model  # 用实际配置的 model
        # incident_id 应该被记录到 input_data 的 hash 里或者 details 里
        details_str = str(entries[0].to_dict())
        assert "inc-1" in details_str

    @pytest.mark.asyncio
    async def test_chat_records_key_missing(self, logger: AuditLogger):
        import app.core.llm_client as lc
        lc.get_audit_logger = lambda: logger

        cfg = LLMConfig(provider="openai", api_key="")
        client = LLMClient(cfg)
        with pytest.raises(Exception):
            await client.chat([{"role": "user", "content": "hi"}])

        # key 缺失不写审计（早期拒绝，不需要审计）
        entries = logger.query()
        assert len(entries) == 0
