"""
测试 Harness promote() → LTM + KB pattern 抽取（审查报告七、长期演 #14）

覆盖：
1. RuntimePatternStore add/query 持久化 + 召回
2. HarnessLoopEngine.promote() 注入了 ltm 后自动 commit lesson
3. HarnessLoopEngine.promote() 注入了 pattern_store 后自动抽取 pattern
4. HarnessLoopEngine.promote() 不注入时保持原行为（向后兼容 1131 测试）
5. LTM commit 失败 / Pattern 抽取失败 不阻断 promote 主链路
6. 多个 promote 累积，query_by_class 能按 promoted_at DESC 召回
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.badcase import BadCaseEntry
from app.models.harness import HarnessRunRecord, PatchCandidate
from app.services.badcase_registry_service import BadCaseRegistryService
from app.services.harness_engine import HarnessLoopEngine
from app.services.runtime_pattern_store import (
    RuntimePattern,
    RuntimePatternStore,
    get_runtime_pattern_store,
    reset_runtime_pattern_store,
)


# ============================================================
# 1. RuntimePatternStore 持久化 + 召回
# ============================================================

@pytest.fixture
def temp_pattern_store():
    """临时 JSONL 文件 fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "patterns.jsonl")
        store = RuntimePatternStore(store_path=path)
        yield store, path


class TestRuntimePatternStore:
    def test_add_returns_id(self, temp_pattern_store):
        store, _ = temp_pattern_store
        p = RuntimePattern(badcase_class="cpu_high", pattern_type="cpu_optimization")
        pid = store.add(p)
        assert pid == p.id

    def test_add_persists_to_jsonl(self, temp_pattern_store):
        store, path = temp_pattern_store
        store.add(RuntimePattern(
            badcase_class="memory_oom_high",
            pattern_type="memory_optimization",
            fix_summary="扩容 + 限流",
        ))
        # 直接读文件验证
        lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["badcase_class"] == "memory_oom_high"
        assert d["pattern_type"] == "memory_optimization"
        assert d["fix_summary"] == "扩容 + 限流"

    def test_query_by_class_returns_matching(self, temp_pattern_store):
        store, _ = temp_pattern_store
        store.add(RuntimePattern(badcase_class="cpu_high", pattern_type="cpu_opt"))
        store.add(RuntimePattern(badcase_class="memory_oom", pattern_type="memory_opt"))
        store.add(RuntimePattern(badcase_class="cpu_high", pattern_type="cpu_opt_v2"))
        results = store.query_by_class("cpu_high")
        assert len(results) == 2
        # 按 promoted_at DESC：v2 在前
        assert results[0].pattern_type == "cpu_opt_v2"

    def test_query_recent_respects_top_k(self, temp_pattern_store):
        store, _ = temp_pattern_store
        for i in range(5):
            store.add(RuntimePattern(badcase_class=f"cls-{i}", pattern_type="x"))
        results = store.query_recent(top_k=3)
        assert len(results) == 3

    def test_count(self, temp_pattern_store):
        store, _ = temp_pattern_store
        assert store.count() == 0
        store.add(RuntimePattern(badcase_class="a"))
        store.add(RuntimePattern(badcase_class="b"))
        assert store.count() == 2

    def test_handles_corrupt_jsonl_lines(self, temp_pattern_store):
        store, path = temp_pattern_store
        # 写入坏行
        Path(path).write_text('{"valid": true}\n{this is not json}\n{"also_valid": true}\n', encoding="utf-8")
        # 应该跳过坏行，不抛异常
        items = store._read_all()
        assert len(items) == 2

    def test_add_failure_does_not_raise(self, temp_pattern_store, monkeypatch):
        store, _ = temp_pattern_store

        def _broken_open(self, *args, **kwargs):
            raise OSError("disk full")

        # Patch Path.open（Path.open 走 io.open，不走 builtins.open）
        monkeypatch.setattr(Path, "open", _broken_open)
        # 不应抛异常
        pid = store.add(RuntimePattern(badcase_class="x"))
        assert pid == ""


# ============================================================
# 2-3. HarnessLoopEngine.promote() 注入 LTM + PatternStore
# ============================================================

def _make_fake_registry():
    """构造一个最小可用的 fake BadCaseRegistryService"""
    registry = MagicMock(spec=BadCaseRegistryService)
    return registry


def _make_fake_ltm():
    """构造一个 fake LongTermMemory（mock .store 方法）"""
    ltm = MagicMock()
    ltm.store = AsyncMock(return_value=None)
    return ltm


def _make_candidate(badcase_class: str = "memory_oom_high") -> PatchCandidate:
    return PatchCandidate(
        patch_id="patch-001",
        badcase_class=badcase_class,
        diff_or_patch="- cpu.limit=80% + cpu.limit=90%",
        rationale="提高 CPU 限制 + 增加 LRU cache size",
        proposed_by="auto_miner_v1",
    )


def _make_run_record(
    badcase_class: str = "memory_oom_high",
    decision: str = "promote",
) -> HarnessRunRecord:
    """构造一条已 promote 的 HarnessRunRecord"""
    from datetime import datetime, timezone
    return HarnessRunRecord(
        run_id="hr-test12345678",
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        badcase_class=badcase_class,
        candidate=_make_candidate(badcase_class),
        base_champion_version="v0.0.0",
        dev_metrics={"precision": 0.9, "recall": 0.85, "f1": 0.87},
        test_metrics={"precision": 0.85, "recall": 0.80, "f1": 0.82},
        decision=decision,
        promoted_at=datetime.now(timezone.utc),
    )


class TestHarnessPromoteToLTM:
    def test_promote_commits_lesson_to_ltm(self, tmp_path):
        registry = _make_fake_registry()
        ltm = _make_fake_ltm()
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            ltm=ltm,
        )
        run = _make_run_record()
        engine.promote(run)

        # 验证 ltm.store 被调用，参数是 lesson 类型
        assert ltm.store.await_count == 1
        lesson = ltm.store.await_args.args[0]
        # PROCEDURAL 类型（程序记忆：操作步骤）
        assert lesson.memory_type.value == "procedural"
        assert lesson.memory_level.value == "long_term"
        # verification_status 直接 VERIFIED（不走晋升）
        assert lesson.verification_status.value == "verified"
        assert lesson.verification_count == 3
        # 至少 2 个不同 verifier（"≥3 次 + ≥2 verifier" 的晋升条件）
        assert len(lesson.verified_by) >= 2
        assert "harness_loop_engine" in lesson.verified_by
        # 内容带 badcase_class
        assert "memory_oom_high" in lesson.content

    def test_promote_without_ltm_does_not_break(self, tmp_path):
        """向后兼容：ltm=None 时 promote() 行为不变（仅写 champion.json）"""
        registry = _make_fake_registry()
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            # ltm 不传
        )
        run = _make_run_record()
        # 不应抛异常
        engine.promote(run)
        # champion.json 仍正常写
        assert (tmp_path / "champion.json").exists()
        data = json.loads((tmp_path / "champion.json").read_text())
        assert "memory_oom_high" in data


class TestHarnessPromoteToPatternStore:
    def test_promote_extracts_pattern_to_kb(self, tmp_path):
        registry = _make_fake_registry()
        pattern_store = RuntimePatternStore(store_path=str(tmp_path / "patterns.jsonl"))
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            pattern_store=pattern_store,
        )
        run = _make_run_record(badcase_class="memory_oom_high")
        engine.promote(run)

        # 验证 pattern 写入
        results = pattern_store.query_by_class("memory_oom_high")
        assert len(results) == 1
        p = results[0]
        assert p.pattern_type == "memory_optimization"  # memory_* → memory_optimization
        assert p.source_run_id == "hr-test12345678"
        assert "提高 CPU 限制" in p.fix_summary

    def test_pattern_type_inference_cpu(self, tmp_path):
        registry = _make_fake_registry()
        pattern_store = RuntimePatternStore(store_path=str(tmp_path / "patterns.jsonl"))
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            pattern_store=pattern_store,
        )
        run = _make_run_record(badcase_class="cpu_spike")
        engine.promote(run)
        results = pattern_store.query_by_class("cpu_spike")
        assert results[0].pattern_type == "cpu_optimization"

    def test_pattern_type_inference_network(self, tmp_path):
        registry = _make_fake_registry()
        pattern_store = RuntimePatternStore(store_path=str(tmp_path / "patterns.jsonl"))
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            pattern_store=pattern_store,
        )
        run = _make_run_record(badcase_class="db_latency_high")
        engine.promote(run)
        results = pattern_store.query_by_class("db_latency_high")
        assert results[0].pattern_type == "network_optimization"

    def test_pattern_type_inference_other(self, tmp_path):
        registry = _make_fake_registry()
        pattern_store = RuntimePatternStore(store_path=str(tmp_path / "patterns.jsonl"))
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            pattern_store=pattern_store,
        )
        run = _make_run_record(badcase_class="config_drift")
        engine.promote(run)
        results = pattern_store.query_by_class("config_drift")
        assert results[0].pattern_type == "configuration_change"

    def test_promote_without_pattern_store_unchanged(self, tmp_path):
        """向后兼容：pattern_store=None 时不写"""
        registry = _make_fake_registry()
        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
        )
        run = _make_run_record()
        engine.promote(run)
        # champion.json 仍正常写
        assert (tmp_path / "champion.json").exists()


# ============================================================
# 5. 失败不阻断 promote 主链路
# ============================================================

class TestFailureDoesNotBreakPromote:
    def test_ltm_failure_does_not_block_promote(self, tmp_path):
        registry = _make_fake_registry()

        ltm = MagicMock()
        ltm.store = AsyncMock(side_effect=OSError("LTM broken"))

        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            ltm=ltm,
        )
        run = _make_run_record()
        # 不应抛异常
        engine.promote(run)
        # champion.json 仍正常写
        assert (tmp_path / "champion.json").exists()

    def test_pattern_store_failure_does_not_block_promote(self, tmp_path):
        registry = _make_fake_registry()
        pattern_store = MagicMock()
        pattern_store.add = MagicMock(side_effect=OSError("disk full"))

        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            pattern_store=pattern_store,
        )
        run = _make_run_record()
        # 不应抛异常
        engine.promote(run)
        # champion.json 仍正常写
        assert (tmp_path / "champion.json").exists()


# ============================================================
# 6. run_sync() 端到端：promote 链路完整
# ============================================================

class TestEndToEndFlywheel:
    def test_run_sync_promote_commits_to_both_ltm_and_kb(self, tmp_path):
        """端到端：run_sync() → _decide="promote" → promote() → LTM + KB"""
        registry = _make_fake_registry()
        ltm = _make_fake_ltm()
        pattern_store = RuntimePatternStore(store_path=str(tmp_path / "patterns.jsonl"))

        # 构造 _eval_fn 直接返回"高指标"保证 promote
        def high_metrics_eval(candidate, entries, champion):
            return (
                {"precision": 0.9, "recall": 0.85, "f1": 0.87},
                {"precision": 0.85, "recall": 0.80, "f1": 0.82},
            )

        engine = HarnessLoopEngine(
            registry=registry,
            runs_path=str(tmp_path / "runs.jsonl"),
            champion_path=str(tmp_path / "champion.json"),
            _eval_fn=high_metrics_eval,
            ltm=ltm,
            pattern_store=pattern_store,
        )

        # 模拟 split_for_eval 返回固定 split
        registry.split_for_eval = MagicMock(
            return_value=(["bc1", "bc2"], ["bc3"])
        )
        registry.query_similar = MagicMock(return_value=["bc1", "bc2", "bc3"])

        # 先写一个基础 champion（确保 _decide 走 promote 分支而非 hold_for_human）
        from datetime import datetime, timezone
        base_run = HarnessRunRecord(
            run_id="hr-base000000",
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            badcase_class="memory_oom_high",
            candidate=_make_candidate("memory_oom_high"),
            base_champion_version="v0.0.0",
            dev_metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
            test_metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
            decision="promote",
            promoted_at=datetime.now(timezone.utc),
        )
        engine.promote(base_run)

        # 重置计数（LTM 和 pattern_store 都被 base 调用过 1 次）
        ltm.store.reset_mock()
        # pattern_store 重新初始化清掉 base 那条
        pattern_store = RuntimePatternStore(store_path=str(tmp_path / "patterns2.jsonl"))
        engine._pattern_store = pattern_store

        candidate = _make_candidate(badcase_class="memory_oom_high")
        record = engine.run_sync(candidate)

        # 验证决策是 promote
        assert record.decision == "promote"
        # 验证 LTM + KB 都被调用
        assert ltm.store.await_count == 1
        assert pattern_store.count() == 1
        # KB 中能查到
        results = pattern_store.query_by_class("memory_oom_high")
        assert len(results) == 1
