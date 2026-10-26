"""
SelfEvolution 5 阶段闭环测试

覆盖：
1. PromptVersion + EvolutionCycle 模型
2. PromptStore CRUD
3. SelfEvolutionOrchestrator.run_cycle 完整 5 阶段
4. A/B 测试有量化指标
5. promote / reject 决策
6. CLI run/status/show/promote/reject（subprocess）
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.evolution.models import (
    EvolutionCycle,
    EvolutionStage,
    PromotionDecision,
    PromptVersion,
)
from app.evolution.store import PromptStore, get_prompt_store, reset_prompt_store
from app.evolution.self_evolution import SelfEvolutionOrchestrator


@pytest.fixture
def store(tmp_path) -> PromptStore:
    db_path = tmp_path / "evolution.db"
    reset_prompt_store()
    s = PromptStore(str(db_path))
    yield s
    reset_prompt_store()


# =====================================================================
# 模型测试
# =====================================================================

class TestEvolutionModels:
    def test_prompt_version_default(self):
        pv = PromptVersion()
        assert pv.version == ""
        assert pv.badcase_ids == []

    def test_cycle_to_dict(self):
        c = EvolutionCycle(badcase_ids=["bc-1"], generated_prompt="test")
        d = c.to_dict()
        assert d["badcase_ids"] == ["bc-1"]
        assert d["current_stage"] == EvolutionStage.BADCASE_COLLECTED.value


# =====================================================================
# PromptStore 测试
# =====================================================================

class TestPromptStore:
    def test_save_and_get_prompt(self, store: PromptStore):
        pv = PromptVersion(
            version="v1.0",
            prompt_segment="test prompt",
            target_dimension="accuracy",
        )
        saved = store.save_prompt_version(pv)
        assert saved.id is not None

        latest = store.get_latest_prompt()
        assert latest is not None
        assert latest.version == "v1.0"
        assert latest.prompt_segment == "test prompt"

    def test_get_prompt_by_version(self, store: PromptStore):
        store.save_prompt_version(PromptVersion(version="v1.0", prompt_segment="a"))
        store.save_prompt_version(PromptVersion(version="v1.1", prompt_segment="b"))

        v11 = store.get_prompt_by_version("v1.1")
        assert v11.prompt_segment == "b"

    def test_save_cycle(self, store: PromptStore):
        c = EvolutionCycle(badcase_ids=["bc-1"], pattern_analysis={"type": "cpu"})
        saved = store.save_cycle(c)
        assert saved.id is not None

        fetched = store.get_cycle(saved.id)
        assert fetched is not None
        assert fetched.badcase_ids == ["bc-1"]
        assert fetched.pattern_analysis.get("type") == "cpu"

    def test_update_cycle(self, store: PromptStore):
        c = EvolutionCycle(badcase_ids=["bc-1"])
        store.save_cycle(c)

        c.generated_prompt = "new prompt"
        c.promotion_decision = "pending"
        c.current_stage = "prompt_generated"
        store.update_cycle(c)

        fetched = store.get_cycle(c.id)
        assert fetched.generated_prompt == "new prompt"
        assert fetched.current_stage == "prompt_generated"

    def test_list_cycles(self, store: PromptStore):
        for i in range(3):
            store.save_cycle(EvolutionCycle(
                badcase_ids=[f"bc-{i}"],
                promotion_decision="pending" if i < 2 else "rejected",
            ))
        all_cycles = store.list_cycles(limit=10)
        assert len(all_cycles) == 3

        rejected = store.list_cycles(limit=10, decision="rejected")
        assert len(rejected) == 1


# =====================================================================
# SelfEvolutionOrchestrator 测试
# =====================================================================

class TestSelfEvolutionOrchestrator:
    def test_run_cycle_empty_badcases(self, store: PromptStore):
        orch = SelfEvolutionOrchestrator(store=store)
        cycle = orch.run_cycle(badcase_ids=[], current_prompt="test")

        assert cycle.id is not None
        assert cycle.review_notes == "无 badcase，跳过"

    def test_run_cycle_full_5_stages(self, store: PromptStore):
        """跑 5 阶段循环（注入 mock analyzer / prompt_generator / rca_evaluator，避免依赖 LLM）"""
        from app.eval.evaluator import RCAOutput

        def mock_analyzer(badcases):
            return {
                "pattern_type": "cpu_saturation",
                "badcase_count": len(badcases),
                "common_keywords": ["cpu", "load"],
                "affected_dimensions": ["accuracy"],
                "proposed_fix_direction": "增加 CPU 阈值说明",
            }

        def mock_generator(analysis, current_prompt):
            return current_prompt + "\n\n## 修复段\n针对 cpu_saturation"

        def mock_rca(alert, evidence):
            return RCAOutput(
                root_cause="cpu_saturation_test",
                confidence=0.9,
                evidence_refs=["mock:cpu"],
                method="mock",
            )

        orch = SelfEvolutionOrchestrator(
            store=store,
            analyzer=mock_analyzer,
            prompt_generator=mock_generator,
            rca_evaluator=mock_rca,
        )
        cycle = orch.run_cycle(
            badcase_ids=["bc-001", "bc-002", "bc-003"],
            current_prompt="你是 SRE 分析师，请分析告警并给出根因。",
        )

        assert cycle.id is not None
        assert cycle.completed_at is not None
        # 阶段 2：pattern_analysis 有结果（来自 mock_analyzer）
        assert cycle.pattern_analysis.get("pattern_type") == "cpu_saturation"
        # 阶段 3：生成 prompt（来自 mock_generator）
        assert "修复段" in cycle.generated_prompt
        # 阶段 4：A/B 测试有结果
        assert "old_pass_rate" in cycle.ab_test_summary
        assert "new_pass_rate" in cycle.ab_test_summary
        assert "delta" in cycle.ab_test_summary
        assert "auto_accept" in cycle.ab_test_summary

    def test_run_cycle_completes_with_pending(self, store: PromptStore):
        """A/B 测试不显著 → 应 PENDING 等待人工 review"""
        from app.eval.evaluator import RCAOutput

        def mock_analyzer(badcases):
            return {"pattern_type": "test", "badcase_count": 1, "common_keywords": [], "affected_dimensions": [], "proposed_fix_direction": ""}

        def mock_generator(analysis, current_prompt):
            return current_prompt + " (improved)"

        def mock_rca(alert, evidence):
            return RCAOutput(root_cause="x", confidence=0.5, evidence_refs=["x"], method="mock")

        orch = SelfEvolutionOrchestrator(
            store=store,
            ab_test_threshold=0.5,  # 高阈值
            analyzer=mock_analyzer,
            prompt_generator=mock_generator,
            rca_evaluator=mock_rca,
        )
        cycle = orch.run_cycle(
            badcase_ids=["bc-001"],
            current_prompt="test prompt",
        )
        # delta 通常 < 0.5，所以 PENDING
        assert cycle.promotion_decision == PromotionDecision.PENDING.value

    def test_run_cycle_with_injected_analyzer(self, store: PromptStore):
        """注入自定义 analyzer（同时注入其他依赖避免 LLM 调用）"""
        from app.eval.evaluator import RCAOutput

        def custom_analyzer(badcases):
            return {
                "pattern_type": "custom_pattern",
                "badcase_count": len(badcases),
                "proposed_fix_direction": "custom fix",
            }

        orch = SelfEvolutionOrchestrator(
            store=store,
            analyzer=custom_analyzer,
            prompt_generator=lambda a, p: p + "\n\n## 修复段\ncustom",
            rca_evaluator=lambda a, e: RCAOutput(root_cause="x", confidence=0.5, evidence_refs=["x"], method="mock"),
        )
        cycle = orch.run_cycle(
            badcase_ids=["bc-001"],
            current_prompt="test",
        )
        assert cycle.pattern_analysis["pattern_type"] == "custom_pattern"

    def test_promote_writes_prompt_version(self, store: PromptStore):
        from app.eval.evaluator import RCAOutput
        from app.evolution.self_evolution import SelfEvolutionOrchestrator

        orch = SelfEvolutionOrchestrator(
            store=store,
            analyzer=lambda bcs: {"pattern_type": "t", "badcase_count": len(bcs), "common_keywords": [], "affected_dimensions": [], "proposed_fix_direction": ""},
            prompt_generator=lambda a, p: p + "\n\n## 修复段\ntest",
            rca_evaluator=lambda a, e: RCAOutput(root_cause="x", confidence=0.5, evidence_refs=["x"], method="mock"),
        )
        cycle = orch.run_cycle(
            badcase_ids=["bc-001"],
            current_prompt="old prompt",
        )

        # promote
        result = orch.promote(cycle.id, reviewer="zhangsan", notes="验证通过")
        assert result.promotion_decision == PromotionDecision.PROMOTED.value
        assert result.reviewer == "zhangsan"

        # 检查新 prompt 版本
        latest = store.get_latest_prompt()
        assert latest is not None
        assert latest.cycle_id == cycle.id
        # v3 E4: mock generator 返回的 prompt 包含修复段（不一定是"自进化修复段"）
        assert "修复段" in latest.prompt_segment
        # 旧 prompt 也保留
        assert "old prompt" in latest.prompt_segment

    def test_reject(self, store: PromptStore):
        from app.eval.evaluator import RCAOutput

        orch = SelfEvolutionOrchestrator(
            store=store,
            analyzer=lambda bcs: {"pattern_type": "t", "badcase_count": len(bcs), "common_keywords": [], "affected_dimensions": [], "proposed_fix_direction": ""},
            prompt_generator=lambda a, p: p + "\n\n## 修复段\ntest",
            rca_evaluator=lambda a, e: RCAOutput(root_cause="x", confidence=0.5, evidence_refs=["x"], method="mock"),
        )
        cycle = orch.run_cycle(badcase_ids=["bc-001"], current_prompt="test")

        result = orch.reject(cycle.id, reviewer="lisi", notes="改动太大")
        assert result.promotion_decision == PromotionDecision.REJECTED.value

    def test_double_promote_warns(self, store: PromptStore):
        from app.eval.evaluator import RCAOutput

        orch = SelfEvolutionOrchestrator(
            store=store,
            analyzer=lambda bcs: {"pattern_type": "t", "badcase_count": len(bcs), "common_keywords": [], "affected_dimensions": [], "proposed_fix_direction": ""},
            prompt_generator=lambda a, p: p + "\n\n## 修复段\ntest",
            rca_evaluator=lambda a, e: RCAOutput(root_cause="x", confidence=0.5, evidence_refs=["x"], method="mock"),
        )
        cycle = orch.run_cycle(badcase_ids=["bc-001"], current_prompt="test")
        orch.promote(cycle.id, reviewer="zhangsan")
        # 第二次 promote 应是 no-op
        result = orch.promote(cycle.id, reviewer="lisi")
        assert result.reviewer == "zhangsan"  # 不变


# =====================================================================
# CLI 测试
# =====================================================================

class TestEvolutionCLI:
    @pytest.fixture
    def isolated_db(self, tmp_path, monkeypatch):
        db_path = tmp_path / "evolution.db"
        monkeypatch.setenv("EVOLUTION_DB_PATH", str(db_path))
        return db_path

    def test_cli_run_and_status(self, isolated_db):
        cwd = Path(__file__).resolve().parents[1]
        # v3 E4: 强制 fallback 避免调真 LLM
        env = {**os.environ, "AIOPS_FORCE_FALLBACK": "1"}
        # 用真实存在的 badcase id 加速（1 个而非全集 49 个）
        run_result = subprocess.run(
            [sys.executable, "-m", "app.evolution", "run",
             "--badcases", "bc-aiops2020-001",
             "--prompt", "test prompt"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
        )
        assert run_result.returncode == 0, f"stdout={run_result.stdout}, stderr={run_result.stderr}"
        assert "cycle#" in run_result.stdout

        # status
        status_result = subprocess.run(
            [sys.executable, "-m", "app.evolution", "status"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
        )
        assert status_result.returncode == 0
        assert "cycle#" in status_result.stdout

    def test_cli_promote(self, isolated_db):
        cwd = Path(__file__).resolve().parents[1]
        env = {**os.environ, "AIOPS_FORCE_FALLBACK": "1"}
        subprocess.run(
            [sys.executable, "-m", "app.evolution", "run",
             "--badcases", "bc-aiops2020-001", "--prompt", "test"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.evolution", "promote", "1",
             "--reviewer", "zhangsan", "--notes", "ok"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
        )
        assert result.returncode == 0
        assert "PROMOTE" in result.stdout

    def test_cli_show(self, isolated_db):
        cwd = Path(__file__).resolve().parents[1]
        env = {**os.environ, "AIOPS_FORCE_FALLBACK": "1"}
        subprocess.run(
            [sys.executable, "-m", "app.evolution", "run",
             "--badcases", "bc-aiops2020-001", "--prompt", "test prompt"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
        )

        result = subprocess.run(
            [sys.executable, "-m", "app.evolution", "show", "1"],
            capture_output=True, text=True, cwd=str(cwd), env=env,
        )
        assert result.returncode == 0
        assert "cycle#1" in result.stdout
        assert "模式分析" in result.stdout
        assert "A/B 测试" in result.stdout
