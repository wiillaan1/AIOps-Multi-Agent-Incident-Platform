"""
[v3 E1+E3 改造测试] 2026-09-01

覆盖：
- BadCaseEntry 统一 schema（评估 + 捕获 + 注册 共用）
- eval_split 三分离 (dev / test / holdout)
- source 字段（数据集溯源）
- severity / category / type / tags 字段
- from_dict 自动生成 id
- BadcaseDataset 加载 v1_dev.json / v1_test.json / v1_holdout.json
- get_split_path
- 缺失字段容错
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from app.models.badcase import (
    BadCaseEntry,
    BadcaseSource,
    BadCaseSeverity,
    EvalSplit,
)
from app.eval.dataset import (
    Badcase,
    BadcaseDataset,
    get_default_dataset,
    get_split_path,
)


# =====================================================================
# 统一 Schema 测试
# =====================================================================


class TestUnifiedSchema:
    """BadCaseEntry 同时承载评估字段 + 捕获字段。"""

    def test_full_entry_creation(self):
        """完整字段：评估 + 捕获 + 元数据都能填"""
        entry = BadCaseEntry(
            id="bc-eval-cap-001",
            alert={"service": "web-svc", "metric_name": "cpu_usage", "value": 95},
            evidence={"cpu_history": [80, 85, 95], "related_services": ["db-svc"]},
            expected_root_cause="cpu_high_db_connection_pool",
            expected_confidence_min=0.8,
            category="single_metric",
            type="cpu_high",
            source="manual",
            eval_split="dev",
            severity="high",
            identified_flaw="CPU spikes but alert payload missing context",
            keywords_for_retrieval=["cpu", "db", "connection"],
            suggestion_or_lesson="Add metric labels",
            tags=["regression", "high-priority"],
            description="CPU saturation caused by db connection pool exhaustion",
            source_ref="manual/2026-09-01/test-001",
        )
        assert entry.id == "bc-eval-cap-001"
        assert entry.severity == BadCaseSeverity.HIGH
        assert entry.eval_split == EvalSplit.DEV
        assert entry.source == "manual"
        assert "cpu" in entry.keywords_for_retrieval

    def test_id_must_start_with_bc_prefix(self):
        with pytest.raises(Exception):
            BadCaseEntry(
                id="test-001",  # 缺 bc- 前缀
                alert={}, evidence={}, expected_root_cause="x",
            )

    def test_id_pattern_min_2_chars(self):
        """id 后缀至少 2 字符"""
        with pytest.raises(Exception):
            BadCaseEntry(
                id="bc-x",  # x 只有 1 字符
                alert={}, evidence={}, expected_confidence_min=0.5,
                expected_root_cause="x",
            )

    def test_id_pattern_max_80_chars(self):
        """id 后缀最多 80 字符（AIOpsLab problem_id 可能 40+ 字符）"""
        with pytest.raises(Exception):
            BadCaseEntry(
                id="bc-" + "a" * 81,
                alert={}, evidence={}, expected_confidence_min=0.5,
                expected_root_cause="x",
            )

    def test_id_pattern_accepts_long_aiopslab_ids(self):
        """AIOpsLab problem_id（如 bc-aiopslab-redeploy_without_PV-analysis-1）应被接受"""
        bc = BadCaseEntry(
            id="bc-aiopslab-redeploy_without_PV-analysis-1",
            alert={}, evidence={}, expected_confidence_min=0.5,
            expected_root_cause="x",
        )
        assert bc.id == "bc-aiopslab-redeploy_without_PV-analysis-1"

    def test_default_eval_split_is_dev(self):
        entry = BadCaseEntry(
            id="bc-default-split",
            alert={}, evidence={}, expected_confidence_min=0.5,
            expected_root_cause="x",
        )
        assert entry.eval_split == EvalSplit.DEV

    def test_default_severity_is_medium(self):
        entry = BadCaseEntry(
            id="bc-default-sev",
            alert={}, evidence={}, expected_confidence_min=0.5,
            expected_root_cause="x",
        )
        assert entry.severity == BadCaseSeverity.MEDIUM

    def test_confidence_min_bounds(self):
        """expected_confidence_min 必须在 [0, 1]"""
        with pytest.raises(Exception):
            BadCaseEntry(
                id="bc-conf-high", alert={}, evidence={},
                expected_root_cause="x", expected_confidence_min=1.5,
            )
        with pytest.raises(Exception):
            BadCaseEntry(
                id="bc-conf-low", alert={}, evidence={},
                expected_root_cause="x", expected_confidence_min=-0.1,
            )

    def test_keywords_dedupe_lowercase(self):
        entry = BadCaseEntry(
            id="bc-kw-1", alert={}, evidence={},
            expected_root_cause="x", expected_confidence_min=0.5,
            keywords_for_retrieval=["CPU", "  cpu  ", "Memory", "memory"],
        )
        # 大写转小写，去空白
        assert all(k == k.lower() for k in entry.keywords_for_retrieval)
        assert all(k == k.strip() for k in entry.keywords_for_retrieval)

    def test_alias_Badcase_is_BadCaseEntry(self):
        """向后兼容：Badcase 别名 = BadCaseEntry"""
        from app.eval.dataset import Badcase as DatasetBadcase
        assert DatasetBadcase is BadCaseEntry


# =====================================================================
# from_dict 容错测试
# =====================================================================


class TestFromDictRobustness:
    def test_auto_id_when_missing(self):
        """id 缺失时基于 alert + root_cause 哈希生成"""
        data = {
            "alert": {"service": "web", "metric_name": "cpu"},
            "evidence": {},
            "expected_root_cause": "resource_saturation",
            "expected_confidence_min": 0.7,
        }
        entry = BadCaseEntry.from_dict(data)
        assert entry.id.startswith("bc-")
        assert len(entry.id) == 15  # bc- + 12 hex chars
        assert entry.expected_confidence_min == 0.7

    def test_default_values_when_missing(self):
        """字段缺失用默认值"""
        data = {
            "id": "bc-defaults",
            "alert": {"service": "x"},
            "expected_root_cause": "x",
        }
        entry = BadCaseEntry.from_dict(data)
        assert entry.category == "single_metric"
        assert entry.type == "unknown"
        assert entry.expected_confidence_min == 0.5
        assert entry.eval_split == EvalSplit.DEV

    def test_idempotent_loading(self):
        """同一份数据 → 同一份 entry"""
        data = {
            "alert": {"service": "web"},
            "evidence": {},
            "expected_root_cause": "saturation",
            "expected_confidence_min": 0.7,
        }
        e1 = BadCaseEntry.from_dict(data)
        e2 = BadCaseEntry.from_dict(data)
        assert e1.id == e2.id


# =====================================================================
# 三分离数据集加载
# =====================================================================


class TestThreeSplitDataset:
    @pytest.fixture
    def temp_dataset_dir(self, tmp_path: Path) -> Path:
        """创建临时三分离数据集"""
        ds_dir = tmp_path / "badcases"
        ds_dir.mkdir()

        # dev: 5 个
        dev_data = {
            "_meta": {"version": "v1", "description": "dev split"},
            "badcases": [
                {
                    "id": "bc-dev-001", "source": "manual",
                    "alert": {"service": "web"}, "evidence": {},
                    "expected_root_cause": "cpu_high", "expected_confidence_min": 0.7,
                    "category": "single_metric", "type": "cpu",
                    "severity": "high", "eval_split": "dev",
                }
            ] * 5,
        }
        # 实际生成 5 条
        for i in range(5):
            dev_data["badcases"][i] = {
                "id": f"bc-dev-{i:03d}",
                "source": "manual",
                "alert": {"service": "web", "metric_name": f"cpu_{i}"},
                "evidence": {},
                "expected_root_cause": f"cpu_high_{i}",
                "expected_confidence_min": 0.7,
                "category": "single_metric",
                "type": "cpu",
                "severity": "high",
                "eval_split": "dev",
            }
        (ds_dir / "v1_dev.json").write_text(json.dumps(dev_data), encoding="utf-8")

        # test: 2 个
        test_data = {
            "_meta": {"version": "v1", "description": "test split"},
            "badcases": [
                {
                    "id": f"bc-test-{i:03d}",
                    "source": "aiops2020",
                    "alert": {"service": "api"},
                    "evidence": {},
                    "expected_root_cause": "db_timeout",
                    "expected_confidence_min": 0.8,
                    "category": "compound",
                    "type": "db",
                    "severity": "critical",
                    "eval_split": "test",
                }
                for i in range(2)
            ],
        }
        (ds_dir / "v1_test.json").write_text(json.dumps(test_data), encoding="utf-8")

        # holdout: 1 个
        holdout_data = {
            "_meta": {"version": "v1", "description": "holdout split"},
            "badcases": [
                {
                    "id": "bc-holdout-001",
                    "source": "aiopslab",
                    "alert": {"service": "frontend"},
                    "evidence": {},
                    "expected_root_cause": "memory_leak",
                    "expected_confidence_min": 0.9,
                    "category": "regression_guard",
                    "type": "memory",
                    "severity": "high",
                    "eval_split": "holdout",
                }
            ],
        }
        (ds_dir / "v1_holdout.json").write_text(json.dumps(holdout_data), encoding="utf-8")

        return ds_dir

    def test_load_dev_split(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_dev.json", eval_split="dev")
        assert len(ds.badcases) == 5
        assert ds.eval_split == "dev"
        assert all(bc.eval_split == EvalSplit.DEV for bc in ds.badcases)

    def test_load_test_split(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_test.json", eval_split="test")
        assert len(ds.badcases) == 2
        assert all(bc.eval_split == EvalSplit.TEST for bc in ds.badcases)

    def test_load_holdout_split(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_holdout.json", eval_split="holdout")
        assert len(ds.badcases) == 1
        assert all(bc.eval_split == EvalSplit.HOLDOUT for bc in ds.badcases)

    def test_eval_split_preserved_on_load(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_test.json", eval_split="test")
        assert ds.eval_split == "test"

    def test_by_source(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_test.json")
        aiops2020_bc = ds.by_source("aiops2020")
        assert len(aiops2020_bc) == 2

    def test_by_severity(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_test.json")
        crit = ds.by_severity("critical")
        assert len(crit) == 2

    def test_stats_includes_new_fields(self, temp_dataset_dir: Path):
        ds = BadcaseDataset.load(temp_dataset_dir / "v1_test.json")
        stats = ds.stats()
        assert "by_source" in stats
        assert "by_severity" in stats
        assert "eval_split" in stats
        assert stats["by_source"]["aiops2020"] == 2
        assert stats["by_severity"]["critical"] == 2

    def test_split_separation(self, temp_dataset_dir: Path):
        """dev / test / holdout 数据互不重叠"""
        dev = BadcaseDataset.load(temp_dataset_dir / "v1_dev.json", eval_split="dev")
        test = BadcaseDataset.load(temp_dataset_dir / "v1_test.json", eval_split="test")
        holdout = BadcaseDataset.load(temp_dataset_dir / "v1_holdout.json", eval_split="holdout")

        dev_ids = {bc.id for bc in dev.badcases}
        test_ids = {bc.id for bc in test.badcases}
        holdout_ids = {bc.id for bc in holdout.badcases}

        assert dev_ids.isdisjoint(test_ids)
        assert dev_ids.isdisjoint(holdout_ids)
        assert test_ids.isdisjoint(holdout_ids)


# =====================================================================
# 路径与异常
# =====================================================================


class TestPathAndExceptions:
    def test_get_split_path_dev(self):
        p = get_split_path("dev")
        assert p.name == "v1_dev.json"

    def test_get_split_path_test(self):
        p = get_split_path("test")
        assert p.name == "v1_test.json"

    def test_get_split_path_holdout(self):
        p = get_split_path("holdout")
        assert p.name == "v1_holdout.json"

    def test_load_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            BadcaseDataset.load(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            BadcaseDataset.load(bad_file)

    def test_default_dataset_helpful_error(self, tmp_path, monkeypatch):
        """默认路径没文件时给清晰提示"""
        # Patch 所有路径到不存在的目录
        import app.eval.dataset as ds_mod
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(ds_mod, "_BASE_DIR", empty)
        monkeypatch.setattr(ds_mod, "_DEFAULT_PATH", empty / "v1.json")
        with pytest.raises(FileNotFoundError, match="请运行数据适配器"):
            ds_mod.get_default_dataset()


# =====================================================================
# 端到端：Badcase → RCAEvaluator 流程跑通
# =====================================================================


class TestEndToEndFlow:
    def test_badcase_fed_into_evaluator(self):
        from app.eval.evaluator import RCAEvaluator, RCAOutput

        # 1. 创建 badcase
        bc = BadCaseEntry(
            id="bc-e2e-001",
            alert={"service": "web", "metric_name": "cpu_usage"},
            evidence={"cpu_history": [95]},
            expected_root_cause="cpu_high",
            expected_confidence_min=0.7,
            category="single_metric",
            type="cpu",
            eval_split="dev",
            severity="high",
        )

        # 2. Mock RCA 函数
        def mock_rca(alert, evidence):
            return RCAOutput(
                root_cause="cpu_high_resource_saturation",
                confidence=0.85,
                evidence_refs=["metric:cpu_usage"],
                method="rules_engine",
            )

        # 3. 评估
        evaluator = RCAEvaluator(rca_func=mock_rca)
        result = evaluator.evaluate_one(bc)

        # 4. 全部通过
        assert result.pass_root_cause_match is True
        assert result.pass_confidence_meet is True
        assert result.pass_citation_present is True
        assert result.pass_no_hallucination is True
        assert result.pass_overall is True

    def test_severity_does_not_affect_eval(self):
        """severity 是元数据，不影响评估结果"""
        from app.eval.evaluator import RCAEvaluator, RCAOutput

        def make_bc(severity):
            return BadCaseEntry(
                id=f"bc-sev-{severity}",
                alert={"service": "web"},
                evidence={},
                expected_root_cause="cpu_high",
                expected_confidence_min=0.7,
                severity=severity,
            )

        def mock_rca(alert, evidence):
            return RCAOutput(
                root_cause="cpu_high", confidence=0.8,
                evidence_refs=["x"],
            )

        evaluator = RCAEvaluator(rca_func=mock_rca)
        r_low = evaluator.evaluate_one(make_bc("low"))
        r_crit = evaluator.evaluate_one(make_bc("critical"))

        # 同样的输入，同样的输出（severity 不影响)
        assert r_low.pass_overall == r_crit.pass_overall
