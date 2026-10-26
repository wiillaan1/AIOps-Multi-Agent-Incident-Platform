"""v3 E5: 评测历史 + baseline 测试

测试 25 个：
- record_eval_run: 4 个
- baseline CRUD: 6 个
- list_runs / get_trend: 4 个
- run_evaluation 自动记录: 3 个
- check_against_baseline: 4 个
- CLI: 4 个
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.eval.history import (
    check_against_baseline,
    get_baselines_dir,
    get_history_db_path,
    get_latest_run,
    get_trend,
    list_baselines,
    list_runs,
    load_baseline,
    record_eval_run,
    save_baseline,
)
from app.eval.evaluator import RCAEvaluator, RCAOutput, run_evaluation
from app.eval.dataset import get_default_dataset


@pytest.fixture
def isolated_history(tmp_path: Path, monkeypatch):
    """隔离历史库到 tmp_path"""
    db_path = tmp_path / "history.db"
    baselines_dir = tmp_path / "baselines"
    monkeypatch.setenv("EVAL_HISTORY_DB_PATH", str(db_path))
    monkeypatch.setenv("EVAL_BASELINES_DIR", str(baselines_dir))
    return db_path, baselines_dir


def _mock_summary_factory(passed=5, total=10, rate=0.5):
    """构造一个 mock 的 RCAEvaluationSummary（用 to_dict 接口）"""
    class _S:
        def to_dict(self):
            return {
                "total": total,
                "passed": passed,
                "pass_rate": rate,
                "avg_confidence": 0.7,
                "by_dimension": {
                    "pass_root_cause_match": {"passed": passed, "total": total, "rate": rate},
                    "pass_confidence_threshold": {"passed": passed, "total": total, "rate": rate},
                    "pass_citation_present": {"passed": passed, "total": total, "rate": rate},
                    "pass_no_hallucination": {"passed": passed, "total": total, "rate": rate},
                },
                "by_category": {
                    "single_metric": {"passed": passed, "total": total, "rate": rate},
                },
                "failing_ids": ["bc-fail-1", "bc-fail-2"],
            }
    return _S()


class TestRecordEvalRun:
    def test_record_returns_eval_id(self, isolated_history):
        eval_id = record_eval_run(
            split="dev", summary=_mock_summary_factory(), note="test run"
        )
        assert eval_id.startswith("eval-")
        assert len(eval_id) > 12

    def test_record_persists_to_db(self, isolated_history):
        db_path, _ = isolated_history
        record_eval_run(
            split="dev",
            analyzer="rules_engine",
            summary=_mock_summary_factory(passed=5, total=10, rate=0.5),
        )
        rows = list_runs()
        assert len(rows) == 1
        r = rows[0]
        assert r["split"] == "dev"
        assert r["analyzer"] == "rules_engine"
        assert r["total"] == 10
        assert r["passed"] == 5
        assert r["pass_rate"] == 0.5

    def test_record_with_config_and_note(self, isolated_history):
        eval_id = record_eval_run(
            split="test",
            model="deepseek-chat",
            prompt_version="v3",
            analyzer="llm_v3",
            summary=_mock_summary_factory(),
            config={"temperature": 0.2, "max_tokens": 600},
            note="production A/B test",
        )
        row = list_runs()[0]
        assert row["model"] == "deepseek-chat"
        assert row["prompt_version"] == "v3"
        assert row["note"] == "production A/B test"
        config = json.loads(row["config_json"])
        assert config["temperature"] == 0.2

    def test_record_upsert_on_same_id(self, isolated_history):
        eval_id = "eval-fixed-id"
        record_eval_run(eval_id=eval_id, split="dev", summary=_mock_summary_factory(rate=0.3))
        record_eval_run(eval_id=eval_id, split="dev", summary=_mock_summary_factory(rate=0.7))
        rows = list_runs()
        assert len(rows) == 1
        assert list_runs()[0]["pass_rate"] == 0.7


class TestBaselineCRUD:
    def test_save_and_load_baseline(self, isolated_history):
        _, baselines_dir = isolated_history
        path = save_baseline(
            "rules_engine_v1",
            pass_rate=0.3,
            total=49,
            passed=15,
            description="baseline after LLM outage",
            split="dev",
        )
        assert path.exists()
        b = load_baseline("rules_engine_v1")
        assert b["pass_rate"] == 0.3
        assert b["total"] == 49
        assert b["split"] == "dev"

    def test_load_nonexistent_returns_none(self, isolated_history):
        b = load_baseline("does-not-exist")
        assert b is None

    def test_list_baselines(self, isolated_history):
        save_baseline("b1", pass_rate=0.3, total=10, passed=3)
        save_baseline("b2", pass_rate=0.5, total=10, passed=5)
        bls = list_baselines()
        names = [b["name"] for b in bls]
        assert "b1" in names
        assert "b2" in names

    def test_save_overwrites_existing(self, isolated_history):
        save_baseline("b1", pass_rate=0.3, total=10, passed=3)
        save_baseline("b1", pass_rate=0.5, total=10, passed=5)
        b = load_baseline("b1")
        assert b["pass_rate"] == 0.5


class TestTrend:
    def test_trend_chronological_order(self, isolated_history):
        # record 3 runs in reverse order (newest first)
        for i, rate in enumerate([0.3, 0.5, 0.7]):
            record_eval_run(
                eval_id=f"eval-{i}",
                split="dev",
                summary=_mock_summary_factory(rate=rate),
            )
        trend = get_trend(split="dev")
        # get_trend returns oldest-first
        assert len(trend) == 3
        assert trend[0]["pass_rate"] == 0.3
        assert trend[2]["pass_rate"] == 0.7

    def test_trend_filters_by_split(self, isolated_history):
        record_eval_run(split="dev", summary=_mock_summary_factory(rate=0.3))
        record_eval_run(split="test", summary=_mock_summary_factory(rate=0.5))
        dev_trend = get_trend(split="dev")
        test_trend = get_trend(split="test")
        assert len(dev_trend) == 1
        assert len(test_trend) == 1
        assert dev_trend[0]["pass_rate"] == 0.3

    def test_latest_run_returns_newest(self, isolated_history):
        record_eval_run(eval_id="e1", split="dev", summary=_mock_summary_factory(rate=0.3))
        record_eval_run(eval_id="e2", split="dev", summary=_mock_summary_factory(rate=0.7))
        latest = get_latest_run(split="dev")
        assert latest["eval_id"] == "e2"


class TestRunEvaluationAutoRecord:
    def test_run_evaluation_records_history(self, isolated_history):
        # 用 dev split 真跑一次
        ds = get_default_dataset(eval_split="dev")
        # mock rca_func 让其全过
        def good_rca(alert, evidence):
            return RCAOutput(
                root_cause="cpu_high_db_connection_pool",
                confidence=0.95,
                evidence_refs=["cpu_history", "related_services"],
                reasoning="test",
                method="test",
            )
        run_evaluation(ds, rca_func=good_rca, analyzer_name="test_mock", history_note="unit test")
        rows = list_runs(split="dev")
        assert len(rows) >= 1
        assert rows[0]["analyzer"] == "test_mock"
        assert rows[0]["note"] == "unit test"

    def test_run_evaluation_record_can_be_disabled(self, isolated_history):
        ds = get_default_dataset(eval_split="dev")
        run_evaluation(ds, record_history=False)
        rows = list_runs()
        assert len(rows) == 0

    def test_run_evaluation_default_uses_pass_rate_in_db(self, isolated_history):
        ds = get_default_dataset(eval_split="dev")
        run_evaluation(ds, analyzer_name="rules_engine_fallback")
        row = get_latest_run()
        assert row["analyzer"] == "rules_engine_fallback"
        assert 0.0 <= row["pass_rate"] <= 1.0


class TestCheckAgainstBaseline:
    def test_pass_when_better_than_baseline(self, isolated_history):
        save_baseline("b1", pass_rate=0.3, total=10, passed=3)
        result = check_against_baseline(_mock_summary_factory(rate=0.5), "b1")
        assert result["passed"] is True
        assert result["delta"] == pytest.approx(0.2)

    def test_fail_when_worse_than_baseline(self, isolated_history):
        save_baseline("b1", pass_rate=0.7, total=10, passed=7)
        result = check_against_baseline(_mock_summary_factory(rate=0.5), "b1")
        assert result["passed"] is False
        assert result["delta"] < 0

    def test_missing_baseline_returns_pass(self, isolated_history):
        result = check_against_baseline(_mock_summary_factory(rate=0.5), "missing")
        assert result["passed"] is True
        assert "不存在" in result["reason"]

    def test_custom_min_pass_rate_overrides(self, isolated_history):
        save_baseline("b1", pass_rate=0.3, total=10, passed=3)
        # 0.5 > 0.4 但 0.5 < 0.3 不成立（实际 0.5 > 0.3）
        result = check_against_baseline(_mock_summary_factory(rate=0.4), "b1", min_pass_rate=0.6)
        assert result["passed"] is False


class TestEvalCLI:
    """测试 eval CLI 子命令（用 subprocess）"""

    def _run_cli(self, args: list[str], env: dict | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
        """辅助函数：跑 CLI 子进程"""
        if cwd is None:
            cwd = str(Path(__file__).resolve().parents[1])
        full_env = {**__import__("os").environ, "AIOPS_FORCE_FALLBACK": "1"}
        if env:
            full_env.update(env)
        return subprocess.run(
            [sys.executable, "-m", "app.eval.cli", *args],
            capture_output=True, text=True, env=full_env, cwd=cwd, timeout=30,
        )

    def test_cli_run_records(self, isolated_history):
        r = self._run_cli(["run", "--split", "dev", "--analyzer", "cli_test", "--note", "cli unit test"])
        assert r.returncode == 0
        # check db has the run
        rows = list_runs(split="dev")
        assert any(row["note"] == "cli unit test" for row in rows)

    def test_cli_history(self, isolated_history):
        # 先跑一次
        self._run_cli(["run", "--split", "dev", "--analyzer", "hist_test"])
        # 再 list
        r = self._run_cli(["history", "--split", "dev"])
        assert r.returncode == 0
        assert "hist_test" in r.stdout or "pass_rate" in r.stdout

    def test_cli_baseline_set_and_list(self, isolated_history):
        # 先跑一次
        self._run_cli(["run", "--split", "dev", "--analyzer", "bl_test"])
        # set baseline
        r = self._run_cli(["baseline", "set", "cli_baseline", "--split", "dev"])
        assert r.returncode == 0
        assert "Saved baseline" in r.stdout
        # list
        r = self._run_cli(["baseline", "list"])
        assert r.returncode == 0
        assert "cli_baseline" in r.stdout

    def test_cli_baseline_check_pass(self, isolated_history):
        self._run_cli(["run", "--split", "dev", "--analyzer", "bl_pass"])
        self._run_cli(["baseline", "set", "low_bar", "--split", "dev"])
        # run with --baseline low_bar (current pass_rate >= 0 应该过)
        r = self._run_cli(["run", "--split", "dev", "--baseline", "low_bar"])
        assert r.returncode == 0  # 应通过