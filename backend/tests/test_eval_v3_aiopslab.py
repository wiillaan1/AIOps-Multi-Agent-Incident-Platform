"""v3 E8/E10/E11: AIOpsLab 对照 + badcase promote + CI gate 测试"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# 直接导入 eval.aiopslab_baseline（避免 src 布局依赖）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# data/adapters 模块在项目根（不在 backend/），需要加 path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.eval.aiopslab_baseline import (
    AIOPSLAB_LEADERBOARD,
    compare_pass_rate_with_leaderboard,
    count_aiopslab_problems_by_task,
    evaluate_aiopslab_static,
    get_all_leaderboard,
    get_leaderboard_score,
)
from app.eval.promote import (
    get_candidate,
    list_candidates,
    promote_candidate,
    reject_candidate,
)


class TestAIOpsLabBaseline:
    def test_leaderboard_has_known_agents(self):
        lb = get_all_leaderboard()
        assert "FLASH (GPT-4)" in lb
        assert "GPT-3.5 w Shell" in lb
        assert "OpsAgent (GPT-4)" in lb

    def test_leaderboard_score_format(self):
        score = get_leaderboard_score("FLASH (GPT-4)", "detection")
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))

    def test_unknown_agent_raises(self):
        with pytest.raises(KeyError):
            get_leaderboard_score("DoesNotExist", "detection")

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            get_leaderboard_score("FLASH (GPT-4)", "invalid_task")

    def test_count_problems_by_task(self):
        counts = count_aiopslab_problems_by_task()
        assert "detection" in counts
        assert "localization" in counts
        assert counts["detection"] > 0

    def test_compare_pass_rate_with_leaderboard(self):
        result = compare_pass_rate_with_leaderboard(0.6, "detection")
        assert "our_pass_rate" in result
        assert "rank" in result
        assert "verdict" in result
        # 0.6 = 60% → 比 FLASH (100%) 低，应该 rank > 1
        assert result["rank"] >= 1

    def test_evaluate_static_4_tasks(self):
        rates = {
            "detection": 0.9,
            "localization": 0.5,
            "diagnosis": 0.4,
            "mitigation": 0.5,
        }
        result = evaluate_aiopslab_static(rates)
        assert "per_task" in result
        assert "overall_average" in result
        assert len(result["per_task"]) == 4

    def test_evaluate_static_hand_to_analysis_alias(self):
        """'analysis' 别名应映射到 'diagnosis'（跟 leaderboard 字段对齐）"""
        rates = {"analysis": 0.5, "detection": 0.9}
        result = evaluate_aiopslab_static(rates)
        # analysis 应该被 rename 成 diagnosis
        assert "diagnosis" in result["per_task"]


class TestBadcasePromote:
    @pytest.fixture
    def candidate_file(self, tmp_path: Path, monkeypatch):
        """写入一个候选 badcase 文件"""
        # 用环境变量覆盖 promote.py 的路径解析
        monkeypatch.setenv("BADCASES_DIR_OVERRIDE", str(tmp_path))

        candidate = {
            "id": "bc-captured-001",
            "category": "single_metric",
            "badcase_class": "rca_predict_mismatch",
            "type": "cpu_high",
            "alert": {"service": "web", "metric_name": "cpu_usage", "value": 95.0},
            "evidence": {"history": [80, 82, 95]},
            "identified_flaw": "RCA predicted disk but actual was cpu",
            "keywords_for_retrieval": ["cpu", "web"],
            "severity": "medium",
            "tags": ["auto_capture"],
        }
        (tmp_path / "candidates.jsonl").write_text(
            json.dumps(candidate, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return tmp_path, candidate

    def test_list_candidates(self, candidate_file):
        tmp_path, candidate = candidate_file
        cands = list_candidates(tmp_path / "candidates.jsonl")
        assert len(cands) == 1
        assert cands[0]["id"] == "bc-captured-001"

    def test_get_candidate(self, candidate_file):
        tmp_path, _ = candidate_file
        c = get_candidate("bc-captured-001", tmp_path / "candidates.jsonl")
        assert c is not None
        assert c["badcase_class"] == "rca_predict_mismatch"

    def test_get_candidate_not_found(self, candidate_file):
        tmp_path, _ = candidate_file
        c = get_candidate("does-not-exist", tmp_path / "candidates.jsonl")
        assert c is None

    def test_promote_candidate_writes_to_dev(self, candidate_file):
        tmp_path, _ = candidate_file
        result = promote_candidate(
            "bc-captured-001",
            expected_root_cause="cpu_high_due_to_traffic_spike",
            dest_split="dev",
            candidates_path=tmp_path / "candidates.jsonl",
        )
        assert result["dest_split"] == "dev"
        assert result["expected_root_cause"] == "cpu_high_due_to_traffic_spike"

        # verify file exists and contains the badcase
        v1_dev = tmp_path / "v1_dev.json"
        assert v1_dev.exists()
        payload = json.loads(v1_dev.read_text(encoding="utf-8"))
        ids = [b["id"] for b in payload["badcases"]]
        assert "bc-captured-001" in ids

    def test_promote_candidate_invalid_split(self, candidate_file):
        tmp_path, _ = candidate_file
        with pytest.raises(ValueError):
            promote_candidate(
                "bc-captured-001",
                expected_root_cause="x",
                dest_split="invalid_split",
                candidates_path=tmp_path / "candidates.jsonl",
            )

    def test_promote_candidate_missing_entry(self, candidate_file):
        tmp_path, _ = candidate_file
        with pytest.raises(FileNotFoundError):
            promote_candidate(
                "does-not-exist",
                expected_root_cause="x",
                candidates_path=tmp_path / "candidates.jsonl",
            )

    def test_promote_duplicate_raises(self, candidate_file):
        tmp_path, _ = candidate_file
        # promote once
        promote_candidate(
            "bc-captured-001",
            expected_root_cause="x",
            candidates_path=tmp_path / "candidates.jsonl",
        )
        # promote again → 应该 raise
        with pytest.raises(ValueError, match="already exists"):
            promote_candidate(
                "bc-captured-001",
                expected_root_cause="x",
                candidates_path=tmp_path / "candidates.jsonl",
            )

    def test_reject_candidate(self, candidate_file):
        tmp_path, _ = candidate_file
        ok = reject_candidate(
            "bc-captured-001",
            candidates_path=tmp_path / "candidates.jsonl",
            reject_log_path=tmp_path / "rejected.jsonl",
        )
        assert ok is True
        assert (tmp_path / "rejected.jsonl").exists()


class TestEvalGateScript:
    """本地 eval_gate.py 脚本测试"""

    def _run_gate(self, *args: str) -> subprocess.CompletedProcess:
        """跑 eval_gate 脚本"""
        gate = Path(__file__).resolve().parents[2] / "scripts" / "eval_gate.py"
        full_env = {**__import__("os").environ, "AIOPS_FORCE_FALLBACK": "1"}
        return subprocess.run(
            [sys.executable, str(gate), *args],
            capture_output=True, text=True, env=full_env,
            timeout=60,
        )

    def test_eval_gate_first_run_creates_baseline(self, tmp_path, monkeypatch):
        # 隔离 history 到 tmp_path
        monkeypatch.setenv("EVAL_HISTORY_DB_PATH", str(tmp_path / "history.db"))
        monkeypatch.setenv("EVAL_BASELINES_DIR", str(tmp_path / "baselines"))
        # 第一次跑：应该创建 baseline
        result = self._run_gate("--split", "dev", "--baseline", "test_baseline")
        assert result.returncode == 0
        assert "Creating baseline" in result.stdout
        # baseline 文件应该存在
        assert (tmp_path / "baselines" / "test_baseline.json").exists()

    def test_eval_gate_second_run_passes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVAL_HISTORY_DB_PATH", str(tmp_path / "history.db"))
        monkeypatch.setenv("EVAL_BASELINES_DIR", str(tmp_path / "baselines"))
        # 第一次创建 baseline
        self._run_gate("--split", "dev", "--baseline", "test_baseline_2")
        # 第二次应该走 check 路径
        result = self._run_gate("--split", "dev", "--baseline", "test_baseline_2")
        # 第二次 pass_rate >= baseline → 应通过
        assert result.returncode == 0