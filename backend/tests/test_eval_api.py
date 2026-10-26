"""
AIOps Agent Platform - Evaluation API Tests

校验 /api/v1/evaluations 与 /api/v1/evaluations/run 路由接通
真实 EvaluationFramework 后行为正确。
"""

from __future__ import annotations

import pytest


@pytest.fixture
def framework_reset():
    """每个测试前后 reset framework 历史，避免污染"""
    from app.evaluation.core import get_evaluation_framework

    fw = get_evaluation_framework()
    fw.reset_history()
    yield fw
    fw.reset_history()


class TestEvaluationsAPI:
    """评测路由联通测试"""

    def test_get_evaluations_returns_real_history(
        self, framework_reset, client
    ) -> None:
        """先跑一次 /evaluations/run 注入数据，GET 必须返回真实历史（非 fixture）"""
        # 先跑一次
        run_resp = client.post("/api/v1/evaluations/run?eval_type=end_to_end")
        assert run_resp.status_code == 200

        # 再查
        resp = client.get("/api/v1/evaluations")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

        # 不应再出现硬编码 fixture 时间戳
        ts = body["items"][0]["timestamp"]
        assert ts != "2026-06-27T08:00:00Z", (
            f"Still returning hardcoded fixture timestamp: {ts}"
        )
        # eval_type 应是合法枚举值
        assert body["items"][0]["eval_type"] in (
            "end_to_end",
            "reasoning",
            "tool_call",
            "rag",
            "scenario",
        )

    def test_get_evaluations_filter_by_eval_type(self, client) -> None:
        """eval_type 过滤生效"""
        # 跑一个 end_to_end
        client.post("/api/v1/evaluations/run?eval_type=end_to_end")
        # 跑一个 reasoning
        client.post("/api/v1/evaluations/run?eval_type=reasoning")

        # 仅查 reasoning
        resp = client.get("/api/v1/evaluations?eval_type=reasoning")
        body = resp.json()
        for item in body["items"]:
            assert item["eval_type"] == "reasoning"

    def test_run_endpoint_returns_completed(
        self, framework_reset, client
    ) -> None:
        """POST /evaluations/run 必须同步返回 completed 状态 + 完整指标"""
        resp = client.post(
            "/api/v1/evaluations/run?eval_type=end_to_end"
        )
        assert resp.status_code == 200  # 同步返回，不再 202
        body = resp.json()
        assert body["status"] == "completed"
        assert "eval_id" in body
        assert "overall_score" in body
        assert isinstance(body["overall_score"], float)
        assert "metric_scores" in body

    def test_run_scenario_type_returns_per_scenario(
        self, framework_reset, client
    ) -> None:
        """POST /evaluations/run?eval_type=scenario 必须返回 19 条 per_scenario"""
        resp = client.post(
            "/api/v1/evaluations/run?eval_type=scenario"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["eval_type"] == "scenario"
        assert body["status"] == "completed"
        assert body["total_scenarios"] >= 19
        assert len(body["per_scenario"]) >= 19
        # per_scenario 每条必有 scenario_id / weighted_score
        for ps in body["per_scenario"]:
            assert "scenario_id" in ps
            assert "weighted_score" in ps
            assert "dimension_scores" in ps

    def test_run_records_history(self, framework_reset, client) -> None:
        """POST /evaluations/run 写入 _run_history"""
        client.post("/api/v1/evaluations/run?eval_type=end_to_end")
        history = framework_reset.get_history()
        assert len(history) >= 1
        assert history[0]["eval_type"] == "end_to_end"

    def test_run_persists_report_file(self, framework_reset, client) -> None:
        """POST /evaluations/run 调用 save_report 落盘"""
        import os
        import glob

        client.post("/api/v1/evaluations/run?eval_type=end_to_end")
        # 默认落盘到 /tmp/eval_reports
        files = glob.glob("/tmp/eval_reports/benchmark_report_*.json")
        assert len(files) >= 1, (
            "save_report must persist a file to /tmp/eval_reports"
        )

    def test_run_no_longer_returns_fake_eval_id(
        self, framework_reset, client
    ) -> None:
        """回归：不再返回 eval-run-xxxxxxxx 形式的 fake id"""
        resp = client.post("/api/v1/evaluations/run?eval_type=tool_call")
        body = resp.json()
        eval_id = body["eval_id"]
        # 应当是 uuid（36 字符）或 scenario-* 时间戳，不是 'eval-run-XXXXXXXX' 8位
        assert not eval_id.startswith("eval-run-"), (
            f"Still returning fake eval_id: {eval_id}"
        )

    def test_no_hardcoded_eval_001_in_response(self, client) -> None:
        """回归：响应里不应再出现硬编码的 EVAL-001 .. EVAL-004 id"""
        client.post("/api/v1/evaluations/run?eval_type=end_to_end")
        resp = client.get("/api/v1/evaluations")
        body = resp.json()
        ids = [item["id"] for item in body["items"]]
        for legacy_id in ("EVAL-001", "EVAL-002", "EVAL-003", "EVAL-004"):
            assert legacy_id not in ids, (
                f"Legacy fixture id {legacy_id} still present"
            )
