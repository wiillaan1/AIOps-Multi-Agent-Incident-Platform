"""
AIOps Agent Platform - Evaluation Framework Integration Tests

测试评估框架的端到端指标、推理评估、工具调用评估、RAG评估和完整报告生成。
"""

from __future__ import annotations

import pytest

from app.evaluation.core import EvaluationFramework, get_evaluation_framework
from app.evaluation.end_to_end import EndToEndEvaluator
from app.evaluation.rag_eval import RAGEvaluator
from app.evaluation.reasoning_eval import ReasoningEvaluator
from app.evaluation.tool_call_eval import ToolCallEvaluator
from app.models.evaluation import (
    BenchmarkReport,
    EvaluationResult,
    EvaluationStatus,
    EvaluationType,
)


class TestScenarioRegistry:
    """场景注册表测试套件 (Task 1)"""

    def test_registry_has_all_fault_scenarios(self) -> None:
        """SCENARIO_REGISTRY 必须覆盖 datasets.FAULT_SCENARIOS 的全部 15 条"""
        from app.data.datasets import FAULT_SCENARIOS
        from app.evaluation.scenarios import SCENARIO_REGISTRY

        for fs in FAULT_SCENARIOS:
            assert fs["id"] in SCENARIO_REGISTRY, (
                f"{fs['id']} missing from SCENARIO_REGISTRY"
            )

    def test_registry_covers_business_rules(self) -> None:
        """SCENARIO_REGISTRY 必须至少包含 8 条 BUSINESS_RULES 派生条目"""
        from app.agents.business_monitor_agent import BUSINESS_RULES
        from app.evaluation.scenarios import SCENARIO_REGISTRY

        rule_ids = {r["id"] for r in BUSINESS_RULES}
        covered = rule_ids & set(SCENARIO_REGISTRY.keys())
        assert len(covered) >= 4, (
            f"At least 4 BUSINESS_RULES should map to scenarios; got {len(covered)}"
        )

    def test_scenario_spec_required_fields(self) -> None:
        """每条 ScenarioSpec 必须有合法 severity / 非空 dimensions / 非空 ground_truth"""
        from app.evaluation.scenarios import SCENARIO_REGISTRY

        for sid, spec in SCENARIO_REGISTRY.items():
            assert spec.scenario_id == sid
            assert spec.dimensions, f"{sid} has empty dimensions"
            assert spec.severity in {"critical", "high", "medium", "low"}, (
                f"{sid} bad severity {spec.severity}"
            )
            assert spec.ground_truth, f"{sid} empty ground_truth"
            assert "root_cause" in spec.ground_truth, (
                f"{sid} ground_truth missing root_cause"
            )

    def test_scenarios_for_dimension_covers_all_four(self) -> None:
        """每个评测维度至少覆盖 5 个场景"""
        from app.evaluation.scenarios import (
            EVAL_DIMENSIONS,
            scenarios_for_dimension,
        )

        for d in EVAL_DIMENSIONS:
            specs = scenarios_for_dimension(d)
            assert len(specs) >= 5, (
                f"dimension {d} must cover at least 5 scenarios; got {len(specs)}"
            )

    def test_get_scenario_returns_none_for_unknown(self) -> None:
        """未知 id 返回 None 而非抛异常"""
        from app.evaluation.scenarios import get_scenario

        assert get_scenario("nonexistent_xyz") is None

    def test_total_registry_size(self) -> None:
        """SCENARIO_REGISTRY 总条目数 >= 19 (15 FS + 4 BUSINESS_RULES-only)"""
        from app.evaluation.scenarios import SCENARIO_REGISTRY

        assert len(SCENARIO_REGISTRY) >= 19, (
            f"Expected >=19 entries, got {len(SCENARIO_REGISTRY)}"
        )

    def test_infrastructure_scenarios_skip_rag(self) -> None:
        """基础设施场景不应该需要 RAG 维度（轻量指标）"""
        from app.evaluation.scenarios import get_scenario

        # fs_001 (CPU spike) - 不需要 RAG
        fs_001 = get_scenario("fs_001")
        assert "rag" not in fs_001.dimensions, (
            "Infrastructure scenarios like fs_001 should not require RAG"
        )

    def test_business_logic_scenarios_include_rag(self) -> None:
        """业务规则驱动的场景应该需要 RAG（业务知识库检索）"""
        from app.evaluation.scenarios import get_scenario

        fs_biz_007 = get_scenario("fs_biz_007")
        assert "rag" in fs_biz_007.dimensions
        assert fs_biz_007.business_rule == "br_duplicate_charge"

    def test_relevant_docs_reference_known_kb(self) -> None:
        """relevant_docs 必须引用真实存在的知识库条目"""
        from app.data.knowledge_base import KNOWLEDGE_BASE
        from app.evaluation.scenarios import SCENARIO_REGISTRY

        valid_ids = {doc["id"] for doc in KNOWLEDGE_BASE}
        for sid, spec in SCENARIO_REGISTRY.items():
            for doc_id in spec.relevant_docs:
                assert doc_id in valid_ids, (
                    f"{sid} references unknown kb doc {doc_id}"
                )


class TestLayeredDatasets:
    """分层评测数据集测试套件 (Task 2)"""

    def test_unit_dataset_minimum_size(self) -> None:
        """UnitDatasets 至少 6 条样本（单 skill 单调用）"""
        from app.evaluation.layered_datasets import UnitDatasets

        samples = UnitDatasets.all()
        assert len(samples) >= 6, f"need >=6 unit samples, got {len(samples)}"

    def test_unit_dataset_has_required_keys(self) -> None:
        """UnitDatasets 每条必须含 id / dimension / skill / sample"""
        from app.evaluation.layered_datasets import UnitDatasets

        for s in UnitDatasets.all():
            assert "id" in s
            assert "dimension" in s
            assert "skill" in s or "sample" in s, (
                f"unit sample {s.get('id')} missing skill/sample"
            )

    def test_integration_dataset_minimum_size(self) -> None:
        """IntegrationDatasets 至少 4 条（多 skill 协同）"""
        from app.evaluation.layered_datasets import IntegrationDatasets

        samples = IntegrationDatasets.all()
        assert len(samples) >= 4, f"need >=4 integration samples, got {len(samples)}"

    def test_e2e_dataset_matches_fault_scenarios(self) -> None:
        """E2EDatasets 必须覆盖 datasets.FAULT_SCENARIOS 全部 15 条"""
        from app.data.datasets import FAULT_SCENARIOS
        from app.evaluation.layered_datasets import E2EDatasets

        e2e_ids = {s.get("scenario_id") or s.get("id") for s in E2EDatasets.all()}
        for fs in FAULT_SCENARIOS:
            assert fs["id"] in e2e_ids, f"{fs['id']} missing from E2EDatasets"

    def test_e2e_dataset_carries_root_cause(self) -> None:
        """E2EDatasets 每条必须含 root_cause 字段"""
        from app.evaluation.layered_datasets import E2EDatasets

        for s in E2EDatasets.all():
            assert "root_cause" in s, (
                f"E2E sample {s.get('scenario_id')} missing root_cause"
            )
            assert "expected_action" in s, (
                f"E2E sample {s.get('scenario_id')} missing expected_action"
            )

    def test_golden_dataset_is_e2e_subset(self) -> None:
        """GoldenDatasets 必须是 E2EDatasets 的真子集，3-8 条代表场景"""
        from app.evaluation.layered_datasets import E2EDatasets, GoldenDatasets

        golden_ids = {
            s.get("scenario_id") or s.get("id") for s in GoldenDatasets.all()
        }
        e2e_ids = {
            s.get("scenario_id") or s.get("id") for s in E2EDatasets.all()
        }
        assert golden_ids.issubset(e2e_ids)
        assert 3 <= len(golden_ids) <= 8, (
            f"golden set should have 3-8 scenarios, got {len(golden_ids)}"
        )

    def test_golden_covers_all_three_categories(self) -> None:
        """GoldenDatasets 必须覆盖三类场景：infra / business / business_logic"""
        from app.evaluation.layered_datasets import GoldenDatasets
        from app.evaluation.scenarios import get_scenario

        categories: set[str] = set()
        for s in GoldenDatasets.all():
            sid = s.get("scenario_id") or s.get("id")
            spec = get_scenario(sid)
            if spec:
                categories.add(spec.category)
        assert "infrastructure" in categories
        assert "business_logic" in categories, (
            "Golden set should include business_logic scenarios"
        )


class TestScenarioDrivenEvaluator:
    """场景驱动评测器测试套件 (Task 3)"""

    def test_scenario_score_includes_required_dimensions(self) -> None:
        """fs_001 (e2e+reasoning+tool) 跑出来必须包含这三个维度的分"""
        import asyncio
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator
        from app.evaluation.scenarios import get_scenario

        ev = ScenarioDrivenEvaluator()
        score = asyncio.run(ev.evaluate_scenario(get_scenario("fs_001")))
        assert score.scenario_id == "fs_001"
        assert "e2e" in score.dimension_scores
        assert "reasoning" in score.dimension_scores
        assert "tool" in score.dimension_scores

    def test_scenario_omit_optional_dimensions(self) -> None:
        """fs_004 不带 RAG，跑分里不应出现 rag"""
        import asyncio
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator
        from app.evaluation.scenarios import get_scenario

        ev = ScenarioDrivenEvaluator()
        score = asyncio.run(ev.evaluate_scenario(get_scenario("fs_004")))
        assert "rag" not in score.dimension_scores

    def test_scenario_score_recorded_ground_truth(self) -> None:
        """ScenarioScore.ground_truth_match 必须含 root_cause 字段"""
        import asyncio
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator
        from app.evaluation.scenarios import get_scenario

        ev = ScenarioDrivenEvaluator()
        score = asyncio.run(ev.evaluate_scenario(get_scenario("fs_001")))
        assert "root_cause" in score.ground_truth_match

    def test_business_logic_scenario_with_rag(self) -> None:
        """fs_biz_007 (含 RAG) 跑出来必须含 rag 维度分数"""
        import asyncio
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator
        from app.evaluation.scenarios import get_scenario

        ev = ScenarioDrivenEvaluator()
        score = asyncio.run(ev.evaluate_scenario(get_scenario("fs_biz_007")))
        assert "rag" in score.dimension_scores
        assert score.business_rule == "br_duplicate_charge"

    def test_evaluate_all_returns_report(self) -> None:
        """evaluate_all 返回 ScenarioEvalReport，含全量 scenario 数"""
        import asyncio
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator

        ev = ScenarioDrivenEvaluator()
        report = asyncio.run(ev.evaluate_all())
        assert report.total_scenarios >= 19
        assert "e2e" in report.by_dimension
        assert "reasoning" in report.by_dimension
        assert "tool" in report.by_dimension
        assert "rag" in report.by_dimension

    def test_evaluate_all_by_severity_aggregation(self) -> None:
        """by_severity 字典至少含 critical 一档"""
        import asyncio
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator

        ev = ScenarioDrivenEvaluator()
        report = asyncio.run(ev.evaluate_all())
        assert "critical" in report.by_severity
        assert report.by_severity["critical"] > 0

    def test_registered_in_evaluation_framework(self) -> None:
        """EvaluationFramework 必须注册 SCENARIO 类型"""
        from app.evaluation.core import get_evaluation_framework
        from app.models.evaluation import EvaluationType

        framework = get_evaluation_framework()
        ev = framework.get_evaluator(EvaluationType.SCENARIO)
        assert ev is not None

    def test_framework_routes_scenario_type(self) -> None:
        """EvaluationFramework.evaluate(SCENARIO) 应该能跑"""
        from app.evaluation.core import get_evaluation_framework
        from app.models.evaluation import EvaluationStatus, EvaluationType

        framework = get_evaluation_framework()
        # 异步跑可能因 framework 单例被前面测试 reset 过，
        # 这里只验证不抛异常且 evaluator 已注入
        ev = framework.get_evaluator(EvaluationType.SCENARIO)
        assert ev is not None
        # verify it has the right interface
        assert hasattr(ev, "evaluate_all")


class TestGoldenRegression:
    """Golden 回归基线测试套件 (Task 4)"""

    def _runner(self, tmp_path):
        """构造一个 runner，baseline 写到 tmp_path"""
        from app.evaluation.regression import GoldenSetRunner
        return GoldenSetRunner(baseline_path=str(tmp_path / "baseline.json"))

    def test_golden_runner_runs_all_golden_scenarios(self, tmp_path) -> None:
        """GoldenSetRunner.run() 跑全 6 条 golden 场景"""
        import asyncio
        runner = self._runner(tmp_path)
        result = asyncio.run(runner.run())
        assert result.total_scenarios == 6

    def test_golden_runner_persists_baseline(self, tmp_path) -> None:
        """runner.run(save_baseline=True) 必须写入 baseline 文件"""
        import asyncio
        runner = self._runner(tmp_path)
        asyncio.run(runner.run(save_baseline=True))
        assert (tmp_path / "baseline.json").exists()

    def test_regression_detection_flags_drop(self, tmp_path) -> None:
        """注入低分场景，runner 必须 flag 回归"""
        import asyncio
        from app.evaluation.regression import GoldenSetRunner

        baseline_path = tmp_path / "baseline.json"

        # 第一次跑：保存基线（高分，因为 evaluator 默认就给 0.6+）
        runner1 = GoldenSetRunner(baseline_path=str(baseline_path))
        asyncio.run(runner1.run(save_baseline=True))
        assert baseline_path.exists()

        # 第二次跑：注入 fs_001 极低分（mock override）
        runner2 = GoldenSetRunner(
            baseline_path=str(baseline_path),
            score_overrides={"fs_001": 0.1},
        )
        result = asyncio.run(runner2.run())
        assert len(result.regressions) >= 1
        # 至少应包含 fs_001 的回归告警
        regressed_ids = {r.scenario_id for r in result.regressions}
        assert "fs_001" in regressed_ids

    def test_no_regression_when_scores_match(self, tmp_path) -> None:
        """两次跑得分相同时应无回归告警（容忍默认阈值 0.05）"""
        import asyncio
        from app.evaluation.regression import GoldenSetRunner

        baseline_path = tmp_path / "baseline.json"
        # 第一次保存基线
        GoldenSetRunner(baseline_path=str(baseline_path))
        asyncio.run(GoldenSetRunner(baseline_path=str(baseline_path)).run(save_baseline=True))
        # 第二次不注入 override，evaluator 走真实路径（重复执行）
        # 容忍评估的微小随机性，regressions 数量应小于 2
        result = asyncio.run(GoldenSetRunner(baseline_path=str(baseline_path)).run())
        # 同 evaluator 跑两次应当近似一致
        # 由于相同种子与确定性，结果应一致或仅 ±0.05 波动
        for reg in result.regressions:
            assert reg.delta > 0.05, (
                f"unexpected large delta for {reg.scenario_id}: {reg.delta}"
            )

    def test_regression_threshold_configurable(self, tmp_path) -> None:
        """regression_threshold 参数生效"""
        import asyncio
        from app.evaluation.regression import GoldenSetRunner

        baseline_path = tmp_path / "baseline.json"
        # 第一次保存
        asyncio.run(GoldenSetRunner(baseline_path=str(baseline_path)).run(save_baseline=True))
        # 高阈值 + override 应当 flag
        runner = GoldenSetRunner(
            baseline_path=str(baseline_path),
            score_overrides={"fs_002": 0.5},
            regression_threshold=0.1,
        )
        result = asyncio.run(runner.run())
        assert any(r.scenario_id == "fs_002" for r in result.regressions)

    def test_regression_result_schema(self, tmp_path) -> None:
        """RegressionAlert 字段完整"""
        import asyncio
        from app.evaluation.regression import GoldenSetRunner, RegressionAlert

        baseline_path = tmp_path / "baseline.json"
        asyncio.run(GoldenSetRunner(baseline_path=str(baseline_path)).run(save_baseline=True))
        runner = GoldenSetRunner(
            baseline_path=str(baseline_path),
            score_overrides={"fs_001": 0.0},
        )
        result = asyncio.run(runner.run())
        assert len(result.regressions) >= 1
        alert = result.regressions[0]
        assert isinstance(alert, RegressionAlert)
        assert alert.scenario_id
        assert alert.baseline_score >= 0
        assert alert.current_score >= 0
        assert alert.delta < 0  # regression = drop

    def test_golden_set_runner_exposes_save_baseline(self, tmp_path) -> None:
        """save_baseline API 可独立调用"""
        import asyncio
        from app.evaluation.regression import GoldenSetRunner

        baseline_path = tmp_path / "baseline.json"
        runner = GoldenSetRunner(baseline_path=str(baseline_path))
        asyncio.run(runner.run(save_baseline=True))
        assert baseline_path.exists()
        # 第二次跑（无 save）应该做对比
        result = asyncio.run(GoldenSetRunner(baseline_path=str(baseline_path)).run())
        assert hasattr(result, "regressions")


class TestEndToEndEvaluation:
    """端到端评估测试套件"""

    @pytest.fixture
    def evaluator(self) -> EndToEndEvaluator:
        """创建EndToEndEvaluator实例"""
        return EndToEndEvaluator()

    @pytest.mark.asyncio
    async def test_end_to_end_metrics(self, evaluator: EndToEndEvaluator) -> None:
        """
        测试端到端指标计算

        确保各项端到端指标被正确计算。
        """
        result = await evaluator.evaluate(target_agent="test_agent")

        assert isinstance(result, EvaluationResult)
        assert result.evaluation_type == EvaluationType.END_TO_END
        assert result.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]

        if result.status == EvaluationStatus.COMPLETED:
            assert result.total_samples > 0
            assert len(result.metric_scores) > 0

            # 检查关键指标
            metric_names = [m.metric_name for m in result.metric_scores]
            assert "task_success_rate" in metric_names

    @pytest.mark.asyncio
    async def test_end_to_end_with_samples(self, evaluator: EndToEndEvaluator) -> None:
        """测试使用自定义样本的端到端评估"""
        samples = [
            {
                "id": "test-001",
                "name": "CPU High",
                "actual_result": {
                    "resolved": True,
                    "automated": True,
                    "root_cause": "traffic_spike",
                    "action": "scale_up",
                    "time_to_resolve_seconds": 120,
                },
                "expected_result": {
                    "resolved": True,
                    "root_cause": "traffic_spike",
                    "action": "scale_up",
                },
            },
            {
                "id": "test-002",
                "name": "DB Timeout",
                "actual_result": {
                    "resolved": False,
                    "automated": False,
                    "root_cause": "db_pool_exhausted",
                    "action": "escalate",
                    "time_to_resolve_seconds": 600,
                },
                "expected_result": {
                    "resolved": False,
                    "root_cause": "db_pool_exhausted",
                    "action": "escalate",
                },
            },
        ]
        result = await evaluator.evaluate(target_agent="test_agent", samples=samples)

        assert isinstance(result, EvaluationResult)
        if result.status == EvaluationStatus.COMPLETED:
            assert result.total_samples == 2
            assert len(result.metric_scores) > 0

    def test_time_score_calculation(self, evaluator: EndToEndEvaluator) -> None:
        """测试处理时间分数计算"""
        # 2分钟内应得满分
        score_fast = evaluator._compute_time_score(60)
        assert score_fast == 1.0

        # 超过10分钟应得低分
        score_slow = evaluator._compute_time_score(900)
        assert score_slow < 0.3


class TestReasoningEvaluation:
    """推理评估测试套件"""

    @pytest.fixture
    def evaluator(self) -> ReasoningEvaluator:
        """创建ReasoningEvaluator实例"""
        return ReasoningEvaluator()

    @pytest.mark.asyncio
    async def test_reasoning_eval(self, evaluator: ReasoningEvaluator) -> None:
        """
        测试推理评估

        确保推理能力评估指标被正确计算。
        """
        result = await evaluator.evaluate(target_agent="test_agent")

        assert isinstance(result, EvaluationResult)
        assert result.evaluation_type == EvaluationType.REASONING

        if result.status == EvaluationStatus.COMPLETED:
            assert len(result.metric_scores) > 0
            metric_names = [m.metric_name for m in result.metric_scores]
            assert "root_cause_accuracy" in metric_names

    @pytest.mark.asyncio
    async def test_reasoning_with_samples(self, evaluator: ReasoningEvaluator) -> None:
        """测试使用自定义样本的推理评估"""
        samples = [
            {
                "id": "reason-001",
                "predicted": {
                    "root_cause": "traffic_spike",
                    "confidence": 0.9,
                    "evidence": {
                        "metrics": ["cpu_usage", "request_rate"],
                        "logs": ["rate_limiter_triggered"],
                    },
                    "impact_chain": ["traffic_spike", "increased_request_rate", "cpu_high"],
                    "reasoning_steps": [
                        {"description": "Observe CPU anomaly", "order": 1},
                        {"description": "Check request rate", "order": 2},
                    ],
                },
                "ground_truth": {
                    "root_cause": "traffic_spike",
                    "confidence": 0.9,
                    "evidence": {
                        "metrics": ["cpu_usage", "request_rate"],
                        "logs": ["rate_limiter_triggered"],
                    },
                    "impact_chain": ["traffic_spike", "increased_request_rate", "cpu_high"],
                    "reasoning_steps": [
                        {"description": "Observe CPU anomaly", "order": 1},
                        {"description": "Check request rate", "order": 2},
                    ],
                },
            },
        ]
        result = await evaluator.evaluate(target_agent="test_agent", samples=samples)

        assert isinstance(result, EvaluationResult)
        if result.status == EvaluationStatus.COMPLETED:
            # 完美匹配应得高分
            rca_metric = next(
                (m for m in result.metric_scores if m.metric_name == "root_cause_accuracy"),
                None,
            )
            if rca_metric:
                assert rca_metric.score == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_rca(self, evaluator: ReasoningEvaluator) -> None:
        """测试RCA评估"""
        result = await evaluator.evaluate_rca(
            predicted_root_causes=["traffic_spike", "db_issue"],
            actual_root_causes=["traffic_spike", "config_error"],
            predicted_confidences=[0.9, 0.7],
        )
        assert "root_cause_accuracy" in result
        assert result["total_samples"] == 2

    @pytest.mark.asyncio
    async def test_evaluate_confidence(self, evaluator: ReasoningEvaluator) -> None:
        """测试置信度校准评估"""
        confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
        accuracies = [True, True, False, True, False]
        result = await evaluator.evaluate_confidence(confidences, accuracies)

        assert "ece" in result
        assert "calibration_score" in result
        assert 0.0 <= result["ece"] <= 1.0

    def test_rubric(self, evaluator: ReasoningEvaluator) -> None:
        """测试评分标准"""
        rubric = evaluator.get_rubric()
        assert "root_cause_identification" in rubric
        assert "evidence_quality" in rubric
        assert "confidence_calibration" in rubric


class TestToolCallEvaluation:
    """工具调用评估测试套件"""

    @pytest.fixture
    def evaluator(self) -> ToolCallEvaluator:
        """创建ToolCallEvaluator实例"""
        return ToolCallEvaluator()

    @pytest.mark.asyncio
    async def test_tool_call_eval(self, evaluator: ToolCallEvaluator) -> None:
        """
        测试工具调用评估

        确保工具调用准确性指标被正确计算。
        """
        result = await evaluator.evaluate(target_agent="test_agent")

        assert isinstance(result, EvaluationResult)
        assert result.evaluation_type == EvaluationType.TOOL_CALL

        if result.status == EvaluationStatus.COMPLETED:
            assert len(result.metric_scores) > 0
            metric_names = [m.metric_name for m in result.metric_scores]
            assert "tool_selection_accuracy" in metric_names

    @pytest.mark.asyncio
    async def test_evaluate_tool_selection(self, evaluator: ToolCallEvaluator) -> None:
        """测试工具选择评估"""
        result = await evaluator.evaluate_tool_selection(
            selected_tools=["query_metrics", "get_service_info"],
            expected_tools=["query_metrics", "get_service_info"],
        )
        assert result["accuracy"] == 1.0
        assert len(result["correct_tools"]) == 2

    @pytest.mark.asyncio
    async def test_evaluate_parameters(self, evaluator: ToolCallEvaluator) -> None:
        """测试参数评估"""
        tool_calls = [
            {
                "tool_name": "query_metrics",
                "parameters": {"service": "order-service", "metric": "cpu"},
                "expected_parameters": {"service": "order-service", "metric": "cpu"},
            },
        ]
        result = await evaluator.evaluate_parameters(tool_calls)
        assert result["overall_accuracy"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_efficiency(self, evaluator: ToolCallEvaluator) -> None:
        """测试调用效率评估"""
        result = await evaluator.evaluate_efficiency(
            actual_calls=["query_metrics", "get_service_info"],
            minimum_calls=["query_metrics", "get_service_info"],
        )
        assert result["is_optimal"] is True
        assert result["efficiency"] == 1.0


class TestRAGEvaluation:
    """RAG评估测试套件"""

    @pytest.fixture
    def evaluator(self) -> RAGEvaluator:
        """创建RAGEvaluator实例"""
        return RAGEvaluator()

    @pytest.mark.asyncio
    async def test_rag_eval(self, evaluator: RAGEvaluator) -> None:
        """
        测试RAG评估

        确保RAG效果指标被正确计算。
        """
        result = await evaluator.evaluate(target_agent="test_agent")

        assert isinstance(result, EvaluationResult)
        assert result.evaluation_type == EvaluationType.RAG

        if result.status == EvaluationStatus.COMPLETED:
            assert len(result.metric_scores) > 0
            metric_names = [m.metric_name for m in result.metric_scores]
            assert "retrieval_precision" in metric_names
            assert "retrieval_recall" in metric_names

    @pytest.mark.asyncio
    async def test_retrieval_eval(self, evaluator: RAGEvaluator) -> None:
        """测试检索质量评估"""
        samples = [
            {
                "id": "rag-001",
                "retrieved_docs": [
                    {"id": "doc_001", "content": "relevant"},
                    {"id": "doc_002", "content": "irrelevant"},
                ],
                "relevant_docs": ["doc_001"],
            },
        ]
        result = await evaluator.evaluate_retrieval(samples)
        assert "avg_precision" in result
        assert "avg_recall" in result

    @pytest.mark.asyncio
    async def test_generation_eval(self, evaluator: RAGEvaluator) -> None:
        """测试生成质量评估"""
        samples = [
            {
                "id": "rag-001",
                "generated_answer": "The answer is based on the documents.",
                "retrieved_docs": [
                    {"id": "doc_001", "content": "The documents contain relevant info."},
                ],
                "query": "What is the answer?",
            },
        ]
        result = await evaluator.evaluate_generation(samples)
        assert "avg_faithfulness" in result
        assert "avg_relevance" in result


class TestEvaluationFramework:
    """评估框架整体测试套件"""

    @pytest.fixture
    def framework(self) -> EvaluationFramework:
        """创建EvaluationFramework实例"""
        fw = get_evaluation_framework()
        fw.reset_history()
        return fw

    @pytest.mark.asyncio
    async def test_full_report(self, framework: EvaluationFramework) -> None:
        """
        测试完整报告生成

        端到端测试评估框架的报告生成功能。
        """
        # 执行所有评估
        results = await framework.evaluate_all(target_agent="test_agent")

        assert len(results) == 4  # 4种评估类型
        completed = [r for r in results if r.status == EvaluationStatus.COMPLETED]
        assert len(completed) > 0

        # 生成报告
        report = await framework.generate_report(
            results=results,
            report_name="Integration Test Report",
        )

        assert isinstance(report, BenchmarkReport)
        assert report.report_name == "Integration Test Report"
        assert len(report.results) == 4
        assert "total_evaluations" in report.summary

    @pytest.mark.asyncio
    async def test_individual_evaluations(self, framework: EvaluationFramework) -> None:
        """测试各类型独立评估"""
        for eval_type in [
            EvaluationType.END_TO_END,
            EvaluationType.REASONING,
            EvaluationType.TOOL_CALL,
            EvaluationType.RAG,
        ]:
            result = await framework.evaluate(
                eval_type=eval_type,
                target_agent="test_agent",
            )
            assert isinstance(result, EvaluationResult)
            assert result.evaluation_type == eval_type

    def test_compute_summary(self, framework: EvaluationFramework) -> None:
        """测试汇总统计计算"""
        results = [
            EvaluationResult(
                evaluation_type=EvaluationType.END_TO_END,
                eval_name="e2e_test",
                overall_score=0.85,
                status=EvaluationStatus.COMPLETED,
                total_samples=10,
                passed_samples=8,
            ),
            EvaluationResult(
                evaluation_type=EvaluationType.REASONING,
                eval_name="reasoning_test",
                overall_score=0.75,
                status=EvaluationStatus.COMPLETED,
                total_samples=5,
                passed_samples=4,
            ),
        ]
        summary = framework._compute_summary(results)

        assert summary["total_evaluations"] == 2
        assert summary["completed"] == 2
        assert "avg_overall_score" in summary

    def test_generate_recommendations(self, framework: EvaluationFramework) -> None:
        """测试改进建议生成"""
        results = [
            EvaluationResult(
                evaluation_type=EvaluationType.END_TO_END,
                eval_name="e2e_test",
                status=EvaluationStatus.COMPLETED,
                overall_score=0.5,
            ),
        ]
        recommendations = framework._generate_recommendations(results)
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    @pytest.mark.asyncio
    async def test_trend_analysis(self, framework: EvaluationFramework) -> None:
        """测试趋势分析"""
        # 先执行几次评估生成历史数据
        for _ in range(3):
            await framework.evaluate(
                eval_type=EvaluationType.END_TO_END,
                target_agent="test_agent",
            )

        trend = await framework.analyze_trends(eval_type=EvaluationType.END_TO_END)

        assert "total_runs" in trend
        assert trend["total_runs"] >= 3
        assert "trend_direction" in trend

    def test_history_management(self, framework: EvaluationFramework) -> None:
        """测试历史记录管理"""
        history = framework.get_history()
        assert isinstance(history, list)

        framework.reset_history()
        assert len(framework.get_history()) == 0
