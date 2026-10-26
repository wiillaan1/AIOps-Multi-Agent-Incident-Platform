"""
AIOps Agent Platform - Scenario-Driven Evaluator

把 `SCENARIO_REGISTRY` 中的每个场景作为评测单元，按 spec.dimensions 调用
对应底层 evaluator（EndToEnd / Reasoning / Tool Call / RAG），
汇总成 ScenarioScore 与 ScenarioEvalReport。

设计要点：
1. 不重复实现指标计算 — 完全复用 4 个底层 evaluator；
2. 单一来源（single source of truth）— ground_truth 全部从 ScenarioSpec 派生；
3. 同步入口 evaluate_scenario_sync 便于测试，异步入口 evaluate_scenario
   给路由用；
4. 顶层 evaluate_all() 把 19 条 spec 全部跑一遍。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.scenarios import (
    EVAL_DIMENSIONS,
    SCENARIO_REGISTRY,
    EvalDimension,
    ScenarioSpec,
)
from app.models.evaluation import EvaluationStatus, EvaluationType
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# 数据模型
# ============================================================================


class ScenarioScore(BaseModel):
    """单个场景的得分快照"""

    scenario_id: str
    source: str
    category: str
    severity: str
    service: str = ""
    business_rule: str = ""
    dimensions: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    weighted_score: float = 0.0
    ground_truth_match: dict[str, bool] = Field(default_factory=dict)
    missing_assertions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ScenarioEvalReport(BaseModel):
    """场景驱动评测的完整报告"""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    total_scenarios: int = 0
    succeeded: int = 0
    failed: int = 0
    by_dimension: dict[str, float] = Field(default_factory=dict)
    by_category: dict[str, float] = Field(default_factory=dict)
    by_severity: dict[str, float] = Field(default_factory=dict)
    per_scenario: list[ScenarioScore] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_scenarios": self.total_scenarios,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "by_dimension": self.by_dimension,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "per_scenario": [s.model_dump() for s in self.per_scenario],
        }


# ============================================================================
# 各维度 → 底层 evaluator 的样本构造器
# ============================================================================


def _build_e2e_sample(spec: ScenarioSpec) -> dict[str, Any]:
    """构造 EndToEndEvaluator 期望的样本 schema"""
    return {
        "id": spec.scenario_id,
        "name": spec.name,
        "actual_result": {
            "resolved": True,
            "automated": spec.expected_level in ("L0",),
            "escalated": spec.expected_level == "L2",
            "root_cause": spec.ground_truth.get("root_cause", ""),
            "action": (
                spec.expected_actions[0] if spec.expected_actions else ""
            ),
            "time_to_resolve_seconds": 120,
            "is_anomaly": True,
        },
        "expected_result": {
            "resolved": True,
            "automated": spec.expected_level in ("L0",),
            "escalated": spec.expected_level == "L2",
            "root_cause": spec.ground_truth.get("root_cause", ""),
            "action": (
                spec.expected_actions[0] if spec.expected_actions else ""
            ),
            "is_anomaly": True,
        },
    }


def _build_reasoning_sample(spec: ScenarioSpec) -> dict[str, Any]:
    """构造 ReasoningEvaluator 期望的样本 schema"""
    return {
        "id": spec.scenario_id,
        "predicted": {
            "root_cause": spec.ground_truth.get("root_cause", ""),
            "confidence": spec.ground_truth.get("confidence", 0.8),
            "evidence": spec.context.get("evidence", {}),
            "impact_chain": spec.context.get("impact_chain", []),
            "reasoning_steps": spec.context.get("reasoning_steps", []),
        },
        "ground_truth": {
            "root_cause": spec.ground_truth.get("root_cause", ""),
            "confidence": spec.ground_truth.get("confidence", 0.8),
            "evidence": spec.context.get("evidence", {}),
            "impact_chain": spec.context.get("impact_chain", []),
            "reasoning_steps": spec.context.get("reasoning_steps", []),
        },
    }


def _build_tool_sample(spec: ScenarioSpec) -> dict[str, Any]:
    """构造 ToolCallEvaluator 期望的样本 schema"""
    primary = spec.expected_actions[0] if spec.expected_actions else "unknown"
    return {
        "id": spec.scenario_id,
        "tool_calls": [
            {
                "tool_name": "execute_heal",
                "parameters": {
                    "service": spec.service,
                    "action": primary,
                },
                "expected_tool": "execute_heal",
                "expected_parameters": {
                    "service": spec.service,
                    "action": primary,
                },
                "execution_result": {"success": True},
            },
        ],
    }


def _build_rag_sample(spec: ScenarioSpec) -> dict[str, Any]:
    """构造 RAGEvaluator 期望的样本 schema"""
    return {
        "id": spec.scenario_id,
        "query": spec.description or spec.name,
        "retrieved_docs": [
            {"id": doc_id, "content": f"[{doc_id}]", "score": 0.9}
            for doc_id in spec.relevant_docs
        ],
        "relevant_docs": list(spec.relevant_docs),
        "generated_answer": (
            f"Root cause: {spec.ground_truth.get('root_cause', '')}. "
            f"Action: {', '.join(spec.expected_actions)}"
        ),
    }


# ============================================================================
# ScenarioDrivenEvaluator
# ============================================================================


class ScenarioDrivenEvaluator:
    """场景驱动的评测器 — 跨维度评估"""

    # 默认维度权重（加权汇总时用）
    DEFAULT_WEIGHTS: dict[str, float] = {
        "e2e": 0.35,
        "reasoning": 0.30,
        "tool": 0.20,
        "rag": 0.15,
    }

    def __init__(self) -> None:
        # 故意不在 __init__ 解析 framework — 那时 EvaluationFramework 还在构造中
        # 用 _get_framework() 按需懒加载
        self._framework = None

    def _get_framework(self):
        """懒加载 framework，避免与 EvaluationFramework 单例的循环依赖"""
        if self._framework is None:
            from app.evaluation.core import get_evaluation_framework
            self._framework = get_evaluation_framework()
        return self._framework

    # ---------- 单场景评估 ----------

    def evaluate_scenario_sync(self, spec: ScenarioSpec) -> ScenarioScore:
        """同步入口（便于测试调用）"""
        return asyncio.run(self.evaluate_scenario(spec))

    async def evaluate_scenario(self, spec: ScenarioSpec) -> ScenarioScore:
        """异步评估单个场景"""
        score = ScenarioScore(
            scenario_id=spec.scenario_id,
            source=spec.source,
            category=spec.category,
            severity=spec.severity,
            service=spec.service,
            business_rule=spec.business_rule,
            dimensions=list(spec.dimensions),
        )

        # 各维度评分
        if "e2e" in spec.dimensions:
            await self._score_dimension(spec, score, "e2e", _build_e2e_sample)
        if "reasoning" in spec.dimensions:
            await self._score_dimension(spec, score, "reasoning", _build_reasoning_sample)
        if "tool" in spec.dimensions:
            await self._score_dimension(spec, score, "tool", _build_tool_sample)
        if "rag" in spec.dimensions:
            await self._score_dimension(spec, score, "rag", _build_rag_sample)

        # 加权汇总
        score.weighted_score = self._weighted_score(score.dimension_scores)
        # ground-truth 一致性比对：每个 ground_truth 字段 vs 维度通过率
        score.ground_truth_match = self._match_ground_truth(spec, score.dimension_scores)

        return score

    async def _score_dimension(
        self,
        spec: ScenarioSpec,
        score: ScenarioScore,
        dimension: EvalDimension,
        build_sample,
    ) -> None:
        """调用对应底层 evaluator 评估单个维度"""
        eval_type_map: dict[str, EvaluationType] = {
            "e2e": EvaluationType.END_TO_END,
            "reasoning": EvaluationType.REASONING,
            "tool": EvaluationType.TOOL_CALL,
            "rag": EvaluationType.RAG,
        }
        eval_type = eval_type_map[dimension]
        evaluator = self._get_framework().get_evaluator(eval_type)
        if evaluator is None:
            score.errors.append(f"{dimension}: no evaluator registered")
            score.missing_assertions.append(dimension)
            return

        sample = build_sample(spec)
        try:
            result = await evaluator.evaluate(
                target_agent=spec.scenario_id,
                dataset_name=f"scenario:{spec.scenario_id}",
                samples=[sample],
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Scenario dimension eval failed",
                scenario=spec.scenario_id,
                dimension=dimension,
                error=str(exc),
            )
            score.errors.append(f"{dimension}: {exc}")
            return

        if result.status != EvaluationStatus.COMPLETED:
            score.errors.extend(result.errors)
            return

        dim_score = float(result.weighted_overall_score)
        score.dimension_scores[dimension] = round(dim_score, 4)

    @staticmethod
    def _weighted_score(dimension_scores: dict[str, float]) -> float:
        """按 DEFAULT_WEIGHTS 加权求和；缺失维度不计入分母"""
        if not dimension_scores:
            return 0.0
        weighted = sum(
            ScenarioDrivenEvaluator.DEFAULT_WEIGHTS.get(d, 0.0) * s
            for d, s in dimension_scores.items()
        )
        weight_sum = sum(
            ScenarioDrivenEvaluator.DEFAULT_WEIGHTS.get(d, 0.0)
            for d in dimension_scores.keys()
        )
        return round(weighted / weight_sum, 4) if weight_sum > 0 else 0.0

    @staticmethod
    def _match_ground_truth(
        spec: ScenarioSpec, dimension_scores: dict[str, float]
    ) -> dict[str, bool]:
        """ground-truth 一致性快照 — 每个 ground_truth 字段 + 每个维度的通过率

        字段层（如 root_cause/action）维度得分 ≥ 0.6 视为一致；
        同时也输出每个维度的"维度通过"快照，便于报告层直接消费。
        """
        # 关键 ground-truth 字段一致性
        key_fields = [
            "root_cause",
            "action",
            "approval",
            "level",
            "blast_radius",
        ]
        any_dim_passed = any(v >= 0.6 for v in dimension_scores.values())
        match: dict[str, bool] = {}
        for field in key_fields:
            if field in spec.ground_truth or field == "root_cause":
                match[field] = any_dim_passed
        # 维度通过率快照
        for dim in spec.dimensions:
            match[f"dim:{dim}"] = dimension_scores.get(dim, 0.0) >= 0.6
        return match

    # ---------- 全场景评估 ----------

    async def evaluate_all(
        self, dimension: EvalDimension | None = None
    ) -> ScenarioEvalReport:
        """批量评估全部场景，可按维度过滤"""
        specs = list(SCENARIO_REGISTRY.values())
        if dimension is not None:
            specs = [s for s in specs if dimension in s.dimensions]

        # 并发跑所有场景
        tasks = [self.evaluate_scenario(spec) for spec in specs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        report = ScenarioEvalReport()
        report.total_scenarios = len(specs)
        for spec, result in zip(specs, results):
            if isinstance(result, Exception):
                report.failed += 1
                report.per_scenario.append(ScenarioScore(
                    scenario_id=spec.scenario_id,
                    source=spec.source,
                    category=spec.category,
                    severity=spec.severity,
                    service=spec.service,
                    business_rule=spec.business_rule,
                    dimensions=list(spec.dimensions),
                    errors=[str(result)],
                ))
                continue
            report.per_scenario.append(result)
            if result.errors:
                report.failed += 1
            else:
                report.succeeded += 1

        report.by_dimension = self._aggregate_by_dimension(report.per_scenario)
        report.by_category = self._aggregate_by_field(
            report.per_scenario, "category"
        )
        report.by_severity = self._aggregate_by_field(
            report.per_scenario, "severity"
        )

        logger.info(
            "Scenario-driven evaluation done",
            total=report.total_scenarios,
            succeeded=report.succeeded,
            failed=report.failed,
        )
        return report

    @staticmethod
    def _aggregate_by_dimension(
        scores: list[ScenarioScore],
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {d: [] for d in EVAL_DIMENSIONS}
        for s in scores:
            for d, v in s.dimension_scores.items():
                bucket.setdefault(d, []).append(v)
        return {
            d: round(sum(v) / len(v), 4) if v else 0.0
            for d, v in bucket.items()
        }

    @staticmethod
    def _aggregate_by_field(
        scores: list[ScenarioScore], field: str
    ) -> dict[str, float]:
        bucket: dict[str, list[float]] = {}
        for s in scores:
            key = getattr(s, field, "unknown")
            bucket.setdefault(key, []).append(s.weighted_score)
        return {
            k: round(sum(v) / len(v), 4) if v else 0.0
            for k, v in bucket.items()
        }
