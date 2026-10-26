"""
AIOps Agent Platform - Golden Set Regression Runner

把 `GoldenDatasets` 跑一遍生成基线；之后每次跑同一组场景，与基线对比，
差值超过 `regression_threshold`（默认 0.1）的场景记为回归。

用途：
- PR 集成前跑一次，确认没把已稳定的 case 跑坏；
- 模型/规则升级后跑一次，验证不是负优化；
- 周期调度任务里跑（每天一次），写入失败列表供后续聚类。

设计：
- baseline 持久化到 JSON 文件（路径可配）；
- 支持 score_overrides 注入，用于单元测试场景；
- 输出 RegressionAlert 列表 + 汇总指标。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.layered_datasets import GoldenDatasets
from app.evaluation.scenario_evaluator import (
    ScenarioDrivenEvaluator,
    ScenarioEvalReport,
    ScenarioScore,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# 数据模型
# ============================================================================


class RegressionAlert(BaseModel):
    """单条回归告警"""

    scenario_id: str
    dimension: str = "weighted"
    baseline_score: float
    current_score: float
    delta: float  # current - baseline; negative 表示下降


class GoldenRunResult(BaseModel):
    """一次 GoldenSetRunner.run() 的完整输出"""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    baseline_path: str = ""
    total_scenarios: int = 0
    scenarios: list[ScenarioScore] = Field(default_factory=list)
    regressions: list[RegressionAlert] = Field(default_factory=list)
    baseline_applied: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "baseline_path": self.baseline_path,
            "total_scenarios": self.total_scenarios,
            "baseline_applied": self.baseline_applied,
            "regressions": [r.model_dump() for r in self.regressions],
            "scenarios": [s.model_dump() for s in self.scenarios],
            "note": self.note,
        }


# ============================================================================
# 基线文件 IO
# ============================================================================


def _load_baseline(path: str) -> dict[str, dict[str, float]] | None:
    """从 JSON 加载基线，返回 {scenario_id: {dim: score}}；不存在返回 None"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("per_scenario", {})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to load baseline", path=path, error=str(exc))
        return None


def _save_baseline(
    path: str,
    scenarios: list[ScenarioScore],
    metadata: dict[str, Any] | None = None,
) -> None:
    """把当前结果写入基线 JSON"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "per_scenario": {
            s.scenario_id: {
                "weighted": s.weighted_score,
                **s.dimension_scores,
            }
            for s in scenarios
        },
    }
    p.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Baseline saved", path=path, scenarios=len(scenarios))


# ============================================================================
# GoldenSetRunner
# ============================================================================


class GoldenSetRunner:
    """Golden 集合的回归基线对比运行器"""

    DEFAULT_BASELINE_PATH = "runtime/eval/golden_baseline.json"
    DEFAULT_THRESHOLD = 0.1

    def __init__(
        self,
        baseline_path: str | None = None,
        regression_threshold: float = DEFAULT_THRESHOLD,
        score_overrides: dict[str, float] | None = None,
    ) -> None:
        self.baseline_path = baseline_path or self.DEFAULT_BASELINE_PATH
        self.regression_threshold = regression_threshold
        # 测试 / 演示用：把指定场景的 weighted_score 强制设值
        self.score_overrides = score_overrides or {}

    async def run(
        self,
        save_baseline: bool = False,
    ) -> GoldenRunResult:
        """跑 GoldenDatasets 全部场景，与基线对比

        Args:
            save_baseline: True 时把本次结果覆盖写入基线文件

        Returns:
            GoldenRunResult：含 scenarios / regressions / baseline_applied
        """
        golden_samples = GoldenDatasets.all()
        if not golden_samples:
            return GoldenRunResult(
                baseline_path=self.baseline_path,
                note="GoldenDatasets empty",
            )

        evaluator = ScenarioDrivenEvaluator()
        scenario_ids = [s["scenario_id"] for s in golden_samples]

        # 逐场景跑（不并发：避免 framework 单例 + asyncio 资源竞争）
        scenario_scores: list[ScenarioScore] = []
        from app.evaluation.scenarios import get_scenario

        for sid in scenario_ids:
            spec = get_scenario(sid)
            if spec is None:
                continue
            score = await evaluator.evaluate_scenario(spec)
            if sid in self.score_overrides:
                score.weighted_score = self.score_overrides[sid]
            scenario_scores.append(score)

        # 基线对比
        baseline_data = _load_baseline(self.baseline_path)
        baseline_applied = baseline_data is not None
        regressions: list[RegressionAlert] = []
        if baseline_applied:
            regressions = self._diff_against_baseline(
                scenario_scores, baseline_data
            )

        result = GoldenRunResult(
            baseline_path=self.baseline_path,
            total_scenarios=len(scenario_scores),
            scenarios=scenario_scores,
            regressions=regressions,
            baseline_applied=baseline_applied,
            note=(
                "baseline not found, run save_baseline=True to create"
                if not baseline_applied else ""
            ),
        )

        if save_baseline:
            _save_baseline(self.baseline_path, scenario_scores)

        logger.info(
            "GoldenSetRunner.run complete",
            total=result.total_scenarios,
            regressions=len(result.regressions),
            baseline_applied=baseline_applied,
        )
        return result

    def _diff_against_baseline(
        self,
        current: list[ScenarioScore],
        baseline: dict[str, dict[str, float]],
    ) -> list[RegressionAlert]:
        """对比每个 scenario 的每个维度分，发现下降超过阈值就报警"""
        alerts: list[RegressionAlert] = []
        for cur in current:
            base = baseline.get(cur.scenario_id)
            if not base:
                continue
            base_weighted = float(base.get("weighted", 0.0))
            cur_weighted = cur.weighted_score
            delta = cur_weighted - base_weighted
            if delta < -self.regression_threshold:
                alerts.append(RegressionAlert(
                    scenario_id=cur.scenario_id,
                    dimension="weighted",
                    baseline_score=round(base_weighted, 4),
                    current_score=round(cur_weighted, 4),
                    delta=round(delta, 4),
                ))
        return alerts

    # ---------- 同步便捷入口 ----------

    def run_sync(self, save_baseline: bool = False) -> GoldenRunResult:
        return asyncio.run(self.run(save_baseline=save_baseline))
