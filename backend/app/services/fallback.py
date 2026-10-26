"""FallbackController — 3 级降级 (W8 Task 3.2, P2 计划).

Implements PPT P5 Fallback:
- L1 自动重试: 轻微超时 / 单次 Gate 失败 → 回退 1-2 步重试
- L2 策略切换: 连续失败 / 方向错误 → 切 Scope / 换假设 / 换 SOP
- L3 人工介入: 推进不动 → 生成状态报告 + Trace 导出 + 转 on-call
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.models.runtime import InvestigationLoopState
from app.models.umodel import InvestigationGraph
from app.services.agent_protocol import W7AgentAdapter


class FallbackTrigger(str, Enum):
    """Reason that triggered a fallback escalation decision."""

    LOOP_EXHAUSTED = "loop_exhausted"
    GATE_REPEATEDLY_FAILING = "gate_repeatedly_failing"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    CONFIDENCE_TOO_LOW = "confidence_too_low"
    TIMEOUT = "timeout"


class FallbackLevel(str, Enum):
    """3-level fallback cascade: L1 retry → L2 strategy switch → L3 human handover."""

    L1_RETRY = "l1_retry"
    L2_STRATEGY_SWITCH = "l2_strategy_switch"
    L3_HUMAN_HANDOVER = "l3_human_handover"


class FallbackDecision(BaseModel):
    """Materialised fallback decision — what to do and why."""

    trigger: FallbackTrigger
    level: FallbackLevel
    reason: str
    action: str


class FallbackController:
    """3 级降级控制器, decides level + applies side-effects to adapter."""

    L1_THRESHOLD = 1
    L2_THRESHOLD = 3
    CONFIDENCE_FLOOR = 0.3

    def decide(
        self,
        state: InvestigationLoopState,
        graph: InvestigationGraph,
        recent_gate_failures: int = 0,
        hypotheses: list[dict[str, Any]] | None = None,
    ) -> FallbackDecision:
        """Pick a fallback level based on loop pressure, confidence, and gate failure streak."""
        if state.current_iteration >= state.max_iterations:
            return FallbackDecision(
                trigger=FallbackTrigger.LOOP_EXHAUSTED,
                level=FallbackLevel.L3_HUMAN_HANDOVER,
                reason=f"current_iteration {state.current_iteration} >= max {state.max_iterations}",
                action="generate_state_report_and_handover_to_oncall",
            )
        avg_conf = self._avg_confidence(graph, hypotheses)
        if avg_conf < self.CONFIDENCE_FLOOR:
            return FallbackDecision(
                trigger=FallbackTrigger.CONFIDENCE_TOO_LOW,
                level=FallbackLevel.L2_STRATEGY_SWITCH,
                reason=f"avg evidence/hypothesis confidence {avg_conf:.2f} < {self.CONFIDENCE_FLOOR}",
                action="switch_scope_or_replace_hypothesis",
            )
        if recent_gate_failures >= self.L2_THRESHOLD:
            return FallbackDecision(
                trigger=FallbackTrigger.GATE_REPEATEDLY_FAILING,
                level=FallbackLevel.L2_STRATEGY_SWITCH,
                reason=f"{recent_gate_failures} recent gate failures >= {self.L2_THRESHOLD}",
                action="switch_scope_or_swap_sop",
            )
        if recent_gate_failures >= self.L1_THRESHOLD:
            return FallbackDecision(
                trigger=FallbackTrigger.GATE_REPEATEDLY_FAILING,
                level=FallbackLevel.L1_RETRY,
                reason=f"{recent_gate_failures} recent gate failures",
                action="revert_one_step_and_retry",
            )
        return FallbackDecision(
            trigger=FallbackTrigger.LOOP_EXHAUSTED,
            level=FallbackLevel.L1_RETRY,
            reason="no fallback trigger met",
            action="continue",
        )

    def apply(
        self,
        decision: FallbackDecision,
        adapter: W7AgentAdapter,
        state: InvestigationLoopState,
    ) -> dict[str, Any]:
        """Apply the decision — revert state / expand scope / persist human-readable report."""
        if decision.level == FallbackLevel.L1_RETRY:
            reverted = 1
            state.current_iteration = max(0, state.current_iteration - reverted)
            return {
                "level": decision.level.value,
                "reverted_iterations": reverted,
                "new_iteration": state.current_iteration,
            }
        if decision.level == FallbackLevel.L2_STRATEGY_SWITCH:
            if adapter.hypotheses:
                adapter.hypotheses.sort(key=lambda h: h.get("confidence", 0))
                dropped = adapter.hypotheses.pop(0)
            else:
                dropped = None
            new_entities: set[str] = set()
            for eid in list(adapter.graph.scope_boundary):
                nbrs = adapter.store.neighbors(adapter.graph, eid, k_hop=1)
                for n in nbrs:
                    if n not in adapter.graph.scope_boundary:
                        adapter.graph.scope_boundary.add(n)
                        new_entities.add(n)
            adapter.persist()
            return {
                "level": decision.level.value,
                "dropped_hypothesis": dropped,
                "scope_expanded": sorted(new_entities),
            }
        if decision.level == FallbackLevel.L3_HUMAN_HANDOVER:
            report = {
                "case_id": state.case_id,
                "tenant": state.tenant,
                "iteration": state.current_iteration,
                "max_iterations": state.max_iterations,
                "entities": sorted(adapter.graph.entities.keys()),
                "evidence_count": len(adapter.graph.evidence_blocks),
                "scope_size": len(adapter.graph.scope_boundary),
                "hypotheses": adapter.hypotheses,
                "decision_reason": decision.reason,
            }
            adapter.persist()
            report_path = (
                Path(adapter.store.base_path) / adapter.tenant / f"{state.case_id}-state-report.json"
            )
            report_path.write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
            return {
                "level": decision.level.value,
                "report_path": str(report_path),
                "handover": "on-call",
            }
        return {"level": decision.level.value, "action": decision.action}

    @staticmethod
    def _avg_confidence(
        graph: InvestigationGraph,
        hypotheses: list[dict[str, Any]] | None = None,
    ) -> float:
        """Mean confidence across evidence blocks and recorded hypotheses; 1.0 when neither is populated."""
        samples: list[float] = [b.confidence for b in graph.evidence_blocks]
        if hypotheses:
            for h in hypotheses:
                c = h.get("confidence")
                if isinstance(c, (int, float)):
                    samples.append(float(c))
        if not samples:
            return 1.0
        return sum(samples) / len(samples)
