"""LoopEngine — 每轮 write_back 状态机 (W8 Task 2.2, P0 pillar 2/5).

Implements PPT P4 Loop: "每一轮调查都是受协议约束的状态更新:
scope/观测/SOP/门禁/证据都要写回图状态."
"""

from __future__ import annotations

from app.models.runtime import (
    InvestigationLoopState,
    IterationRecord,
    LoopStatus,
)
from app.models.umodel import EvidenceBlock, InvestigationGraph
from app.services.scope import DFSScope
from app.services.umodel_store import InvestigationGraphStore


class LoopEngine:
    """Per-iteration state machine that writes back scope, evidence, and audit records."""

    def __init__(
        self,
        store: InvestigationGraphStore,
        scope_engine: DFSScope,
        max_iterations: int = 10,
    ) -> None:
        """Wire a graph store, a scope engine, and the iteration cap."""
        self.store = store
        self.scope_engine = scope_engine
        self.max_iterations = max_iterations

    def run_iteration(
        self,
        state: InvestigationLoopState,
        graph: InvestigationGraph,
        action: str,
        evidence: EvidenceBlock | None = None,
    ) -> tuple[InvestigationLoopState, InvestigationGraph]:
        """Advance the loop by one iteration; append evidence, expand scope, persist graph."""
        if state.status != LoopStatus.RUNNING:
            raise ValueError(f"loop is in status {state.status}, cannot iterate")
        state.current_iteration += 1
        graph.iteration = state.current_iteration

        if evidence is not None:
            graph.evidence_blocks.append(evidence)
            self.scope_engine.expand_scope_on_evidence(graph, evidence)

        record = IterationRecord(
            iteration=state.current_iteration,
            scope_size=len(graph.scope_boundary),
            evidence_count=len(graph.evidence_blocks),
            actions=[action],
        )
        state.records.append(record)
        graph.iteration_history.append({
            "iteration": state.current_iteration,
            "action": action,
            "scope_size": len(graph.scope_boundary),
            "evidence_count": len(graph.evidence_blocks),
        })
        self.store.save(state.tenant, state.case_id, graph)
        return state, graph

    def should_continue(self, state: InvestigationLoopState) -> bool:
        """Return True only while status is RUNNING and iteration cap is not exhausted."""
        return (
            state.status == LoopStatus.RUNNING
            and state.current_iteration < self.max_iterations
        )

    def mark_converged(self, state: InvestigationLoopState) -> InvestigationLoopState:
        """Flip status to CONVERGED so the loop driver stops iterating."""
        state.status = LoopStatus.CONVERGED
        return state