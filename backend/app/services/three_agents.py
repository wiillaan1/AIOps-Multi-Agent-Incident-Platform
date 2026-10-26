"""Benchmark / RCA / Optimization 三 Agent 共享 Investigation Graph (W8 Task 3.3).

P2 计划收口: BenchmarkAgent 评测 + OptimizationAgent 补丁建议 + ThreeAgentOrchestrator
端到端编排. 三者共享同一个 W7AgentAdapter 上的 Investigation Graph, 通过协议外壳
(Evidence Block / Hypothesis) 协作.
"""

from __future__ import annotations

from typing import Any

from app.models.incident import Incident
from app.models.runtime import InvestigationLoopState, LoopStatus
from app.models.umodel import EvidenceBlock
from app.services.agent_protocol import W7AgentAdapter
from app.services.fallback import FallbackController, FallbackLevel
from app.services.loop_engine import LoopEngine
from app.services.rule_engine import RuleEngine
from app.services.scope import DFSScope
from app.services.umodel_ingestor import UModelIngestor
from app.services.umodel_store import InvestigationGraphStore

# Iteration timestamp literal — pinned so deterministic within a single run.
_ITER_TIMESTAMP = "2026-07-24T10:00:00Z"

# How many top patches an iteration tries to apply; bounded to keep steps bounded.
_PATCHES_PER_ITERATION = 2

# Number of graph entities scanned when building the rule-match report.
_RULE_SCAN_LIMIT = 10


class BenchmarkAgent:
    """Run a small eval case library against the current graph and emit coverage metrics."""

    def run(
        self,
        adapter: W7AgentAdapter,
        eval_cases: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute entity coverage = matched expected entities / total expected entities."""
        if not eval_cases:
            return {"entity_coverage": 0.0, "case_count": 0}
        hit = 0
        total = 0
        for case in eval_cases:
            expected = set(case.get("expected_entities", []))
            total += len(expected)
            hit += sum(1 for e in expected if e in adapter.graph.entities)
        return {
            "entity_coverage": hit / total if total else 0.0,
            "case_count": float(len(eval_cases)),
        }


class OptimizationAgent:
    """Inspect the current graph and propose rule-driven optimization patches."""

    def propose_patch(self, adapter: W7AgentAdapter) -> list[dict[str, Any]]:
        """Run RuleEngine against each entity and collect candidate observation plans."""
        rule_engine = RuleEngine()
        patches: list[dict[str, Any]] = []
        for eid in adapter.graph.entities:
            for match in rule_engine.match(adapter.graph, eid):
                patches.append({
                    "target_entity": eid,
                    "rule_id": match.rule.rule_id,
                    "produced_observations": match.produced_observations,
                    "matched_evidence_ids": match.matched_evidence_ids,
                })
        return patches


class ThreeAgentOrchestrator:
    """End-to-end driver: iterate Scope/Observe/Decide across Benchmark/Optimization/Adapter."""

    def __init__(
        self,
        store: InvestigationGraphStore,
        max_iterations: int = 5,
        k_hop: int = 2,
    ) -> None:
        """Wire store, scope engine, loop engine, rule engine, and fallback controller."""
        self.store = store
        self.max_iterations = max_iterations
        self.k_hop = k_hop
        self.scope_engine = DFSScope(store, k_hop=k_hop)
        self.loop_engine = LoopEngine(store, self.scope_engine, max_iterations)
        self.rule_engine = RuleEngine()
        self.benchmark = BenchmarkAgent()
        self.optimizer = OptimizationAgent()
        self.fallback_controller = FallbackController()

    def run_case(
        self,
        incident: Incident,
        ingestor: UModelIngestor,
        tenant: str = "t1",
    ) -> dict[str, Any]:
        """Run a single case end-to-end and return a structured result dict."""
        case_id = incident.incident_id
        graph = ingestor.ingest_incident(incident, tenant=tenant)
        state = InvestigationLoopState(
            case_id=case_id, tenant=tenant, max_iterations=self.max_iterations,
        )
        adapter = W7AgentAdapter(
            store=self.store,
            graph=graph,
            case_id=case_id,
            tenant=tenant,
            k_hop=self.k_hop,
        )

        # Stage 1 — Scope: seed a focused scope boundary when a root entity is known.
        root: str | None = None
        if "service" in (incident.context or {}):
            candidate = f"service:{incident.context['service']}"
            if candidate in graph.entities:
                root = candidate
                adapter.graph = self.scope_engine.build_scope(adapter.graph, root)

        # Stage 2 — Observe: loop proposal -> run iteration -> fallback decide.
        gate_failures = 0
        fallback_used = False
        while self.loop_engine.should_continue(state):
            patches = self.optimizer.propose_patch(adapter)
            if patches:
                for p in patches[:_PATCHES_PER_ITERATION]:
                    adapter.record_hypothesis(
                        "optimization",
                        f"rule {p['rule_id']} -> {p['produced_observations']}",
                        confidence=0.5,
                    )
                    eb = EvidenceBlock(
                        block_id=f"eb-iter{state.current_iteration + 1}-{p['rule_id']}",
                        object_ref=p["target_entity"],
                        time=_ITER_TIMESTAMP,
                        observation=f"matched rule {p['rule_id']}",
                        mechanism="rule_engine",
                        confidence=0.5,
                    )
                    state, graph = self.loop_engine.run_iteration(
                        state, graph, "execute_rule", evidence=eb,
                    )
                    adapter.graph = graph
            else:
                # No patches: still advance the loop with a placeholder observation
                # so should_continue() can move past the cap and trigger L3.
                state, graph = self.loop_engine.run_iteration(
                    state, graph, "observe_no_patches",
                )
                adapter.graph = graph

            dec = self.fallback_controller.decide(
                state, graph, recent_gate_failures=gate_failures, hypotheses=adapter.hypotheses,
            )
            if dec.level == FallbackLevel.L3_HUMAN_HANDOVER:
                self.fallback_controller.apply(dec, adapter, state)
                fallback_used = True
                break
            if dec.action != "continue":
                # Genuine L1 / L2 trigger; default "continue" decisions are no-ops.
                self.fallback_controller.apply(dec, adapter, state)
            gate_failures = 0

        # Stage 3 — Decide: benchmark metrics + rule-match report + persist.
        cases = [{"case_id": case_id, "expected_entities": list(graph.entities.keys())[:5]}]
        bench_metrics = self.benchmark.run(adapter, cases)
        rule_matches: list[dict[str, Any]] = []
        for eid in list(graph.entities.keys())[:_RULE_SCAN_LIMIT]:
            for m in self.rule_engine.match(graph, eid):
                rule_matches.append({"entity": eid, "rule_id": m.rule.rule_id})

        adapter.persist()
        if state.status == LoopStatus.RUNNING:
            state.status = LoopStatus.EXHAUSTED

        return {
            "case_id": case_id,
            "iterations": state.current_iteration,
            "metrics": {
                **bench_metrics,
                "entities_count": len(graph.entities),
                "evidence_count": len(graph.evidence_blocks),
                "hypotheses_count": len(adapter.hypotheses),
                "rule_matches": len(rule_matches),
            },
            "fallback_used": fallback_used,
            "rule_matches": rule_matches[:5],
            "root_entity": root,
        }