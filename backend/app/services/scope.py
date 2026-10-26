"""DFS Scope — investigation scope constraint (W8 Task 2.1, P0 pillar 1/5).

Implements the "Scope" pillar from 刘贵阳 PPT P4: "RCA 最大的问题不是'看不见对象',
而是'看太多对象'. DFS Scope 把注意力管理变成运行时机制."
"""

from __future__ import annotations

from app.models.umodel import EvidenceBlock, InvestigationGraph
from app.services.umodel_store import InvestigationGraphStore


class DFSScope:
    """N-hop neighborhood scope constructor + dynamic evidence-driven expansion."""

    def __init__(self, store: InvestigationGraphStore, k_hop: int = 2) -> None:
        """Initialize with a graph store and the default hop radius for ``build_scope``."""
        self.store = store
        self.k_hop = k_hop

    def build_scope(self, graph: InvestigationGraph, root: str) -> InvestigationGraph:
        """Return a deep copy of ``graph`` whose ``scope_boundary`` covers ``root`` + N-hop neighbors."""
        if root not in graph.entities:
            raise ValueError(f"root {root} not in graph entities")
        nbrs = self.store.neighbors(graph, root, k_hop=self.k_hop)
        out = graph.model_copy(deep=True)
        out.scope_boundary = {root} | set(nbrs)
        return out

    def expand_scope_on_evidence(
        self, graph: InvestigationGraph, evidence: EvidenceBlock
    ) -> set[str]:
        """Admit ``evidence.object_ref`` into the live scope boundary; return the newly added ids."""
        added: set[str] = set()
        ref = evidence.object_ref
        if ref in graph.entities and ref not in graph.scope_boundary:
            graph.scope_boundary.add(ref)
            added.add(ref)
        return added

    def failure_tri_query(
        self, graph: InvestigationGraph, claim: str
    ) -> dict[str, str]:
        """Diagnose investigation failure: scope / observation / report errors."""
        out: dict[str, str] = {}
        # (1) Scope 选错?
        if not graph.entities:
            out["scope_error"] = (
                "Scope selection failed: graph is empty; root or entry entity not registered"
            )
        elif not graph.scope_boundary:
            out["scope_error"] = "Scope boundary not built; DFS Scope step skipped"
        elif len(graph.scope_boundary) >= len(graph.entities) * 0.9:
            out["scope_error"] = (
                f"Scope too broad ({len(graph.scope_boundary)}/{len(graph.entities)}); "
                "consider shrinking k_hop"
            )
        else:
            out["scope_error"] = "scope looks normal"

        # (2) Observation 不够?
        if len(graph.evidence_blocks) < 2:
            out["observation_error"] = (
                f"Only {len(graph.evidence_blocks)} evidence block(s); "
                "counterfactual check needs >= 2"
            )
        else:
            out["observation_error"] = "observation coverage looks normal"

        # (3) Report 失真?
        avg_conf = (
            sum(b.confidence for b in graph.evidence_blocks) / len(graph.evidence_blocks)
            if graph.evidence_blocks else 0.0
        )
        if avg_conf < 0.4:
            out["report_error"] = (
                f"Average evidence confidence {avg_conf:.2f} < 0.4; report may distort"
            )
        else:
            out["report_error"] = "report confidence looks normal"

        return out
