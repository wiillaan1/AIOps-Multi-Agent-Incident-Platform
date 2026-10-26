"""InvestigationGraphStore — NetworkX-backed storage for UModel (W8 Task 1.2)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import networkx as nx

from app.models.umodel import InvestigationGraph

# Identifiers must be safe file-system segments: alphanumerics, dot,
# underscore, hyphen. Prevents path traversal (`../`) and absolute paths
# from escaping base_path during load/save.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class InvestigationGraphStore:
    """Persists InvestigationGraph as JSON, exposes graph algorithms via NetworkX.

    Storage layout: ``{base_path}/{tenant}/{case_id}.json`` (atomic write).
    Each case is an independent graph file; multi-tenant isolation by directory.
    """

    def __init__(self, base_path: str = "backend/data/umodel") -> None:
        self.base_path = base_path

    # ------------------------------------------------------------------ IO

    def load(self, tenant: str, case_id: str) -> InvestigationGraph:
        """Load the persisted InvestigationGraph for tenant/case_id; returns an empty graph when no file exists."""
        path = self._path(tenant, case_id)
        if not path.exists():
            return InvestigationGraph()
        data = json.loads(path.read_text(encoding="utf-8"))
        return InvestigationGraph.model_validate(data)

    def save(self, tenant: str, case_id: str, graph: InvestigationGraph) -> None:
        """Atomically persist graph as JSON under {base_path}/{tenant}/{case_id}.json (temp + rename)."""
        path = self._path(tenant, case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = graph.model_dump(mode="json")
        # atomic write via temp + rename
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{case_id}.", suffix=".tmp", dir=str(path.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_name, path)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _path(self, tenant: str, case_id: str) -> Path:
        # Reject identifiers with path separators / `..` / absolute paths
        # before joining under base_path.
        if not _SAFE_ID_RE.match(tenant):
            raise ValueError(f"invalid tenant identifier: {tenant!r}")
        if not _SAFE_ID_RE.match(case_id):
            raise ValueError(f"invalid case_id identifier: {case_id!r}")
        return Path(self.base_path) / tenant / f"{case_id}.json"

    # ---------------------------------------------------------- graph algos

    def _to_nx(self, g: InvestigationGraph) -> nx.DiGraph:
        nxg = nx.DiGraph()
        for eid, ent in g.entities.items():
            nxg.add_node(eid, **ent.attributes, state=ent.state.value, type=ent.entity_type.value)
        for link in g.links:
            nxg.add_edge(link.src, link.dst, type=link.link_type.value, weight=link.weight)
        return nxg

    def neighbors(self, g: InvestigationGraph, entity_id: str, k_hop: int = 1) -> list[str]:
        """Return the union of predecessors and successors within k_hop BFS, excluding entity_id itself."""
        nxg = self._to_nx(g)
        if entity_id not in nxg:
            return []
        visited = {entity_id}
        frontier = {entity_id}
        for _ in range(k_hop):
            new_frontier: set[str] = set()
            for node in frontier:
                for nbr in nxg.successors(node):
                    if nbr not in visited:
                        new_frontier.add(nbr)
                        visited.add(nbr)
                for nbr in nxg.predecessors(node):
                    if nbr not in visited:
                        new_frontier.add(nbr)
                        visited.add(nbr)
            frontier = new_frontier
            if not frontier:
                break
        visited.discard(entity_id)
        return sorted(visited)

    def path(self, g: InvestigationGraph, src: str, dst: str) -> list[str] | None:
        """Return the shortest directed path from src to dst, or None if either endpoint is missing or unreachable."""
        nxg = self._to_nx(g)
        if src not in nxg or dst not in nxg:
            return None
        try:
            return list(nx.shortest_path(nxg, src, dst))
        except nx.NetworkXNoPath:
            return None

    def subgraph_within_scope(self, g: InvestigationGraph) -> InvestigationGraph:
        """Return a deep copy of g restricted to entities/links fully inside g.scope_boundary (full graph if scope is empty)."""
        if not g.scope_boundary:
            return g.model_copy(deep=True)
        sub = InvestigationGraph(scope_boundary=set(g.scope_boundary), iteration=g.iteration)
        for eid, ent in g.entities.items():
            if eid in g.scope_boundary:
                sub.add_entity(ent)
        sub.links = [
            link for link in g.links
            if link.src in g.scope_boundary and link.dst in g.scope_boundary
        ]
        return sub
