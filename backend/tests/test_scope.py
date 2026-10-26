"""Unit tests for DFS Scope + failure_tri_query (Task 2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.umodel import (
    EntitySet,
    EntitySetLink,
    EntityType,
    EvidenceBlock,
    InvestigationGraph,
    LinkType,
)
from app.services.scope import DFSScope
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    """A fresh InvestigationGraphStore rooted at a tmp dir."""
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


def _build_chain_graph() -> InvestigationGraph:
    """svc -> pod -> node -> pod2; svc reads db."""
    g = InvestigationGraph()
    for eid in ["service:checkout", "pod:pod-1", "node:node-a", "pod:pod-2", "db:orders"]:
        t, raw = eid.split(":")
        g.add_entity(EntitySet(entity_type=EntityType(t), entity_id=raw))
    g.add_link(EntitySetLink(src="pod:pod-1", dst="service:checkout", link_type=LinkType.HOSTED_ON))
    g.add_link(EntitySetLink(src="node:node-a", dst="pod:pod-1", link_type=LinkType.RUNS_ON))
    g.add_link(EntitySetLink(src="node:node-a", dst="pod:pod-2", link_type=LinkType.RUNS_ON))
    g.add_link(EntitySetLink(src="service:checkout", dst="db:orders", link_type=LinkType.READS))
    return g


def test_build_scope_includes_one_hop_neighbors(store):
    """k_hop=1 reaches pod and db but not 2-hop node."""
    g = _build_chain_graph()
    scope = DFSScope(store, k_hop=1)
    g2 = scope.build_scope(g, "service:checkout")
    assert "pod:pod-1" in g2.scope_boundary
    assert "db:orders" in g2.scope_boundary
    assert "node:node-a" not in g2.scope_boundary  # 2-hop
    g2.scope_boundary.add("node:node-a")
    assert "node:node-a" not in g.scope_boundary


def test_build_scope_k_hop_two_includes_two_hops(store):
    """k_hop=2 reaches node through pod->node edge."""
    g = _build_chain_graph()
    scope = DFSScope(store, k_hop=2)
    g2 = scope.build_scope(g, "service:checkout")
    assert "node:node-a" in g2.scope_boundary


def test_build_scope_includes_root(store):
    """Root entity must always be present in its own scope boundary."""
    g = _build_chain_graph()
    scope = DFSScope(store, k_hop=1)
    g2 = scope.build_scope(g, "service:checkout")
    assert "service:checkout" in g2.scope_boundary


def test_build_scope_raises_for_unknown_root(store):
    """Unknown root entity must raise ValueError, not silently return empty scope."""
    g = _build_chain_graph()
    scope = DFSScope(store, k_hop=1)
    with pytest.raises(ValueError):
        scope.build_scope(g, "service:does-not-exist")


def test_expand_scope_on_evidence_adds_object_ref(store):
    """Evidence referencing a known entity must be admitted into the live scope boundary."""
    g = _build_chain_graph()
    g.scope_boundary = {"service:checkout"}
    scope = DFSScope(store, k_hop=1)
    new = scope.expand_scope_on_evidence(
        g,
        EvidenceBlock(
            block_id="eb-1",
            object_ref="pod:pod-1",
            time="t",
            observation="x",
            mechanism="m",
            confidence=0.5,
        ),
    )
    assert "pod:pod-1" in new
    assert "pod:pod-1" in g.scope_boundary


def test_expand_scope_on_evidence_returns_empty_for_in_scope_ref(store):
    """Evidence already represented in scope must produce no expansion delta."""
    g = _build_chain_graph()
    g.scope_boundary = {"service:checkout", "pod:pod-1"}
    scope = DFSScope(store, k_hop=1)
    new = scope.expand_scope_on_evidence(
        g,
        EvidenceBlock(
            block_id="eb-existing",
            object_ref="pod:pod-1",
            time="t",
            observation="x",
            mechanism="m",
            confidence=0.5,
        ),
    )
    assert new == set()


def test_failure_tri_query_returns_three_keys(store):
    """failure_tri_query must always return the three PPT failure modes."""
    g = _build_chain_graph()
    scope = DFSScope(store, k_hop=1)
    out = scope.failure_tri_query(g, "CPU 100%")
    assert set(out.keys()) == {"scope_error", "observation_error", "report_error"}


def test_failure_tri_query_detects_empty_scope(store):
    """On an empty graph, scope_error must flag the missing root/entry entity."""
    g = InvestigationGraph()  # no entities
    scope = DFSScope(store, k_hop=1)
    out = scope.failure_tri_query(g, "anything")
    assert "scope" in out["scope_error"].lower()


def test_failure_tri_query_flags_single_evidence_as_insufficient(store):
    """One evidence block must be reported as insufficient observation coverage."""
    g = _build_chain_graph()
    g.scope_boundary = {"service:checkout", "pod:pod-1"}
    g.evidence_blocks.append(
        EvidenceBlock(
            block_id="eb-only",
            object_ref="pod:pod-1",
            time="t",
            observation="x",
            mechanism="m",
            confidence=0.8,
        )
    )
    out = DFSScope(store, k_hop=1).failure_tri_query(g, "CPU 100%")
    assert "Only 1 evidence block(s)" in out["observation_error"]
    assert "needs >= 2" in out["observation_error"]
