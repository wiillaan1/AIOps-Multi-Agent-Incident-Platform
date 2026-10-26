"""Unit + integration tests for InvestigationGraphStore (Task 1.2)."""

from __future__ import annotations

import pytest

from app.models.umodel import (
    EntitySet,
    EntitySetLink,
    EntityType,
    EvidenceBlock,
    InvestigationGraph,
    LinkType,
)
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path) -> InvestigationGraphStore:
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


def _sample_graph() -> InvestigationGraph:
    g = InvestigationGraph()
    svc = EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout")
    db = EntitySet(entity_type=EntityType.DB, entity_id="orders-db")
    pod = EntitySet(entity_type=EntityType.POD, entity_id="pod-1")
    g.add_entity(svc)
    g.add_entity(db)
    g.add_entity(pod)
    g.add_link(EntitySetLink(src=pod.id, dst=svc.id, link_type=LinkType.HOSTED_ON))
    g.add_link(EntitySetLink(src=svc.id, dst=db.id, link_type=LinkType.READS))
    return g


def test_load_returns_empty_graph_when_file_missing(store):
    g = store.load("tenant-a", "case-1")
    assert isinstance(g, InvestigationGraph)
    assert len(g.entities) == 0
    assert g.iteration == 0


def test_save_then_load_round_trip(store):
    g = _sample_graph()
    g.iteration = 3
    g.evidence_blocks.append(
        EvidenceBlock(
            block_id="eb-1",
            object_ref="service:checkout",
            time="2026-07-23T10:00:00Z",
            observation="latency p95 spiked",
            mechanism="metric",
            confidence=0.8,
        )
    )
    store.save("tenant-a", "case-1", g)
    g2 = store.load("tenant-a", "case-1")
    assert len(g2.entities) == 3
    assert g2.iteration == 3
    assert len(g2.evidence_blocks) == 1


def test_save_is_atomic_writes_temp_then_rename(store, tmp_path):
    g = _sample_graph()
    store.save("tenant-a", "case-1", g)
    target = tmp_path / "umodel" / "tenant-a" / "case-1.json"
    assert target.exists()
    # No leftover .tmp files
    leftovers = list((tmp_path / "umodel" / "tenant-a").glob("*.tmp"))
    assert leftovers == []


def test_save_cleans_up_temp_file_on_serialization_failure(store, tmp_path, monkeypatch):
    """If JSON serialization raises mid-write, the temp file must be removed (no leftover)."""
    g = _sample_graph()

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr("app.services.umodel_store.json.dump", boom)

    with pytest.raises(RuntimeError, match="simulated disk full"):
        store.save("tenant-a", "case-1", g)

    leftovers = list((tmp_path / "umodel" / "tenant-a").glob("*.tmp"))
    assert leftovers == []
    # Final target never written either
    assert not (tmp_path / "umodel" / "tenant-a" / "case-1.json").exists()


def test_save_rejects_path_traversal_in_tenant(store):
    g = _sample_graph()
    with pytest.raises(ValueError, match="invalid tenant"):
        store.save("../escape", "case-1", g)
    with pytest.raises(ValueError, match="invalid tenant"):
        store.save("/etc/passwd", "case-1", g)


def test_save_rejects_path_traversal_in_case_id(store):
    g = _sample_graph()
    with pytest.raises(ValueError, match="invalid case_id"):
        store.save("tenant-a", "../escape", g)
    with pytest.raises(ValueError, match="invalid case_id"):
        store.save("tenant-a", "case/with/slashes", g)


def test_neighbors_returns_one_hop_default(store):
    g = _sample_graph()
    nbrs = store.neighbors(g, "pod:pod-1", k_hop=1)
    assert "service:checkout" in nbrs


def test_neighbors_returns_two_hops(store):
    g = _sample_graph()
    nbrs = store.neighbors(g, "pod:pod-1", k_hop=2)
    assert "service:checkout" in nbrs
    assert "db:orders-db" in nbrs


def test_path_finds_shortest_path(store):
    g = _sample_graph()
    p = store.path(g, "pod:pod-1", "db:orders-db")
    assert p == ["pod:pod-1", "service:checkout", "db:orders-db"]


def test_subgraph_within_scope_excludes_outside_entities(store):
    g = _sample_graph()
    g.scope_boundary = {"service:checkout", "pod:pod-1"}
    sub = store.subgraph_within_scope(g)
    assert "service:checkout" in sub.entities
    assert "pod:pod-1" in sub.entities
    assert "db:orders-db" not in sub.entities
