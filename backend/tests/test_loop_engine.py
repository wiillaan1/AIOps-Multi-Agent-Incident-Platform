"""Unit tests for LoopEngine (Task 2.2, P0 pillar 2/5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.runtime import (
    InvestigationLoopState,
    IterationRecord,
    LoopStatus,
)
from app.models.umodel import (
    EntitySet,
    EntitySetLink,
    EntityType,
    EvidenceBlock,
    InvestigationGraph,
    LinkType,
)
from app.services.loop_engine import LoopEngine
from app.services.scope import DFSScope
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


@pytest.fixture
def scope(store) -> DFSScope:
    return DFSScope(store, k_hop=1)


def _state(case_id: str = "INC-1") -> InvestigationLoopState:
    return InvestigationLoopState(case_id=case_id, tenant="t1", max_iterations=5)


def _graph() -> InvestigationGraph:
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout"))
    g.add_entity(EntitySet(entity_type=EntityType.DB, entity_id="orders"))
    g.add_link(EntitySetLink(src="service:checkout", dst="db:orders", link_type=LinkType.READS))
    return g


def test_run_iteration_increments_current_iteration(store, scope):
    engine = LoopEngine(store, scope, max_iterations=5)
    state = _state()
    g = _graph()
    state, g = engine.run_iteration(state, g, action="collect_metric", evidence=None)
    assert state.current_iteration == 1


def test_run_iteration_appends_evidence_block_to_graph(store, scope):
    engine = LoopEngine(store, scope)
    state = _state()
    g = _graph()
    eb = EvidenceBlock(
        block_id="eb-1", object_ref="service:checkout", time="t", observation="p95 spike",
        mechanism="metric", confidence=0.7,
    )
    state, g = engine.run_iteration(state, g, action="collect", evidence=eb)
    assert any(b.block_id == "eb-1" for b in g.evidence_blocks)


def test_run_iteration_writes_record(store, scope):
    engine = LoopEngine(store, scope)
    state = _state()
    g = _graph()
    state, g = engine.run_iteration(state, g, action="act", evidence=None)
    assert len(state.records) == 1
    assert state.records[0].iteration == 1
    assert state.records[0].actions == ["act"]


def test_should_continue_true_when_iterations_left(store, scope):
    engine = LoopEngine(store, scope, max_iterations=5)
    state = _state()
    g = _graph()
    state, g = engine.run_iteration(state, g, action="a", evidence=None)
    assert engine.should_continue(state) is True


def test_should_continue_false_when_max_iterations_reached(store, scope):
    engine = LoopEngine(store, scope, max_iterations=2)
    state = _state()
    g = _graph()
    state, g = engine.run_iteration(state, g, action="a", evidence=None)
    state, g = engine.run_iteration(state, g, action="a", evidence=None)
    assert engine.should_continue(state) is False


def test_mark_converged_sets_status(store, scope):
    engine = LoopEngine(store, scope)
    state = _state()
    state = engine.mark_converged(state)
    assert state.status == LoopStatus.CONVERGED


def test_run_iteration_persists_graph(store, scope):
    engine = LoopEngine(store, scope)
    state = _state()
    g = _graph()
    state, g = engine.run_iteration(state, g, action="a", evidence=None)
    g2 = store.load("t1", "INC-1")
    assert g2.iteration == 1
    assert len(g2.evidence_blocks) == 0  # no evidence in this run


def test_run_iteration_evidence_appended_preserves_order(store, scope):
    engine = LoopEngine(store, scope)
    state = _state()
    g = _graph()
    for i in range(3):
        eb = EvidenceBlock(
            block_id=f"eb-{i}", object_ref="service:checkout", time="t",
            observation=f"obs {i}", mechanism="m", confidence=0.5,
        )
        state, g = engine.run_iteration(state, g, action="a", evidence=eb)
    assert [b.block_id for b in g.evidence_blocks] == ["eb-0", "eb-1", "eb-2"]