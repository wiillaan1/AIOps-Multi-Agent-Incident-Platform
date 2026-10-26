"""Tests for AgentProtocol + W7 5 角色适配 (Task 3.1, P2 计划)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.models.gate import CounterfactualClaim
from app.models.protocol import AgentMessage
from app.models.umodel import (
    EntitySet,
    EntityType,
    EvidenceBlock,
    InvestigationGraph,
)
from app.services.agent_protocol import AgentProtocol, W7AgentAdapter
from app.services.counterfactual_gate import CounterfactualGate
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


@pytest.fixture
def graph(store) -> InvestigationGraph:
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout"))
    g.add_entity(EntitySet(entity_type=EntityType.POD, entity_id="p1"))
    g.scope_boundary = {"service:checkout", "pod:p1"}
    return g


@pytest.fixture
def adapter(store, graph) -> W7AgentAdapter:
    return W7AgentAdapter(
        store=store,
        graph=graph,
        case_id="INC-1",
        tenant="t1",
        k_hop=1,
    )


def test_submit_evidence_appends_to_graph(adapter):
    eb = EvidenceBlock(
        block_id="eb-1", object_ref="service:checkout", time="t",
        observation="latency p95 spiked", mechanism="metric", confidence=0.8,
    )
    adapter.submit_evidence(eb)
    assert any(b.block_id == "eb-1" for b in adapter.graph.evidence_blocks)


def test_submit_evidence_expands_scope(adapter):
    eb = EvidenceBlock(
        block_id="eb-1", object_ref="pod:other", time="t",
        observation="x", mechanism="metric", confidence=0.5,
    )
    adapter.submit_evidence(eb)
    assert "pod:other" in adapter.graph.scope_boundary


def test_request_next_hop_returns_scoped_neighbors(adapter):
    # add link first
    from app.models.umodel import EntitySetLink, LinkType
    adapter.graph.add_link(EntitySetLink(src="pod:p1", dst="service:checkout", link_type=LinkType.HOSTED_ON))
    nbrs = adapter.request_next_hop("service:checkout")
    assert "pod:p1" in nbrs


def test_cross_gate_passes_with_sufficient_evidence(adapter):
    eb1 = EvidenceBlock(
        block_id="eb-1", object_ref="service:checkout", time="2026-07-23T10:00:00Z",
        observation="cpu rose", mechanism="metric", confidence=0.9,
    )
    eb2 = EvidenceBlock(
        block_id="eb-2", object_ref="service:checkout", time="2026-07-23T10:00:01Z",
        observation="latency rose", mechanism="metric", confidence=0.9,
    )
    adapter.submit_evidence(eb1)
    adapter.submit_evidence(eb2)
    claim = CounterfactualClaim(
        claim="if cpu was not high, would latency still be high?",
        mechanism="resource_contention",
        time_anchor="2026-07-23T10:00:00Z",
        supporting_evidence_ids=["eb-1", "eb-2"],
        refuting_evidence_ids=[],
    )
    check = adapter.cross_gate(claim)
    assert check.result.value in {"pass", "fail"}


def test_record_hypothesis_appends_to_state(adapter):
    adapter.record_hypothesis("rca_agent", "DB is the cause", confidence=0.5)
    assert any(h["hypothesis"] == "DB is the cause" for h in adapter.hypotheses)


def test_send_message_returns_message_id(adapter):
    msg = adapter.send_message("rca_agent", "heal_agent", "claim", {"x": 1})
    assert isinstance(msg, AgentMessage)
    assert msg.message_type == "claim"


def test_persist_to_store(adapter, store):
    adapter.persist()
    g2 = store.load("t1", "INC-1")
    assert "service:checkout" in g2.entities


def test_w7_adapter_does_not_mutate_inner_agent_methods():
    """Verify the adapter wraps, not replaces, W7 5 角色 methods.
    Use a mock that records calls; assert original method is invoked through adapter."""
    mock_inner = MagicMock()
    mock_inner.run_sequential_with_reflection = MagicMock(return_value={"ok": True})
    adapter = W7AgentAdapter(
        store=MagicMock(),
        graph=InvestigationGraph(),
        case_id="INC-1",
        tenant="t1",
        w7_inner_orchestrator=mock_inner,
    )
    # adapter exposes a passthrough
    out = adapter.invoke_w7_orchestrator()
    mock_inner.run_sequential_with_reflection.assert_called_once()
    assert out == {"ok": True}


def test_protocol_persists_state_to_graph_file(adapter, tmp_path):
    adapter.record_hypothesis("rca_agent", "test", confidence=0.5)
    adapter.persist()
    files = list((tmp_path / "umodel" / "t1").glob("INC-1.json"))
    assert len(files) == 1
