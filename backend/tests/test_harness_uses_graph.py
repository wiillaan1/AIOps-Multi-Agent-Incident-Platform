"""Tests for HarnessLoopEngine optional P0 Gate and P1 Graph dependencies."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.badcase import BadCaseEntry
from app.models.gate import CounterfactualClaim
from app.models.umodel import EntitySet, EntityType, EvidenceBlock, InvestigationGraph
from app.services.counterfactual_gate import CounterfactualGate
from app.services.harness_engine import HarnessLoopEngine
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


def _make_entry(
    suffix: str,
    badcase_class: str = "rca_predict_mismatch",
) -> BadCaseEntry:
    return BadCaseEntry(
        id=f"bc-{suffix}",
        created_at=datetime.now(timezone.utc),
        badcase_class=badcase_class,
        incident_id="INC-1",
        audit_trail_excerpt=[{"step": "x"}],
        identified_flaw="flaw text long enough for validation",
        keywords_for_retrieval=["k"],
        suggestion_or_lesson="lesson text long enough for validation",
        eval_split="dev",
    )


def test_harness_without_graph_still_works(tmp_path: Path) -> None:
    """Constructing without graph dependencies remains backward compatible."""
    engine = HarnessLoopEngine(
        registry=None,  # type: ignore[arg-type]
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
    )

    assert engine is not None


def test_harness_optional_graph_args_default_to_none(tmp_path: Path) -> None:
    """Optional graph dependencies default to None."""
    engine = HarnessLoopEngine(
        registry=None,  # type: ignore[arg-type]
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
    )

    assert getattr(engine, "_graph_store", None) is None
    assert getattr(engine, "_gate_factory", None) is None


def test_harness_accepts_graph_store_and_gate_factory(
    tmp_path: Path,
    store: InvestigationGraphStore,
) -> None:
    """Constructor retains injected graph and gate dependencies."""
    engine = HarnessLoopEngine(
        registry=None,  # type: ignore[arg-type]
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        graph_store=store,
        gate_factory=CounterfactualGate,
    )

    assert engine._graph_store is store
    assert engine._gate_factory is CounterfactualGate


def test_harness_accepts_graph_store_without_gate_factory(
    tmp_path: Path,
    store: InvestigationGraphStore,
) -> None:
    """Graph storage can be injected independently from the gate factory."""
    engine = HarnessLoopEngine(
        registry=None,  # type: ignore[arg-type]
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        graph_store=store,
    )

    assert engine._graph_store is store
    assert engine._gate_factory is None


def test_harness_accepts_gate_factory_without_graph_store(tmp_path: Path) -> None:
    """Gate construction can be injected independently from graph storage."""
    engine = HarnessLoopEngine(
        registry=None,  # type: ignore[arg-type]
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        gate_factory=CounterfactualGate,
    )

    assert engine._graph_store is None
    assert engine._gate_factory is CounterfactualGate


def test_harness_injected_dependencies_are_runtime_compatible(
    tmp_path: Path,
    store: InvestigationGraphStore,
) -> None:
    """Injected dependencies retain their existing graph and gate behavior."""
    engine = HarnessLoopEngine(
        registry=None,  # type: ignore[arg-type]
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        graph_store=store,
        gate_factory=CounterfactualGate,
    )
    graph = InvestigationGraph()
    graph.add_entity(
        EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout")
    )
    graph.evidence_blocks.extend(
        [
            EvidenceBlock(
                block_id="eb-1",
                object_ref="service:checkout",
                time="t",
                observation="x",
                mechanism="metric",
                confidence=0.9,
            ),
            EvidenceBlock(
                block_id="eb-2",
                object_ref="service:checkout",
                time="t",
                observation="x",
                mechanism="metric",
                confidence=0.9,
            ),
        ]
    )
    engine._graph_store.save("tenant", "case", graph)
    persisted = engine._graph_store.load("tenant", "case")
    claim = CounterfactualClaim(
        claim="if x, would y?",
        mechanism="m",
        time_anchor="t",
        supporting_evidence_ids=["eb-1", "eb-2"],
        refuting_evidence_ids=[],
    )

    check = engine._gate_factory(persisted).evaluate(claim)

    assert "service:checkout" in persisted.entities
    assert check.result.value in {"pass", "fail"}
