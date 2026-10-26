"""Tests for FallbackController (Task 3.2, P2 计划)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.models.runtime import InvestigationLoopState
from app.models.umodel import (
    EntitySet,
    EntityType,
    EvidenceBlock,
    InvestigationGraph,
)
from app.services.agent_protocol import W7AgentAdapter
from app.services.fallback import (
    FallbackController,
    FallbackDecision,
    FallbackLevel,
    FallbackTrigger,
)
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


@pytest.fixture
def graph() -> InvestigationGraph:
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout"))
    g.scope_boundary = {"service:checkout"}
    return g


@pytest.fixture
def adapter(store, graph) -> W7AgentAdapter:
    return W7AgentAdapter(store=store, graph=graph, case_id="INC-1", tenant="t1")


def test_l1_retry_when_loop_not_exhausted_but_gate_fails_once(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=5, current_iteration=2)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=1)
    assert dec.level == FallbackLevel.L1_RETRY


def test_l2_strategy_switch_when_l1_exhausted(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=5, current_iteration=2)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=3)
    assert dec.level == FallbackLevel.L2_STRATEGY_SWITCH


def test_l3_human_handover_when_loop_exhausted(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=3, current_iteration=3)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=0)
    assert dec.level == FallbackLevel.L3_HUMAN_HANDOVER
    assert dec.trigger == FallbackTrigger.LOOP_EXHAUSTED


def test_confidence_too_low_triggers_l2(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=5, current_iteration=2)
    adapter.hypotheses.append({"agent": "rca", "hypothesis": "x", "confidence": 0.1})
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=0, hypotheses=adapter.hypotheses)
    assert dec.level == FallbackLevel.L2_STRATEGY_SWITCH
    assert dec.trigger == FallbackTrigger.CONFIDENCE_TOO_LOW


def test_l1_retry_action_is_revert_one_step(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=5, current_iteration=2)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=1)
    assert "revert" in dec.action.lower() or "retry" in dec.action.lower()


def test_l2_action_suggests_scope_or_hypothesis_change(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=5, current_iteration=2)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=3)
    assert any(kw in dec.action.lower() for kw in ["scope", "hypothesis", "sop"])


def test_l3_action_mentions_human_or_oncall(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=3, current_iteration=3)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=0)
    assert "human" in dec.action.lower() or "on-call" in dec.action.lower() or "report" in dec.action.lower()


def test_apply_l1_retry_reverts_iteration(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=5, current_iteration=2)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=1)
    out = fc.apply(dec, adapter, state)
    assert "reverted_iterations" in out
    assert out["reverted_iterations"] == 1


def test_apply_l3_persists_state_report(adapter, tmp_path):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=3, current_iteration=3)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=0)
    fc.apply(dec, adapter, state)
    files = list((tmp_path / "umodel" / "t1").glob("INC-1.json"))
    assert len(files) == 1


def test_fallback_decision_carries_reason_string(adapter):
    state = InvestigationLoopState(case_id="INC-1", max_iterations=3, current_iteration=3)
    fc = FallbackController()
    dec = fc.decide(state, adapter.graph, recent_gate_failures=0)
    assert isinstance(dec.reason, str) and len(dec.reason) > 0
