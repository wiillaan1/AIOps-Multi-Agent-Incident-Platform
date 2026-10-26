"""E2E tests for three-agent protocol-first loop (Task 3.3, P2 plan)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.incident import Incident
from app.models.umodel import EvidenceBlock
from app.services.agent_protocol import W7AgentAdapter
from app.services.umodel_ingestor import UModelIngestor
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    """Return a fresh InvestigationGraphStore rooted at tmp_path."""
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


@pytest.fixture
def sample_incident() -> Incident:
    """Return a checkout-503 incident with service/pod/db context for graph ingest."""
    return Incident(
        incident_id="INC-2026-300",
        title="checkout 503",
        status="pending",
        severity="high",
        created_at=datetime.now(timezone.utc),
        context={"service": "checkout", "pod": "checkout-7d9f", "db": "orders"},
    )


def test_benchmark_agent_runs_eval_cases(store, sample_incident):
    """BenchmarkAgent must compute entity coverage against the current graph."""
    from app.services.three_agents import BenchmarkAgent

    ingestor = UModelIngestor(store=store)
    g = ingestor.ingest_incident(sample_incident, tenant="t1")
    adapter = W7AgentAdapter(
        store=store, graph=g, case_id=sample_incident.incident_id, tenant="t1"
    )
    bench = BenchmarkAgent()
    cases = [
        {"case_id": "c1", "expected_entities": ["service:checkout"]},
        {"case_id": "c2", "expected_entities": ["pod:checkout-7d9f"]},
        {"case_id": "c3", "expected_entities": ["db:orders"]},
    ]
    metrics = bench.run(adapter, cases)
    assert "entity_coverage" in metrics
    assert 0.0 <= metrics["entity_coverage"] <= 1.0


def test_optimization_agent_proposes_patch_for_high_cpu(store, sample_incident):
    """OptimizationAgent must surface a rule-driven patch when CPU evidence is present."""
    from app.services.three_agents import OptimizationAgent

    ingestor = UModelIngestor(store=store)
    g = ingestor.ingest_incident(sample_incident, tenant="t1")
    g.evidence_blocks.append(EvidenceBlock(
        block_id="eb-cpu-1", object_ref="service:checkout", time="t",
        observation="cpu 95%", mechanism="metric", confidence=0.9,
    ))
    adapter = W7AgentAdapter(
        store=store, graph=g, case_id=sample_incident.incident_id, tenant="t1"
    )
    opt = OptimizationAgent()
    patches = opt.propose_patch(adapter)
    assert isinstance(patches, list)
    assert any("cpu" in str(p).lower() for p in patches)


def test_three_agent_orchestrator_runs_full_loop(store, sample_incident):
    """ThreeAgentOrchestrator.run_case must return a structured result dict."""
    from app.services.three_agents import ThreeAgentOrchestrator

    ingestor = UModelIngestor(store=store)
    g = ingestor.ingest_incident(sample_incident, tenant="t1")
    orch = ThreeAgentOrchestrator(store=store, max_iterations=5)
    result = orch.run_case(sample_incident, ingestor=ingestor)
    assert result["case_id"] == sample_incident.incident_id
    assert "iterations" in result
    assert "metrics" in result
    assert "fallback_used" in result


def test_three_agent_orchestrator_respects_max_iterations(store, sample_incident):
    """Loop cap must be respected: iterations never exceed max + 1 slack."""
    from app.services.three_agents import ThreeAgentOrchestrator

    orch = ThreeAgentOrchestrator(store=store, max_iterations=2)
    ingestor = UModelIngestor(store=store)
    result = orch.run_case(sample_incident, ingestor=ingestor)
    assert result["iterations"] <= 3  # allow one final iteration


def test_three_agent_loop_writes_back_to_graph(store, sample_incident):
    """After run_case the persisted graph must reflect iteration progress."""
    from app.services.three_agents import ThreeAgentOrchestrator

    orch = ThreeAgentOrchestrator(store=store, max_iterations=3)
    ingestor = UModelIngestor(store=store)
    orch.run_case(sample_incident, ingestor=ingestor)
    g2 = store.load("t1", sample_incident.incident_id)
    assert g2.iteration > 0


def test_three_agent_orchestrator_collects_metrics(store, sample_incident):
    """Result metrics dict must surface counts for entities/evidence/hypotheses."""
    from app.services.three_agents import ThreeAgentOrchestrator

    orch = ThreeAgentOrchestrator(store=store, max_iterations=3)
    ingestor = UModelIngestor(store=store)
    result = orch.run_case(sample_incident, ingestor=ingestor)
    metrics = result["metrics"]
    assert "entities_count" in metrics
    assert "evidence_count" in metrics
    assert "hypotheses_count" in metrics


def test_three_agent_orchestrator_uses_rule_engine(store, sample_incident):
    """Result metrics must include the rule_matches count surfaced by RuleEngine."""
    from app.services.three_agents import ThreeAgentOrchestrator

    orch = ThreeAgentOrchestrator(store=store, max_iterations=3)
    ingestor = UModelIngestor(store=store)
    result = orch.run_case(sample_incident, ingestor=ingestor)
    assert "rule_matches" in result["metrics"]


def test_three_agent_orchestrator_uses_fallback_when_loop_exhausted(store, sample_incident):
    """When max_iterations is 1, fallback_used must flip True (L3 or earlier path)."""
    from app.services.three_agents import ThreeAgentOrchestrator

    orch = ThreeAgentOrchestrator(store=store, max_iterations=1)
    ingestor = UModelIngestor(store=store)
    result = orch.run_case(sample_incident, ingestor=ingestor)
    assert result["fallback_used"] is True


def test_three_agent_orchestrator_persists_final_graph(store, sample_incident):
    """After run_case the JSON artifact must exist under {tenant}/{case_id}.json."""
    from app.services.three_agents import ThreeAgentOrchestrator

    orch = ThreeAgentOrchestrator(store=store, max_iterations=3)
    ingestor = UModelIngestor(store=store)
    orch.run_case(sample_incident, ingestor=ingestor)
    assert (Path(store.base_path) / "t1" / f"{sample_incident.incident_id}.json").exists()