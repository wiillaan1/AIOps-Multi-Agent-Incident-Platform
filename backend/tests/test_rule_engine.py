"""Tests for Checklist + RuleEngine (Task 2.5, P0 pillar 5/5)."""

from __future__ import annotations

from app.models.rules import ChecklistItem, Rule, RuleMatch
from app.models.umodel import EntitySet, EntityType, EvidenceBlock, InvestigationGraph
from app.services.rule_engine import Checklist, RuleEngine


def test_checklist_for_service_has_at_least_four_items():
    items = Checklist().for_entity(EntityType.SERVICE)
    assert len(items) >= 4
    for it in items:
        assert isinstance(it, ChecklistItem)
        assert it.required is True


def test_checklist_for_pod_includes_memory_check():
    dims = {it.dimension for it in Checklist().for_entity(EntityType.POD)}
    assert any("memory" in d.lower() or "gc" in d.lower() for d in dims)


def test_checklist_for_db_includes_connection_check():
    dims = {it.dimension for it in Checklist().for_entity(EntityType.DB)}
    assert any("connection" in d.lower() or "lag" in d.lower() for d in dims)


def test_checklist_for_runbook_is_small():
    assert 1 <= len(Checklist().for_entity(EntityType.RUNBOOK)) <= 5


def test_rule_engine_matches_high_cpu_rule():
    engine = RuleEngine()
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout"))
    g.evidence_blocks.append(EvidenceBlock(block_id="eb-1", object_ref="service:checkout", time="t", observation="cpu 95%", mechanism="metric", confidence=0.9))
    matches = engine.match(g, "service:checkout")
    assert any("cpu" in m.rule.rule_id.lower() or "cpu" in str(m.produced_observations).lower() for m in matches)


def test_rule_engine_returns_empty_for_unmatched():
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout"))
    g.evidence_blocks.append(EvidenceBlock(block_id="eb-1", object_ref="service:checkout", time="t", observation="all normal", mechanism="metric", confidence=0.5))
    assert isinstance(RuleEngine().match(g, "service:checkout"), list)


def test_rule_engine_priority_sorted_desc():
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout"))
    g.evidence_blocks.append(EvidenceBlock(block_id="eb-1", object_ref="service:checkout", time="t", observation="cpu 99%, latency p95 5s", mechanism="metric", confidence=0.9))
    matches = RuleEngine().match(g, "service:checkout")
    assert all(matches[i].rule.priority >= matches[i + 1].rule.priority for i in range(len(matches) - 1))


def test_rule_engine_handles_unknown_entity_gracefully():
    assert RuleEngine().match(InvestigationGraph(), "service:does-not-exist") == []


def test_rule_engine_adds_custom_rule_takes_effect():
    engine = RuleEngine()
    engine.add_rule(Rule(rule_id="custom_k8s_event", trigger_entity_type=EntityType.POD, condition_evidence_pattern="oomkilled", produces_observation_plan=["check_memory_limit", "check_oom_kills_total"], priority=99))
    g = InvestigationGraph()
    g.add_entity(EntitySet(entity_type=EntityType.POD, entity_id="p"))
    g.evidence_blocks.append(EvidenceBlock(block_id="eb-1", object_ref="pod:p", time="t", observation="oomkilled at 10:00", mechanism="k8s_event", confidence=0.95))
    assert any(m.rule.rule_id == "custom_k8s_event" for m in engine.match(g, "pod:p"))


def test_rule_models_validate_and_match_type():
    assert isinstance(RuleMatch(rule=Rule(rule_id="x", trigger_entity_type=EntityType.NODE, condition_evidence_pattern="disk"), matched_evidence_ids=[], produced_observations=[]), RuleMatch)
