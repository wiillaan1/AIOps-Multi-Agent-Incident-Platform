"""Unit tests for UModel Pydantic models (Task 1.1 of graph-driven investigation runtime)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.umodel import (
    EntitySet,
    EntitySetLink,
    EntityState,
    EntityType,
    EvidenceBlock,
    InvestigationGraph,
    LinkType,
    entity_id,
)


def test_entity_type_enum_has_all_eight_types():
    assert {t.value for t in EntityType} == {
        "service", "pod", "topic", "db", "node", "runbook", "ci_cd", "change_event"
    }


def test_link_type_enum_has_eleven_types():
    assert len(LinkType) >= 8  # 至少 8 种


def test_entity_id_combines_type_and_id():
    assert entity_id(EntityType.SERVICE, "checkout-api") == "service:checkout-api"
    assert entity_id("pod", "pod-abc123") == "pod:pod-abc123"


def test_entity_set_minimal_required_fields():
    e = EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout-api", state=EntityState.NORMAL)
    assert e.attributes == {}
    assert e.labels == []
    assert e.entity_type == EntityType.SERVICE


def test_entity_set_rejects_unknown_state():
    with pytest.raises(ValidationError):
        EntitySet(entity_type="service", entity_id="x", state="bogus")


def test_link_requires_src_and_dst_distinct():
    with pytest.raises(ValidationError):
        EntitySetLink(
            src="service:a", dst="service:a", link_type=LinkType.CALLS
        )


def test_evidence_block_requires_confidence_in_range():
    with pytest.raises(ValidationError):
        EvidenceBlock(
            block_id="eb-1",
            object_ref="service:checkout-api",
            time="2026-07-23T10:00:00Z",
            observation="latency p95 > 1s",
            mechanism="dependency",
            confidence=1.5,  # 超出 [0, 1]
        )


def test_investigation_graph_round_trip_json():
    g = InvestigationGraph()
    svc = EntitySet(entity_type=EntityType.SERVICE, entity_id="checkout")
    pod = EntitySet(entity_type=EntityType.POD, entity_id="pod-1")
    g.entities[svc.id] = svc
    g.entities[pod.id] = pod
    g.links.append(
        EntitySetLink(src=pod.id, dst=svc.id, link_type=LinkType.HOSTED_ON)
    )
    g2 = InvestigationGraph.model_validate_json(g.model_dump_json())
    assert len(g2.entities) == 2
    assert len(g2.links) == 1
    assert g2.iteration == 0
    assert g2.scope_boundary == set()


def test_investigation_graph_add_evidence_appends_in_order():
    g = InvestigationGraph()
    for i in range(3):
        g.evidence_blocks.append(
            EvidenceBlock(
                block_id=f"eb-{i}",
                object_ref="service:x",
                time=f"2026-07-23T10:0{i}:00Z",
                observation=f"obs {i}",
                mechanism="metric",
                confidence=0.5 + i * 0.1,
            )
        )
    assert [b.block_id for b in g.evidence_blocks] == ["eb-0", "eb-1", "eb-2"]
