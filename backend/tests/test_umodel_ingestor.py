"""Tests for UModelIngestor (Task 1.3)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.models.incident import Incident
from app.models.umodel import (
    EntitySet,
    EntityType,
    InvestigationGraph,
    LinkType,
)
from app.services.umodel_ingestor import UModelIngestor
from app.services.umodel_store import InvestigationGraphStore


@pytest.fixture
def store(tmp_path: Path) -> InvestigationGraphStore:
    return InvestigationGraphStore(base_path=str(tmp_path / "umodel"))


@pytest.fixture
def mock_runbook_service() -> MagicMock:
    svc = MagicMock()
    svc.list_published_runbooks.return_value = [
        {"runbook_id": "rb-checkout-503", "service_name": "checkout", "title": "Checkout 503 playbook"},
        {"runbook_id": "rb-db-orders", "service_name": "orders-db", "title": "Orders DB recovery"},
    ]
    return svc


def _make_incident(service_name: str = "checkout") -> Incident:
    return Incident(
        incident_id="INC-2026-100",
        title="checkout 503",
        status="pending",
        severity="high",
        created_at=datetime.now(timezone.utc),
        context={"service": service_name, "pod": "checkout-7d9f"},
    )


def test_ingest_incident_creates_service_entity(store, mock_runbook_service):
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    g = ingestor.ingest_incident(_make_incident("checkout"), tenant="t1")
    assert "service:checkout" in g.entities
    assert g.entities["service:checkout"].entity_type == EntityType.SERVICE


def test_ingest_incident_links_pod_to_service(store, mock_runbook_service):
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    g = ingestor.ingest_incident(_make_incident("checkout"), tenant="t1")
    links = [l for l in g.links if l.link_type == LinkType.HOSTED_ON]
    assert any(l.src == "pod:checkout-7d9f" and l.dst == "service:checkout" for l in links)


def test_ingest_incident_registers_linked_runbooks(store, mock_runbook_service):
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    g = ingestor.ingest_incident(_make_incident("checkout"), tenant="t1")
    rb_entities = [e for e in g.entities.values() if e.entity_type == EntityType.RUNBOOK]
    assert len(rb_entities) == 1
    assert rb_entities[0].entity_id == "rb-checkout-503"


def test_ingest_incident_links_runbook_to_service(store, mock_runbook_service):
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    g = ingestor.ingest_incident(_make_incident("checkout"), tenant="t1")
    rb_links = [l for l in g.links if l.dst.startswith("runbook:")]
    assert any(l.src == "service:checkout" for l in rb_links)


def test_ingest_incident_persists_to_store(store, mock_runbook_service):
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    incident = _make_incident("checkout")
    g = ingestor.ingest_incident(incident, tenant="t1")
    g2 = store.load("t1", incident.incident_id)
    assert len(g2.entities) == len(g.entities)


def test_ingest_incident_missing_context_skips_pod(store, mock_runbook_service):
    incident = Incident(
        incident_id="INC-2026-101",
        title="x",
        status="pending",
        severity="low",
        created_at=datetime.now(timezone.utc),
        context={},  # no service, no pod
    )
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    g = ingestor.ingest_incident(incident, tenant="t1")
    assert "service:unknown" not in g.entities
    pod_entities = [e for e in g.entities.values() if e.entity_type == EntityType.POD]
    assert pod_entities == []


def test_ingest_runbook_creates_entity_only(store, mock_runbook_service):
    ingestor = UModelIngestor(store=store, runbook_service=mock_runbook_service)
    e = ingestor.ingest_runbook("rb-x", tenant="t1")
    assert e.entity_type == EntityType.RUNBOOK
    assert e.entity_id == "rb-x"
    assert e.id == "runbook:rb-x"


def test_ingest_incident_without_runbook_service_works(store):
    incident = _make_incident("checkout")
    ingestor = UModelIngestor(store=store, runbook_service=None)
    g = ingestor.ingest_incident(incident, tenant="t1")
    assert "service:checkout" in g.entities
    # No runbook entities when service is None
    rb = [e for e in g.entities.values() if e.entity_type == EntityType.RUNBOOK]
    assert rb == []
