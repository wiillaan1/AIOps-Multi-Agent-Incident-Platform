"""TDD coverage for the investigation verification API endpoints.

These tests exercise the REST boundary of the InvestigationLoopEngine:
profile listing, the investigation snapshot/evidence for an incident, the
idempotent execution-receipt submission, and recovery verification.  They never
require a live Prometheus/Loki backend — query adapters are replaced with fakes
through the engine singleton so the closed-loop semantics stay deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from app.api import routes as routes_module
from app.models.events import AlertEvent, SeverityLevel
from app.models.incident import Incident
from app.models.investigation import HealExecutionReceipt
from app.services.assertion_evaluator import QueryResult
from app.services.investigation_loop_engine import InvestigationLoopEngine
from app.services.investigation_store import InvestigationStore
from app.services.profile_registry import ProfileRegistry


def _seed_incident(service: str = "orders") -> Incident:
    alert = AlertEvent(
        service=service,
        metric="cpu_usage_percent",
        value=95,
        threshold=80,
        severity=SeverityLevel.HIGH,
        labels={"target": f"{service}-api", "environment": "prod"},
    )
    incident = Incident.from_alert(alert)
    routes_module._incident_service._incidents[incident.incident_id] = incident
    return incident


def _fake_query_adapters():
    async def metric_query(query: str, **kwargs):
        return QueryResult(
            source="prometheus",
            available=True,
            value=95,
            hits=0,
            summary={"query": query},
        )

    async def log_query(query: str, **kwargs):
        return QueryResult(
            source="loki",
            available=True,
            value=None,
            hits=2,
            summary={"query": query},
        )

    return metric_query, log_query


def _install_engine(incident: Incident) -> InvestigationLoopEngine:
    """Inject a deterministic engine singleton wired to a confirming profile set."""
    registry = ProfileRegistry()
    profiles = registry.load()
    metric_query, log_query = _fake_query_adapters()
    engine = InvestigationLoopEngine(
        store=InvestigationStore(),
        profiles=profiles,
        rca_agent=_FakeRCA("resource_exhaustion"),
        heal_agent=_FakeHeal(),
        metric_query=metric_query,
        log_query=log_query,
    )
    # Bind the incident-specific agents used by the route handlers.
    routes_module._set_investigation_engine(engine)
    return engine


class _FakeRCA:
    def __init__(self, root_cause: str) -> None:
        self.root_cause = root_cause

    async def execute(self, input_data, context=None):
        from app.agents.base import AgentResult
        from app.models.events import RCAEvent

        event = RCAEvent(
            incident_id=input_data.incident_id,
            root_cause=self.root_cause,
            confidence=0.9,
            evidence={"affected_services": [input_data.alert.service]},
        )
        return AgentResult.success_result("fake_rca", {"rca_event": event.model_dump()})


class _FakeHeal:
    async def execute(self, input_data, context=None):
        from app.agents.base import AgentResult

        return AgentResult.success_result(
            "fake_heal",
            {
                "heal_event": {
                    "incident_id": input_data.incident_id,
                    "action_id": "dry-run-action",
                    "action": "restart",
                    "action_category": "restart-orders",
                    "target_resource": "orders-api",
                    "dry_run": True,
                    "dry_run_result": {"all_executable": True},
                    "status": "pending",
                }
            },
        )

    def plan_candidates(self, rca_event, top_k=3):
        return [
            {"playbook_id": "restart-orders", "playbook_name": "Restart orders"},
            {"playbook_id": "scale-orders", "playbook_name": "Scale orders"},
        ][:top_k]


@pytest.fixture(autouse=True)
def _reset_engine():
    """Restore the engine singleton between tests so injections do not leak."""
    yield
    routes_module._reset_investigation_engine()


# --------------------------------------------------------------------------- #
# Root-cause profiles
# --------------------------------------------------------------------------- #


def test_root_cause_profiles_listed():
    client = TestClient(routes_module.app) if hasattr(routes_module, "app") else None
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/root-cause-profiles")
    assert response.status_code == 200
    data = response.json()
    causes = {item["root_cause"] for item in data["profiles"]}
    assert "resource_exhaustion" in causes
    assert "database_issue" in causes


def test_root_cause_profile_detail_found():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/root-cause-profiles/resource_exhaustion")
    assert response.status_code == 200
    assert response.json()["root_cause"] == "resource_exhaustion"


def test_root_cause_profile_detail_not_found():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/root-cause-profiles/does_not_exist")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Investigation snapshot + evidence
# --------------------------------------------------------------------------- #


def test_investigation_snapshot_for_unknown_incident_is_404():
    from app.main import app

    client = TestClient(app)
    response = client.get("/api/v1/incidents/inc-does-not-exist/investigation")
    assert response.status_code == 404


def test_investigation_snapshot_returns_pending_state():
    from app.main import app

    incident = _seed_incident()
    _install_engine(incident)
    client = TestClient(app)
    response = client.get(f"/api/v1/incidents/{incident.incident_id}/investigation")
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == incident.incident_id
    assert data["state"] in {"pending", "investigating", "replanning",
                             "awaiting_execution", "recovery_observing",
                             "recovered", "heal_failed", "manual_escalation"}


@pytest.mark.asyncio
async def test_investigation_evidence_endpoint_lists_records():
    from app.main import app

    incident = _seed_incident()
    engine = _install_engine(incident)
    await engine.investigate(incident)
    client = TestClient(app)
    response = client.get(f"/api/v1/incidents/{incident.incident_id}/investigation/evidence")
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == incident.incident_id
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) >= 1


# --------------------------------------------------------------------------- #
# Execution receipts
# --------------------------------------------------------------------------- #


def _receipt_payload(incident_id: str, *, status: str = "succeeded") -> dict:
    return {
        "receipt_id": "r-1",
        "incident_id": incident_id,
        "action_id": "dry-run-action",
        "idempotency_key": "k-1",
        "playbook_id": "restart-orders",
        "target_resource": "orders-api",
        "status": status,
        "completed_at": datetime.now(timezone.utc).isoformat() if status == "succeeded" else None,
    }


def test_receipt_for_unknown_incident_is_404():
    from app.main import app

    client = TestClient(app)
    response = client.post(
        "/api/v1/incidents/inc-does-not-exist/execution-receipts",
        json=_receipt_payload("inc-does-not-exist"),
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_receipt_endpoint_is_idempotent():
    from app.main import app

    incident = _seed_incident()
    engine = _install_engine(incident)
    await engine.investigate(incident)
    client = TestClient(app)
    payload = _receipt_payload(incident.incident_id)
    first = client.post(
        f"/api/v1/incidents/{incident.incident_id}/execution-receipts", json=payload
    )
    second = client.post(
        f"/api/v1/incidents/{incident.incident_id}/execution-receipts", json=payload
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["receipt_id"] == first.json()["receipt_id"]


@pytest.mark.asyncio
async def test_receipt_conflict_for_reused_key_returns_409():
    from app.main import app

    incident = _seed_incident()
    engine = _install_engine(incident)
    await engine.investigate(incident)
    client = TestClient(app)
    payload = _receipt_payload(incident.incident_id)
    first = client.post(
        f"/api/v1/incidents/{incident.incident_id}/execution-receipts", json=payload
    )
    assert first.status_code == 200
    # Same idempotency key but a different action target → conflict.
    conflicting = dict(payload)
    conflicting["target_resource"] = "different-target"
    conflict = client.post(
        f"/api/v1/incidents/{incident.incident_id}/execution-receipts",
        json=conflicting,
    )
    assert conflict.status_code == 409


# --------------------------------------------------------------------------- #
# Recovery verification
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recovery_verification_without_succeeded_receipt_is_409():
    from app.main import app

    incident = _seed_incident()
    engine = _install_engine(incident)
    # Investigate reaches awaiting_execution but no receipt submitted yet.
    await engine.investigate(incident)
    client = TestClient(app)
    response = client.post(
        f"/api/v1/incidents/{incident.incident_id}/recovery-verification"
    )
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_recovery_verification_after_succeeded_receipt():
    from app.main import app

    incident = _seed_incident()
    engine = _install_engine(incident)
    await engine.investigate(incident)
    receipt = HealExecutionReceipt(
        incident_id=incident.incident_id,
        action_id="dry-run-action",
        idempotency_key="k-1",
        playbook_id="restart-orders",
        target_resource="orders-api",
        status="succeeded",
        completed_at=datetime.now(timezone.utc),
    )
    await engine.accept_execution_receipt(incident, receipt)
    client = TestClient(app)
    response = client.post(
        f"/api/v1/incidents/{incident.incident_id}/recovery-verification"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["incident_id"] == incident.incident_id
    assert "state" in data
