"""Test badcase analyze + list endpoints (Task 3 of W7.5 data flywheel)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.models.events import AlertEvent, SeverityLevel
from app.models.incident import Incident
from app.models.badcase import BadCaseEntry
from app.services.badcase_registry_service import BadCaseRegistryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_incident(
    incident_id: str = "INC-TEST-001",
    audit_trail: list[dict] | None = None,
    verification: dict | None = None,
) -> Incident:
    alert = AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.HIGH,
        labels={},
        annotations={},
    )
    inc = Incident.from_alert(alert)
    inc.incident_id = incident_id
    inc.context["audit_trail"] = audit_trail or []
    if verification is not None:
        inc.context["verification"] = verification
    return inc


def _make_entry(
    id_suffix: str,
    badcase_class: str = "rca_predict_mismatch",
    incident_id: str = "INC-PRIOR-001",
    eval_split: str = "dev",
    created_at: datetime | None = None,
) -> BadCaseEntry:
    return BadCaseEntry(
        id=f"bc-{id_suffix}",
        created_at=created_at or datetime.now(timezone.utc),
        badcase_class=badcase_class,
        incident_id=incident_id,
        audit_trail_excerpt=[{"step": "verify_reflection"}],
        identified_flaw="RCA confidence was 0.4 below threshold causing skip",
        keywords_for_retrieval=["rca", "low_confidence", "rca_mismatch"],
        suggestion_or_lesson="Add hard threshold check before skipping self-critique",
        severity="high",
        eval_split=eval_split,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_registry(tmp_path: Path):
    """Bare BadCaseRegistryService for tests that need to seed entries directly."""
    return BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch, fresh_registry: BadCaseRegistryService):
    """FastAPI TestClient with tmp-path BadCaseRegistryService + clean incident store.

    We monkeypatch the routes module's `_get_badcase_registry_service` so the
    endpoints return *the same* fresh_registry fixture instance that the test
    seeds directly. Both fixtures share the same `tmp_path` because pytest
    reuses the same `tmp_path` per-test when both fixtures declare it.
    """
    from app.api import routes as routes_module
    from app.main import app

    def _factory():
        return fresh_registry

    monkeypatch.setattr(routes_module, "_get_badcase_registry_service", _factory)
    routes_module._incident_service._incidents.clear()
    # Also reset the module-level cache so the patched factory is used.
    monkeypatch.setattr(routes_module, "_badcase_registry_service", None)

    return TestClient(app)


@pytest.fixture
def routes_module():
    """Return the routes module so tests can poke _incident_service directly."""
    from app.api import routes as rm

    return rm


# ---------------------------------------------------------------------------
# POST /api/v1/badcase/{incident_id}/analyze
# ---------------------------------------------------------------------------


class TestAnalyzeBadcaseEndpoint:
    def test_analyze_returns_200_with_draft_and_similar_cases(
        self,
        client: TestClient,
        routes_module,
    ) -> None:
        """Incident with capture trigger → 200 + draft + similar_cases (empty)."""
        incident = _make_incident(
            audit_trail=[{
                "step": "verify_reflection",
                "actions_taken": ["next_action=escalate"],
            }],
            verification={"rca_match_status": "mismatch"},
        )
        routes_module._incident_service._incidents[incident.incident_id] = incident

        resp = client.post(
            f"/api/v1/badcase/{incident.incident_id}/analyze"
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "draft" in data
        draft = data["draft"]
        # draft is a BadCaseEntry-shaped dict
        assert draft["incident_id"] == incident.incident_id
        assert draft["badcase_class"] == "rca_predict_mismatch"  # mismatch has priority
        assert draft["eval_split"] == "dev"
        assert "id" in draft and draft["id"].startswith("bc-")
        assert len(draft["keywords_for_retrieval"]) >= 3
        assert "similar_cases" in data
        assert data["similar_cases"] == []  # no prior entries

    def test_analyze_returns_404_when_incident_missing(
        self,
        client: TestClient,
    ) -> None:
        resp = client.post("/api/v1/badcase/INC-DOES-NOT-EXIST/analyze")
        assert resp.status_code == 404, resp.text
        assert "not found" in resp.json()["detail"].lower()

    def test_analyze_returns_400_when_no_triggers(
        self,
        client: TestClient,
        routes_module,
    ) -> None:
        """Clean run (no capture signals) → 400 with explanatory detail."""
        incident = _make_incident(
            audit_trail=[{
                "step": "verify_reflection",
                "actions_taken": ["next_action=complete"],
            }],
            verification={"rca_match_status": "match", "plan_b_triggered": False},
        )
        routes_module._incident_service._incidents[incident.incident_id] = incident

        resp = client.post(
            f"/api/v1/badcase/{incident.incident_id}/analyze"
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "no capture triggers" in detail.lower() or "no badcase" in detail.lower()


# ---------------------------------------------------------------------------
# GET /api/v1/badcase
# ---------------------------------------------------------------------------


class TestListBadcaseEndpoint:
    def test_list_returns_entries_sorted_by_created_at_desc(
        self,
        client: TestClient,
        fresh_registry: BadCaseRegistryService,
    ) -> None:
        # Pre-populate registry with three dev entries; one older, two newer.
        e_old = _make_entry(
            id_suffix="1" * 12,
            badcase_class="reflect_escalated",
            created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        )
        e_mid = _make_entry(
            id_suffix="2" * 12,
            badcase_class="rca_predict_mismatch",
            created_at=datetime(2026, 7, 10, tzinfo=timezone.utc),
        )
        e_new = _make_entry(
            id_suffix="3" * 12,
            badcase_class="heal_plan_b_triggered",
            created_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        )
        for entry in (e_old, e_mid, e_new):
            fresh_registry.create(entry)

        resp = client.get("/api/v1/badcase")
        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert len(items) == 3
        # Descending by created_at: new → mid → old
        assert items[0]["id"] == e_new.id
        assert items[1]["id"] == e_mid.id
        assert items[2]["id"] == e_old.id

    def test_list_filters_by_class_and_limit(
        self,
        client: TestClient,
        fresh_registry: BadCaseRegistryService,
    ) -> None:
        for i, cls in enumerate(
            ["rca_predict_mismatch", "rca_predict_mismatch", "reflect_escalated"]
        ):
            entry = _make_entry(
                id_suffix=f"{i:012x}",
                badcase_class=cls,
                eval_split="dev",
                created_at=datetime(2026, 7, i + 1, tzinfo=timezone.utc),
            )
            fresh_registry.create(entry)

        resp = client.get(
            "/api/v1/badcase?badcase_class=rca_predict_mismatch&limit=10"
        )
        assert resp.status_code == 200, resp.text
        items = resp.json()
        assert len(items) == 2
        assert all(item["badcase_class"] == "rca_predict_mismatch" for item in items)

    def test_list_rejects_invalid_limit(
        self,
        client: TestClient,
    ) -> None:
        # limit > 500 must be rejected by FastAPI Query(le=500)
        resp = client.get("/api/v1/badcase?limit=99999")
        assert resp.status_code == 422