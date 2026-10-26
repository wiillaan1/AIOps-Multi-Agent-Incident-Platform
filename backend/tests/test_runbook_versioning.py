"""
Tests for Runbook Version Store + API endpoints.

Covers:
- RunbookVersionStore: publish / archive / list / get / latest_published
- API: POST publish, POST archive, GET versions, GET version
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import _incident_service
from app.services.runbook_version_store import RunbookVersionStore


# ============================ Store unit tests ============================


@pytest.fixture
def fresh_store():
    return RunbookVersionStore()


def _draft(title: str = "Test Runbook", **overrides) -> dict:
    base = {
        "title": title,
        "service": "order-service",
        "root_cause": "resource_exhaustion",
        "confidence": 0.75,
        "markdown": "# Test\n",
        "sections": [{"heading": "概述", "kind": "overview", "text": "test"}],
    }
    base.update(overrides)
    return base


def test_publish_returns_v1_on_first_call(fresh_store):
    v = fresh_store.publish("INC-1", _draft())
    assert v.version_number == 1
    assert v.status == "published"
    assert v.incident_id == "INC-1"


def test_publish_increments_version_monotonically(fresh_store):
    fresh_store.publish("INC-1", _draft(title="v1"))
    fresh_store.publish("INC-1", _draft(title="v2"))
    fresh_store.publish("INC-1", _draft(title="v3"))

    versions = fresh_store.list_versions("INC-1")
    assert [v["version_number"] for v in versions] == [1, 2, 3]
    titles = [v["title"] for v in versions]
    assert titles == ["v1", "v2", "v3"]


def test_publish_isolated_per_incident(fresh_store):
    fresh_store.publish("INC-A", _draft())
    fresh_store.publish("INC-A", _draft())
    fresh_store.publish("INC-B", _draft())

    assert len(fresh_store.list_versions("INC-A")) == 2
    assert len(fresh_store.list_versions("INC-B")) == 1


def test_archive_marks_version_archived(fresh_store):
    fresh_store.publish("INC-1", _draft())
    fresh_store.publish("INC-1", _draft())
    archived = fresh_store.archive("INC-1", version_number=1, change_note="obsolete")

    assert archived.status == "archived"
    assert "archive: obsolete" in archived.change_note

    # latest_published 应跳过 archived 返回 v2
    latest = fresh_store.latest_published("INC-1")
    assert latest is not None
    assert latest.version_number == 2


def test_archive_missing_version_raises_keyerror(fresh_store):
    fresh_store.publish("INC-1", _draft())
    with pytest.raises(KeyError):
        fresh_store.archive("INC-1", version_number=99)


def test_list_versions_returns_summaries(fresh_store):
    fresh_store.publish("INC-1", _draft(title="full title"))
    versions = fresh_store.list_versions("INC-1")
    assert len(versions) == 1
    summary = versions[0]
    # 摘要不含 markdown / sections
    assert "markdown" not in summary
    assert "sections" not in summary
    assert summary["title"] == "full title"


def test_get_version_returns_full_dict(fresh_store):
    fresh_store.publish("INC-1", _draft(markdown="# A\n## B\n"))
    full = fresh_store.get_version("INC-1", 1)
    assert full is not None
    assert full.markdown == "# A\n## B\n"
    assert full.to_dict()["markdown"] == "# A\n## B\n"


def test_latest_published_returns_none_when_no_versions(fresh_store):
    assert fresh_store.latest_published("INC-DOES-NOT-EXIST") is None


# ============================ API endpoint tests ============================


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_incident_with_runbook(runbook: dict) -> str:
    resp = TestClient(app).post(
        "/api/v1/incidents/trigger",
        json={
            "service": "order-service",
            "metric": "cpu_usage_percent",
            "severity": "high",
            "value": 95.0,
            "threshold": 80.0,
            "source": "test",
        },
    )
    assert resp.status_code in (201, 202), resp.text
    incident_id = resp.json()["incident_id"]
    _incident_service._incidents[incident_id].context["runbook_draft"] = runbook
    return incident_id


def test_api_publish_creates_version(client: TestClient):
    incident_id = _seed_incident_with_runbook(_draft(title="first publish"))

    resp = client.post(
        f"/api/v1/incidents/{incident_id}/runbook/publish",
        json={"change_note": "initial", "published_by": "tester"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]["version_number"] == 1
    assert body["version"]["status"] == "published"
    assert body["version"]["change_note"] == "initial"
    assert body["version"]["published_by"] == "tester"


def test_api_publish_404_when_no_draft(client: TestClient):
    # 触发但不写 draft
    resp = client.post(
        "/api/v1/incidents/trigger",
        json={
            "service": "x", "metric": "y", "severity": "high",
            "value": 1, "threshold": 0, "source": "test",
        },
    )
    iid = resp.json()["incident_id"]
    resp = client.post(f"/api/v1/incidents/{iid}/runbook/publish", json={})
    assert resp.status_code == 404


def test_api_list_versions_returns_in_order(client: TestClient):
    incident_id = _seed_incident_with_runbook(_draft(title="v1"))

    for title in ["v1", "v2", "v3"]:
        _incident_service._incidents[incident_id].context["runbook_draft"] = _draft(
            title=title
        )
        client.post(f"/api/v1/incidents/{incident_id}/runbook/publish", json={})

    resp = client.get(f"/api/v1/incidents/{incident_id}/runbook/versions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 3
    assert [v["version_number"] for v in body["versions"]] == [1, 2, 3]


def test_api_archive_marks_and_keeps(client: TestClient):
    incident_id = _seed_incident_with_runbook(_draft())
    client.post(f"/api/v1/incidents/{incident_id}/runbook/publish", json={})
    client.post(f"/api/v1/incidents/{incident_id}/runbook/publish", json={})

    resp = client.post(
        f"/api/v1/incidents/{incident_id}/runbook/archive",
        json={"version_number": 1, "change_note": "superseded"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]["status"] == "archived"

    versions = client.get(
        f"/api/v1/incidents/{incident_id}/runbook/versions"
    ).json()["versions"]
    statuses = {v["version_number"]: v["status"] for v in versions}
    assert statuses == {1: "archived", 2: "published"}


def test_api_get_version_returns_full(client: TestClient):
    incident_id = _seed_incident_with_runbook(
        _draft(markdown="# Full\n## Section\n", title="full version")
    )
    client.post(f"/api/v1/incidents/{incident_id}/runbook/publish", json={})

    resp = client.get(f"/api/v1/incidents/{incident_id}/runbook/versions/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]["markdown"] == "# Full\n## Section\n"
    assert body["version"]["title"] == "full version"


def test_api_archive_missing_version_returns_404(client: TestClient):
    incident_id = _seed_incident_with_runbook(_draft())
    resp = client.post(
        f"/api/v1/incidents/{incident_id}/runbook/archive",
        json={"version_number": 99},
    )
    assert resp.status_code == 404


def test_api_archive_bad_payload_returns_400(client: TestClient):
    incident_id = _seed_incident_with_runbook(_draft())
    resp = client.post(
        f"/api/v1/incidents/{incident_id}/runbook/archive",
        json={"version_number": "not-a-number"},
    )
    assert resp.status_code == 400


def test_api_versions_endpoint_404_for_missing_incident(client: TestClient):
    resp = client.get("/api/v1/incidents/INC-NOPE/runbook/versions")
    assert resp.status_code == 404