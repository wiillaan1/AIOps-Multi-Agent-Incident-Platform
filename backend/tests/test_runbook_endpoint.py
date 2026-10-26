"""
Tests for GET /incidents/{id}/runbook endpoint.

We inject a Runbook draft directly into incident.context via the seed-demo
endpoint, then verify the endpoint can retrieve it.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import _incident_service


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_incident_with_runbook(runbook: dict) -> str:
    """Trigger an incident then patch its context with a runbook draft."""
    client = TestClient(app)
    resp = client.post(
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

    # 直接写入 runbook_draft 到 context（模拟 Orchestrator 行为）
    incident = _incident_service._incidents[incident_id]
    incident.context["runbook_draft"] = runbook
    return incident_id


def test_runbook_endpoint_returns_draft(client: TestClient):
    sample = {
        "title": "[Runbook] order-service cpu_usage_percent 根因=resource_exhaustion",
        "service": "order-service",
        "root_cause": "resource_exhaustion",
        "confidence": 0.78,
        "sections": [{"heading": "概述", "kind": "overview", "text": "test"}],
        "markdown": "# Runbook\n\n## 概述\n\ntest\n",
        "source": {"type": "auto_generated", "log_evidence_available": True},
    }
    incident_id = _seed_incident_with_runbook(sample)

    resp = client.get(f"/api/v1/incidents/{incident_id}/runbook")
    assert resp.status_code == 200
    body = resp.json()
    assert body["incident_id"] == incident_id
    assert body["runbook"]["title"] == sample["title"]
    assert body["runbook"]["confidence"] == 0.78
    assert body["runbook"]["sections"][0]["kind"] == "overview"


def test_runbook_endpoint_404_when_no_draft(client: TestClient):
    """故障存在但 Runbook 尚未生成时返回 404。"""
    # 触发一个新告警但不写 runbook_draft
    resp = client.post(
        "/api/v1/incidents/trigger",
        json={
            "service": "payment-service",
            "metric": "error_rate_percent",
            "severity": "critical",
            "value": 0.5,
            "threshold": 0.05,
            "source": "test",
        },
    )
    assert resp.status_code in (201, 202), resp.text
    incident_id = resp.json()["incident_id"]

    resp = client.get(f"/api/v1/incidents/{incident_id}/runbook")
    assert resp.status_code == 404
    assert "Runbook draft not available" in resp.json()["detail"]


def test_runbook_endpoint_404_when_incident_missing(client: TestClient):
    resp = client.get("/api/v1/incidents/INC-DOES-NOT-EXIST/runbook")
    assert resp.status_code == 404


def test_runbook_endpoint_preserves_log_evidence_samples(client: TestClient):
    """确保日志证据样本能在端点中保留下来。"""
    sample = {
        "title": "[Runbook] x",
        "service": "order-service",
        "root_cause": "resource_exhaustion",
        "confidence": 0.8,
        "sections": [
            {
                "heading": "日志证据（Loki）",
                "kind": "log_evidence",
                "text": "ERROR OOM killed",
                "available": True,
                "samples_count": 1,
            }
        ],
        "markdown": "# x",
        "source": {"type": "auto_generated", "log_evidence_available": True},
    }
    incident_id = _seed_incident_with_runbook(sample)

    resp = client.get(f"/api/v1/incidents/{incident_id}/runbook")
    assert resp.status_code == 200
    sections = resp.json()["runbook"]["sections"]
    log_section = next(s for s in sections if s["kind"] == "log_evidence")
    assert log_section["available"] is True
    assert log_section["samples_count"] == 1