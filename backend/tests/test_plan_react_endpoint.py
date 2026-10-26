"""Test POST /incidents/{id}/pipeline/plan-react (PlanReActRunner wrapper)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from app.main import app


def test_plan_react_endpoint_returns_task_dict():
    client = TestClient(app)

    # 先创建一个最小 incident（避免依赖真实触发）
    from app.models.events import AlertEvent, SeverityLevel
    from app.models.incident import Incident, IncidentState
    from app.api.routes import _incident_service

    alert = AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.HIGH,
        labels={},
        annotations={},
    )
    incident = asyncio.run(_incident_service.create_from_alert(alert))

    # Spec §4.2 step 1: plan-react 回放要求 incident 已完成 RCA —— 把新建 incident
    # 从 NEW 推进到 RCA_COMPLETED，否则新加的 state guard 会直接返回 409。
    asyncio.run(
        _incident_service.update_state(
            incident.incident_id, IncidentState.RCA_COMPLETED
        )
    )

    # mock 4 个 step 的 executor，避免真触发业务逻辑
    async def fake_step(step):
        return {"status": "success", "step_id": step.step_id}

    with patch("app.api.routes._pipeline_step_dispatch", side_effect=fake_step):
        resp = client.post(
            f"/api/v1/incidents/{incident.incident_id}/pipeline/plan-react",
            json={"requester": "test"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["incident_id"] == incident.incident_id
    task = body["task"]
    assert task["step_count"] == 4
    assert task["status"] == "completed"
    assert any(t["thought"] == "plan_generated" for t in task["react_trace"])


def test_plan_react_endpoint_rejects_pre_rca_state():
    """Spec §4.2: incident 处于 NEW（RCA 还没跑）时必须拒绝 plan-react 回放。"""
    client = TestClient(app)

    from app.models.events import AlertEvent, SeverityLevel
    from app.models.incident import Incident, IncidentState
    from app.api.routes import _incident_service

    alert = AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.HIGH,
        labels={},
        annotations={},
    )
    incident = asyncio.run(_incident_service.create_from_alert(alert))
    # 默认 state == NEW（RCA 还没完成），必被 guard 拒掉
    assert incident.state == IncidentState.NEW

    resp = client.post(
        f"/api/v1/incidents/{incident.incident_id}/pipeline/plan-react",
        json={"requester": "test"},
    )

    assert resp.status_code == 409
    body = resp.json()
    assert "plan-react replay" in body["detail"]
    assert IncidentState.NEW.value in body["detail"]