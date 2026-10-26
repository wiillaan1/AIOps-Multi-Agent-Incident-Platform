"""Test _run_pipeline_with_reflection writes audit_trail end-to-end."""

import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.incident import IncidentState


_TERMINAL_STATES = {
    IncidentState.RESOLVED,
    IncidentState.ESCALATED,
    IncidentState.CLOSED,
    IncidentState.CANCELLED,
}


@pytest.mark.asyncio
async def test_trigger_incident_writes_audit_trail():
    """触发故障 → 后台管道跑完（含 verify_phase）→ audit_trail 出现 verify_reflection。

    使用 AsyncClient + 共享事件循环（与 test_e2e_runbook_pipeline.py 一致），
    通过 await asyncio.sleep 把控制权交回事件循环，让 asyncio.create_task 调度的
    管道任务得以推进。
    """
    import asyncio
    import httpx
    from httpx import ASGITransport

    from app.api.routes import _incident_service

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        body = {
            "service": "order-service",
            "metric": "cpu_usage_percent",
            "value": 95.0,
            "threshold": 80.0,
            "severity": "high",
            "labels": {"tier": "critical"},
            "annotations": {},
        }
        # 模拟 Prometheus（5s 超时内的快速响应）
        from unittest.mock import patch as _patch
        with _patch(
            "app.api.routes.PrometheusClient"
        ) as mock_client:
            mock_client.return_value.query_instant = AsyncMock(return_value=95.0)
            resp = await ac.post("/api/v1/incidents/trigger", json=body)

        assert resp.status_code == 202, resp.text
        incident_id = resp.json()["incident_id"]

        # 轮询直到 pipeline 跑完（含 verify_phase → audit_trail 写入）
        deadline = time.time() + 30.0
        last_state = None
        found = False
        while time.time() < deadline:
            incident = _incident_service._incidents.get(incident_id)
            if incident is not None:
                last_state = incident.state
                trail = incident.context.get("audit_trail", [])
                if any(e.get("step") == "verify_reflection" for e in trail):
                    found = True
                    break
                # pipeline 仍在跑，继续 yield
                if last_state in _TERMINAL_STATES:
                    # state 进入终态但 audit_trail 还没写到 → 再等一会
                    pass
            await asyncio.sleep(0.2)

        assert incident is not None, f"incident {incident_id} disappeared"
        assert found, (
            f"audit_trail missing verify_reflection within timeout "
            f"(state={last_state}): {incident.context.get('audit_trail', [])}"
        )