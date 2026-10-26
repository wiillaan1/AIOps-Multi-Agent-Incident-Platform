"""Test Prometheus unreachable → verification.prometheus_available=False.

模拟 Prometheus 抛 ConnectionError（query_instant 失败）→ verify_phase
捕获异常、pre_value/post_value 都为 None、rca_match_status="unavailable"、
plan_b_triggered=False、rca_rerun_count=0、incident.state == RESOLVED
（主管道仍正常跑完）。

设计说明（M8 修正）：
1. 原 brief 使用 sync TestClient + time.sleep，改用 AsyncClient + asyncio.sleep
   让 asyncio.create_task 调度的管道任务可以推进。
2. Prometheus mock 必须在整个轮询窗口内保持有效：polling loop 必须保留在
   `with patch(...)` 内部。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_pipeline_succeeds_when_prometheus_unreachable() -> None:
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

        def boom(query, *a, **kw):
            raise ConnectionError("prometheus not reachable")

        from app.api.routes import _incident_service

        with patch("app.api.routes.PrometheusClient") as mock_client:
            mock_client.return_value.query_instant = boom
            resp = await ac.post("/api/v1/incidents/trigger", json=body)
            assert resp.status_code == 202, resp.text
            incident_id = resp.json()["incident_id"]

            # 等 verify_phase 完成（必须在 patch 块内做轮询）
            deadline = time.time() + 30.0
            while time.time() < deadline:
                inc = _incident_service._incidents.get(incident_id)
                if inc is not None and "verification" in inc.context:
                    break
                await asyncio.sleep(0.1)

            inc = _incident_service._incidents.get(incident_id)
            assert inc is not None, f"incident {incident_id} disappeared"
            verification = inc.context.get("verification") or {}
            assert verification.get("prometheus_available") is False, (
                f"expected prometheus_available=False, got {verification}"
            )
            assert verification.get("plan_b_triggered") is False, (
                f"expected plan_b_triggered=False, got {verification}"
            )
            assert verification.get("rca_rerun_count") == 0, (
                f"expected rca_rerun_count=0, got {verification}"
            )
            # 主管道仍 RESOLVED
            assert inc.state.value == "resolved", (
                f"expected state=resolved, got {inc.state.value}"
            )