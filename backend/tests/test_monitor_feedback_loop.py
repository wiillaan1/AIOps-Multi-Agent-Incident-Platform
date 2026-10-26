"""Test monitor.record_algorithm_feedback is called from verify_phase.

Trigger incident → 主管道跑完 → verify_phase 末尾对 3 个算法都调用一次
monitor.record_algorithm_feedback(key, algo, success_flag)。

设计说明（M8 修正）：原 brief 使用 sync TestClient + time.sleep 轮询，但
sync TestClient 不会把 asyncio.create_task 调度的管道任务推进到底，导致
测试永远超时。改用 AsyncClient + asyncio.sleep（与 test_pipeline_audit_trail.py
一致），让控制权交回事件循环让 pipeline 任务推进。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from httpx import ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_monitor_feedback_called_three_times_after_pipeline() -> None:
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
        with patch(
            "app.api.routes.PrometheusClient"
        ) as mock_client, patch(
            "app.api.routes._get_monitor"
        ) as mock_get_monitor:
            mock_client.return_value.query_instant = AsyncMock(return_value=95.0)
            mock_monitor = mock_get_monitor.return_value
            # 用 sync Mock 记录 calls (record_algorithm_feedback 是同步方法)
            mock_monitor.record_algorithm_feedback = Mock()

            resp = await ac.post("/api/v1/incidents/trigger", json=body)
            assert resp.status_code == 202, resp.text
            incident_id = resp.json()["incident_id"]

            # 等异步管道（含 verify_phase）跑完
            from app.api.routes import _incident_service

            deadline = time.time() + 30.0
            while time.time() < deadline:
                inc = _incident_service._incidents.get(incident_id)
                if inc is not None and "verification" in inc.context:
                    break
                await asyncio.sleep(0.1)

            inc = _incident_service._incidents.get(incident_id)
            assert inc is not None, f"incident {incident_id} disappeared"
            assert "verification" in inc.context, (
                f"verify_phase did not complete within 30s: "
                f"trail={inc.context.get('audit_trail', [])}"
            )

            # 主管道应该对 3 个算法都调过一次 feedback
            assert mock_monitor.record_algorithm_feedback.call_count >= 3, (
                f"expected >=3 calls, got "
                f"{mock_monitor.record_algorithm_feedback.call_count}: "
                f"{mock_monitor.record_algorithm_feedback.call_args_list}"
            )

            for call in mock_monitor.record_algorithm_feedback.call_args_list[:3]:
                args, _ = call
                # args = (key, algo, success_flag)
                assert args[0] == "order-service:cpu_usage_percent", args
                assert args[1] in {"3-sigma", "ewma", "isolation_forest"}, args