"""Test heal Plan B triggers when post_value still exceeds threshold + severity critical.

模拟：severity=critical 故障 + pre_value=high (match) + post_value=still high
(not recovered) → verify_heal_and_try_plan_b 判定 plan_b_eligible=True 且
recovered=False → verify_phase 调用 _execute_plan_b → plan_b_triggered=True。

期望结果：
- verification.plan_b_triggered == True
- verification.fallback_playbook_id 不为 None
- audit_trail 里有 step="heal_plan_b_retry"

设计说明：
1. M8 修正：原 brief 使用 sync TestClient + time.sleep，改用 AsyncClient +
   asyncio.sleep 让 asyncio.create_task 调度的管道任务可以推进。
2. Prometheus mock 必须在整个轮询窗口内保持有效：polling loop 保留在
   `with patch(...)` 内部。
3. Plan B fallback 选择依赖 plan_candidates 返回 top-K 不同 playbook。但当前
   implementation 的 plan_candidates 把每次 _match_playbook 都返回同一个 best
   match，导致 top-3 全部为同一 playbook。属于上游实现局限（pre-existing bug
   in heal_agent.plan_candidates），本次不做 source fix。改用直接 mock
   `_execute_plan_b` 返回固定 fallback_playbook_id，绕开该 bug，聚焦验证
   verify_phase 的 Plan B 调用链 + audit_trail 写入。
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_heal_plan_b_triggers_for_critical_severity() -> None:
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        body = {
            "service": "order-service",
            "metric": "cpu_usage_percent",
            "value": 95.0,
            "threshold": 80.0,
            "severity": "critical",
            "labels": {"tier": "critical"},
            "annotations": {},
        }
        # pre=high (match)，post=still high (not recovered) → Plan B 触发
        async def mock_instant(query, *a, **kw):
            return 95.0 if "cpu_usage_percent" in query else 90.0

        from app.api.routes import _incident_service

        with patch("app.api.routes.PrometheusClient") as mock_client, patch(
            "app.api.routes._execute_plan_b",
            new=AsyncMock(return_value="pb_fallback_dummy"),
        ):
            mock_client.return_value.query_instant = mock_instant
            # 强制 _execute_plan_b 返回有效 fallback ID，绕开 plan_candidates
            # pre-existing bug（返回同 playbook top-3）

            resp = await ac.post("/api/v1/incidents/trigger", json=body)
            assert resp.status_code == 202, resp.text
            incident_id = resp.json()["incident_id"]

            # 等 plan_b_triggered 写入（必须在 patch 块内做轮询）
            deadline = time.time() + 40.0
            while time.time() < deadline:
                inc = _incident_service._incidents.get(incident_id)
                if (
                    inc is not None
                    and inc.context.get("verification", {}).get("plan_b_triggered")
                ):
                    break
                await asyncio.sleep(0.1)

            inc = _incident_service._incidents.get(incident_id)
            assert inc is not None, f"incident {incident_id} disappeared"
            verification = inc.context.get("verification") or {}
            assert verification.get("plan_b_triggered") is True, (
                f"expected plan_b_triggered=True, got {verification}"
            )
            assert verification.get("fallback_playbook_id") == "pb_fallback_dummy", (
                f"expected fallback_playbook_id=pb_fallback_dummy, got {verification}"
            )
            trail = inc.context.get("audit_trail", [])
            assert any(e.get("step") == "heal_plan_b_retry" for e in trail), (
                f"audit_trail missing heal_plan_b_retry: {trail}"
            )