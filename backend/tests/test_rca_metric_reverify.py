"""Test RCA mismatch triggers rerun and writes rca_reverify_rerun audit entry.

模拟 Prometheus 返回：第 1 次 query_instant (pre_value) = 20.0（远低于 RCA 认定的
高根因置信区间）→ verify_rca_against_metric 判定 mismatch → 触发 RCA 重跑。
第 2 次 query_instant (post_value) 可任意返回（rca 重跑阶段不再查 Prometheus）。

期望结果：
- verification.rca_match_status == "mismatch"
- verification.rca_rerun_count == 1
- audit_trail 里有 step="rca_reverify_rerun"

设计说明（M8 修正）：
1. 原 brief 使用 sync TestClient + time.sleep 轮询，但 sync TestClient
   不会把 asyncio.create_task 调度的管道任务推进到底。改用 AsyncClient +
   asyncio.sleep（与 test_pipeline_audit_trail.py 一致）。
2. Prometheus mock 必须在整个轮询窗口内保持有效：把 polling loop 保留在
   `with patch(...)` 内部，否则 patch 在 `await asyncio.sleep` 之前就被
   撤销了。
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
async def test_rca_mismatch_triggers_rerun() -> None:
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
        # 第一次 query_instant 返回 mismatch 值（高 root cause 但实际低）
        # 后面调用可任意返回（rca 重跑时不再查 prometheus）
        call_count = {"n": 0}

        async def mock_instant(query, *a, **kw):
            call_count["n"] += 1
            # pre_value(RCA核对) -> 不一致 (20.0 < threshold=80.0);
            # post_value(heal核对) -> 任意
            return 20.0 if call_count["n"] == 1 else 90.0

        from app.api.routes import _incident_service

        with patch("app.api.routes.PrometheusClient") as mock_client:
            mock_client.return_value.query_instant = mock_instant
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
            assert "verification" in inc.context, (
                f"verify_phase did not complete within 30s: "
                f"trail={inc.context.get('audit_trail', [])}"
            )

            verification = inc.context.get("verification") or {}
            assert verification.get("rca_match_status") == "mismatch", (
                f"expected rca_match_status=mismatch, got {verification}"
            )
            assert verification.get("rca_rerun_count") == 1, (
                f"expected rca_rerun_count=1, got {verification}"
            )
            trail = inc.context.get("audit_trail", [])
            assert any(e.get("step") == "rca_reverify_rerun" for e in trail), (
                f"audit_trail missing rca_reverify_rerun: {trail}"
            )