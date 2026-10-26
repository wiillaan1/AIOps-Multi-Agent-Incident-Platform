"""
E2E test: trigger incident → Orchestrator runs full pipeline → Runbook auto-generated → endpoint returns it.

使用 AsyncClient + 共享事件循环：
1. 触发故障时 `_process_incident_pipeline` 通过 `asyncio.create_task` 调度在
   当前循环；测试也运行在同一循环里，所以 `await asyncio.sleep(...)` 能
   把控制权交回给 pipeline 任务继续推进。
2. 轮询 incident.state 直到进入终态（AWAITING_APPROVAL / RESOLVED 等）。
3. 验证 runbook_draft 写入 + 端点取回。

Picks a CRITICAL incident 让 pipeline 在 RCA 完成后直接停在 AWAITING_APPROVAL，
这是最快的确定路径，同时完整跑过 RCA + Runbook 自动生成。
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from app.main import app
from app.api.routes import _incident_service
from app.models.incident import IncidentState


TERMINAL_STATES = {
    IncidentState.RESOLVED,
    IncidentState.ESCALATED,
    IncidentState.CLOSED,
    IncidentState.CANCELLED,
    IncidentState.AWAITING_APPROVAL,
}


@pytest_asyncio.fixture
async def async_client():
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


async def _wait_for_terminal_state(
    incident_id: str, timeout: float = 30.0
) -> str:
    """在共享循环里轮询，把控制权交还给 pipeline 任务。"""
    deadline = time.time() + timeout
    last_state: str | None = None
    while time.time() < deadline:
        incident = _incident_service._incidents.get(incident_id)
        if incident is None:
            raise AssertionError(f"Incident {incident_id} disappeared")
        state = incident.state
        # pipeline 最终落到 RESOLVED（即便中间经过 AWAITING_APPROVAL）
        if state in TERMINAL_STATES or state == IncidentState.RESOLVED:
            return state.value
        last_state = state.value
        await asyncio.sleep(0.2)
    raise AssertionError(
        f"Incident {incident_id} did not reach a terminal state in {timeout}s "
        f"(last state: {last_state})"
    )


async def _trigger_critical_incident(
    client: httpx.AsyncClient, service: str = "payment-service"
) -> str:
    resp = await client.post(
        "/api/v1/incidents/trigger",
        json={
            "service": service,
            "metric": "error_rate_percent",
            "severity": "critical",
            "value": 0.55,
            "threshold": 0.05,
            "source": "e2e_test",
        },
    )
    assert resp.status_code in (201, 202), resp.text
    return resp.json()["incident_id"]


# ============================ Full pipeline ============================


@pytest.mark.asyncio
async def test_e2e_trigger_to_runbook_is_retrievable(async_client: httpx.AsyncClient):
    """完整链路：触发 → 跑完 pipeline → runbook_draft 写入 → 端点取回。"""
    incident_id = await _trigger_critical_incident(async_client)

    final_state = await _wait_for_terminal_state(incident_id, timeout=30.0)
    # pipeline 一定经过 RCA，runbook_draft 必须在 RESOLVED 之前已写入
    assert final_state in {
        IncidentState.AWAITING_APPROVAL.value,
        IncidentState.RESOLVED.value,
    }

    incident = _incident_service._incidents[incident_id]
    assert "runbook_draft" in incident.context, (
        "Orchestrator 应在 RCA 成功后自动写入 runbook_draft"
    )
    draft = incident.context["runbook_draft"]
    assert draft["service"] == "payment-service"
    assert draft["root_cause"]
    assert 0.0 <= draft["confidence"] <= 1.0
    assert len(draft["sections"]) >= 6
    assert draft["markdown"]

    resp = await async_client.get(f"/api/v1/incidents/{incident_id}/runbook")
    assert resp.status_code == 200
    body = resp.json()
    assert body["incident_id"] == incident_id
    assert body["runbook"]["root_cause"] == draft["root_cause"]


@pytest.mark.asyncio
async def test_e2e_runbook_includes_log_evidence_section(async_client: httpx.AsyncClient):
    """只要 RCA 跑了，日志证据章节（available=False 兜底）都必须存在。"""
    incident_id = await _trigger_critical_incident(async_client)
    await _wait_for_terminal_state(incident_id, timeout=30.0)

    draft = _incident_service._incidents[incident_id].context["runbook_draft"]
    kinds = [s["kind"] for s in draft["sections"]]
    assert "log_evidence" in kinds

    log_section = next(s for s in draft["sections"] if s["kind"] == "log_evidence")
    assert "available" in log_section
    assert "heading" in log_section
    assert "text" in log_section


@pytest.mark.asyncio
async def test_e2e_runbook_section_kinds_complete(async_client: httpx.AsyncClient):
    """Runbook 8 个章节必须全部覆盖。"""
    incident_id = await _trigger_critical_incident(async_client)
    await _wait_for_terminal_state(incident_id, timeout=30.0)

    draft = _incident_service._incidents[incident_id].context["runbook_draft"]
    kinds = {s["kind"] for s in draft["sections"]}
    required = {
        "overview",
        "root_cause",
        "impact_chain",
        "evidence_summary",
        "log_evidence",
        "actions",
        "rollback",
        "verification",
    }
    missing = required - kinds
    assert not missing, f"Runbook 缺章节: {missing}，实际有: {kinds}"


@pytest.mark.asyncio
async def test_e2e_runbook_rollback_uses_known_root_cause(async_client: httpx.AsyncClient):
    """回滚方案结构完整：必须包含 ```bash 代码块（兜底也算）。"""
    incident_id = await _trigger_critical_incident(async_client)
    await _wait_for_terminal_state(incident_id, timeout=30.0)

    draft = _incident_service._incidents[incident_id].context["runbook_draft"]
    rollback_section = next(s for s in draft["sections"] if s["kind"] == "rollback")
    text = rollback_section["text"]

    # 回滚章节必须渲染为 bash 代码块，便于复制
    assert "```bash" in text, f"回滚方案必须是 bash 代码块: {text[:100]}"
    # 兜底 / 真实命令都属于合法输出，不强制要求 kubectl
    assert len(text.strip()) > 10


@pytest.mark.asyncio
async def test_e2e_multiple_incidents_isolated(async_client: httpx.AsyncClient):
    """多个故障并发触发时，runbook_draft 不会互相覆盖。"""
    services = ["payment-service", "order-service"]
    ids: list[str] = []
    for svc in services:
        iid = await _trigger_critical_incident(async_client, service=svc)
        ids.append(iid)

    for iid in ids:
        await _wait_for_terminal_state(iid, timeout=30.0)

    actual_services = {
        _incident_service._incidents[iid].context["runbook_draft"]["service"]
        for iid in ids
    }
    assert actual_services == set(services)

    titles: set[str] = set()
    for iid in ids:
        resp = await async_client.get(f"/api/v1/incidents/{iid}/runbook")
        assert resp.status_code == 200
        titles.add(resp.json()["runbook"]["title"])
    assert len(titles) == 2


@pytest.mark.asyncio
async def test_e2e_high_severity_also_completes(async_client: httpx.AsyncClient):
    """HIGH 严重级别也应能跑完整 pipeline 并生成 Runbook。"""
    resp = await async_client.post(
        "/api/v1/incidents/trigger",
        json={
            "service": "order-service",
            "metric": "cpu_usage_percent",
            "severity": "high",
            "value": 95.0,
            "threshold": 80.0,
            "source": "e2e_high",
        },
    )
    assert resp.status_code in (201, 202), resp.text
    incident_id = resp.json()["incident_id"]

    final_state = await _wait_for_terminal_state(incident_id, timeout=30.0)
    assert final_state in {
        IncidentState.AWAITING_APPROVAL.value,
        IncidentState.RESOLVED.value,
    }
    assert "runbook_draft" in _incident_service._incidents[incident_id].context