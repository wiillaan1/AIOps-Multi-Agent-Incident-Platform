#!/usr/bin/env python3
"""
Runbook E2E smoke script.

手动验收用：触发一个 critical 故障 → 等 pipeline 跑完 → 取回 Runbook →
打印关键章节 + 落盘 Markdown 文件到 data/runbooks/。

用法：
    python scripts/runbook_smoke.py [service] [metric]
    python scripts/runbook_smoke.py order-service cpu_usage_percent
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
from httpx import ASGITransport

from app.main import app
from app.api.routes import _incident_service
from app.models.incident import IncidentState


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "runbooks"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def main(service: str, metric: str) -> None:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://smoke") as client:
        # 1. 触发故障
        print(f"\n[1/4] Triggering incident: service={service} metric={metric}")
        resp = await client.post(
            "/api/v1/incidents/trigger",
            json={
                "service": service,
                "metric": metric,
                "severity": "critical",
                "value": 95.0,
                "threshold": 80.0,
                "source": "smoke_script",
            },
        )
        if resp.status_code not in (201, 202):
            print(f"FAIL: trigger returned {resp.status_code}: {resp.text}")
            sys.exit(1)
        incident_id = resp.json()["incident_id"]
        print(f"      ✓ incident_id = {incident_id}")

        # 2. 轮询等 pipeline 跑完
        print("\n[2/4] Waiting for pipeline completion (max 60s)...")
        deadline = time.time() + 60
        while time.time() < deadline:
            incident = _incident_service._incidents.get(incident_id)
            if incident is None:
                print("FAIL: incident disappeared")
                sys.exit(1)
            if incident.state in {IncidentState.RESOLVED, IncidentState.AWAITING_APPROVAL}:
                break
            await asyncio.sleep(0.3)
        else:
            print(f"FAIL: pipeline did not complete in 60s, last state: {incident.state}")
            sys.exit(1)
        print(f"      ✓ final state = {incident.state.value}")
        print(f"      ✓ runbook_draft present: {'runbook_draft' in incident.context}")

        # 3. 取回 Runbook
        print("\n[3/4] Fetching Runbook via GET /incidents/{id}/runbook")
        resp = await client.get(f"/api/v1/incidents/{incident_id}/runbook")
        if resp.status_code != 200:
            print(f"FAIL: endpoint returned {resp.status_code}: {resp.text}")
            sys.exit(1)
        body = resp.json()
        draft = body["runbook"]
        print(f"      ✓ title     = {draft['title']}")
        print(f"      ✓ service   = {draft['service']}")
        print(f"      ✓ root_cause= {draft['root_cause']}")
        print(f"      ✓ confidence= {draft['confidence']:.2%}")
        print(f"      ✓ sections  = {len(draft['sections'])}")
        kinds = [s["kind"] for s in draft["sections"]]
        print(f"      ✓ section kinds = {kinds}")

        log_section = next((s for s in draft["sections"] if s["kind"] == "log_evidence"), None)
        if log_section:
            print(f"      ✓ log_evidence: available={log_section.get('available')}, entries={log_section.get('entries_count', 0)}, samples={log_section.get('samples_count', 0)}")
        if draft.get("source", {}).get("log_evidence_available"):
            print("      ✓ source flagged log_evidence_available")

        # 4. 落盘 Markdown
        out_path = OUTPUT_DIR / f"{incident_id}.md"
        out_path.write_text(draft["markdown"], encoding="utf-8")
        print(f"\n[4/4] Runbook markdown saved: {out_path}")
        print(f"      ✓ file size = {out_path.stat().st_size} bytes")

        # 5. 版本化 demo：发布 + 列出版本
        print("\n[bonus] Publishing v1 + listing versions")
        resp = await client.post(
            f"/api/v1/incidents/{incident_id}/runbook/publish",
            json={"change_note": "smoke initial publish", "published_by": "smoke_script"},
        )
        assert resp.status_code == 200, resp.text
        v1 = resp.json()["version"]
        print(f"      ✓ published v{v1['version_number']} (id={v1['version_id']})")

        resp = await client.get(f"/api/v1/incidents/{incident_id}/runbook/versions")
        versions = resp.json()["versions"]
        print(f"      ✓ versions count = {len(versions)}")

        print("\n✅ SMOKE PASSED\n")


if __name__ == "__main__":
    svc = sys.argv[1] if len(sys.argv) > 1 else "order-service"
    mtr = sys.argv[2] if len(sys.argv) > 2 else "error_rate_percent"
    asyncio.run(main(svc, mtr))
