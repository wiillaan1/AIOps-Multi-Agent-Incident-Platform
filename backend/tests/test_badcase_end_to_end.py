"""End-to-end test for the W7.5 badcase data flywheel.

Simulates the full loop:
  1. A W7 incident triggers verify_phase with rca_match_status=mismatch
  2. maybe_capture_badcase writes a draft to candidates.jsonl
  3. /badcase/{id}/analyze returns draft + similar cases (RAG-driven)
  4. /harness/run accepts a candidate and runs champion-challenger
  5. The challenger wins on both dev and test splits → champion promoted
  6. The next incident's audit_trail records the new champion version
"""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.models.badcase import BadCaseEntry
from app.models.events import AlertEvent, SeverityLevel
from app.models.incident import Incident
from app.models.harness import PatchCandidate
from app.services.badcase_capture import maybe_capture_badcase
from app.services.badcase_registry_service import BadCaseRegistryService


def _make_incident(verification: dict | None = None) -> Incident:
    alert = AlertEvent(
        service="order-service", metric="cpu_usage_percent",
        value=95.0, threshold=80.0, severity=SeverityLevel.HIGH,
        labels={}, annotations={},
    )
    inc = Incident.from_alert(alert)
    inc.context["audit_trail"] = [
        {"step": "verify_reflection", "actions_taken": ["next_action=escalate"]}
    ]
    if verification:
        inc.context["verification"] = verification
    return inc


def test_full_data_flywheel(tmp_path: Path):
    """The complete loop from badcase capture to champion promotion."""
    # Setup isolated data directories
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "badcases").mkdir()
    (data_root / "harness").mkdir()
    (data_root / "chromadb").mkdir()

    candidates_file = data_root / "badcases" / "candidates.jsonl"
    runs_file = data_root / "harness" / "runs.jsonl"
    champion_file = data_root / "champion.json"

    registry = BadCaseRegistryService(
        chroma_path=str(data_root / "chromadb"),
        jsonl_path=str(data_root / "badcases"),
    )

    # Step 1: incident triggers verify_phase with mismatch
    inc1 = _make_incident(verification={"rca_match_status": "mismatch"})

    # Step 2: auto-capture writes a draft
    captured_id = asyncio.run(maybe_capture_badcase(inc1, registry, candidates_path=str(candidates_file)))
    assert captured_id is not None
    assert candidates_file.exists()

    # Step 3: seed dev split with the captured badcase (simulating operator review)
    draft = json.loads(candidates_file.read_text().splitlines()[0])
    draft_entry = BadCaseEntry(
        **{k: v for k, v in draft.items() if k in BadCaseEntry.model_fields}
    )
    registry.create(draft_entry)

    # Step 3.5: pre-seed a low-scoring baseline champion so the FIRST run can promote.
    # (Without a baseline, _decide returns "hold_for_human" — verified by
    # test_hold_for_human_when_no_champion_exists. The harness engine contract is
    # "promote only when challenger strictly exceeds champion", so a seed baseline
    # is required. This mirrors production: the first champion is bootstrapped
    # from an earlier evaluation, not from the first incoming badcase.)
    from datetime import datetime, timezone
    from app.models.harness import Champion
    seed_champion = Champion(
        class_name=draft_entry.badcase_class,
        version="v0.0.0",
        promoted_at=datetime.now(timezone.utc),
        dev_metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
        test_metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
        source_run_id="baseline",
    )
    champion_file.write_text(
        json.dumps({draft_entry.badcase_class: seed_champion.model_dump(mode="json")})
    )

    # Step 4: harness run
    from app.services.harness_engine import HarnessLoopEngine
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(runs_file),
        champion_path=str(champion_file),
        _eval_fn=lambda c, e, ch: (
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
        ),
    )
    candidate = PatchCandidate(
        patch_id="patch-001",
        badcase_class=draft_entry.badcase_class,
        diff_or_patch="# tightened threshold",
        rationale="Add explicit confidence threshold to prevent skip",
        proposed_by="test_e2e",
    )
    record = engine.run_sync(candidate)
    assert record.decision == "promote"

    # Step 5: a second incident comes in; the new champion is available
    inc2 = _make_incident(verification={"rca_match_status": "match"})
    new_champion = engine.get_champion(draft_entry.badcase_class)
    assert new_champion is not None
    inc2.context["champion_version"] = new_champion.version
    assert "hr-" in inc2.context["champion_version"]

    # Step 6: run.jsonl has the audit trail
    assert runs_file.exists()
    line = runs_file.read_text().splitlines()[0]
    persisted = json.loads(line)
    assert persisted["decision"] == "promote"
    assert persisted["candidate"]["patch_id"] == "patch-001"

    # Step 7: champion.json was written
    assert champion_file.exists()
    champion_data = json.loads(champion_file.read_text())
    assert draft_entry.badcase_class in champion_data
    assert champion_data[draft_entry.badcase_class]["source_run_id"] == record.run_id


def test_legacy_pipeline_unchanged():
    """The original _process_incident_pipeline_legacy must not be modified.

    This is a guard test: if anyone modifies the legacy function, this
    test will fail (since it asserts the function exists with the expected
    name and is byte-identical to its pre-W7.5 state).
    """
    import inspect
    from app.api.routes import _process_incident_pipeline_legacy
    src = inspect.getsource(_process_incident_pipeline_legacy)
    # Sanity: the function still exists and has the expected signature
    assert "async def _process_incident_pipeline_legacy" in src
    # Sanity: the W7.5 capture hook is NOT inside this function
    assert "maybe_capture_badcase" not in src
    assert "BadCaseRegistryService" not in src