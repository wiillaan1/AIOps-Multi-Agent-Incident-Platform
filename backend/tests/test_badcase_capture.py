"""Test W7 audit_trail auto-capture hook."""

import asyncio
import json
from pathlib import Path

from app.models.events import AlertEvent, SeverityLevel
from app.models.incident import Incident
from app.services.badcase_capture import detect_triggers, maybe_capture_badcase
from app.services.badcase_registry_service import BadCaseRegistryService


def _run(coro):
    return asyncio.run(coro)


def _make_incident(audit_trail: list[dict], verification: dict | None = None) -> Incident:
    alert = AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.HIGH,
        labels={},
        annotations={},
    )
    inc = Incident.from_alert(alert)
    inc.context["audit_trail"] = audit_trail
    if verification is not None:
        inc.context["verification"] = verification
    return inc


def test_detect_reflect_escalate_trigger():
    trail = [{"step": "verify_reflection", "actions_taken": ["next_action=escalate"]}]
    triggers = detect_triggers(_make_incident(trail))
    assert "REFLECT_ESCALATE" in [t.value for t in triggers]


def test_detect_rca_mismatch_trigger():
    trail = [{"step": "verify_reflection"}]
    triggers = detect_triggers(
        _make_incident(trail, verification={"rca_match_status": "mismatch"})
    )
    assert "RCA_MISMATCH" in [t.value for t in triggers]


def test_detect_plan_b_triggered():
    trail = [{"step": "verify_reflection"}]
    triggers = detect_triggers(
        _make_incident(trail, verification={"plan_b_triggered": True})
    )
    assert "PLAN_B_TRIGGERED" in [t.value for t in triggers]


def test_no_triggers_on_clean_run():
    triggers = detect_triggers(
        _make_incident(
            [{"step": "verify_reflection", "actions_taken": ["next_action=complete"]}],
            verification={"rca_match_status": "match", "plan_b_triggered": False},
        )
    )
    assert triggers == []


def test_capture_writes_to_candidates_file(tmp_path: Path):
    candidates = tmp_path / "candidates.jsonl"
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    inc = _make_incident(
        [{"step": "verify_reflection", "actions_taken": ["next_action=escalate"]}],
        verification={"rca_match_status": "mismatch"},
    )
    captured_id = _run(maybe_capture_badcase(inc, registry, candidates_path=str(candidates)))
    assert captured_id is not None
    assert candidates.exists()
    line = candidates.read_text(encoding="utf-8").strip()
    draft = json.loads(line)
    assert draft["id"] == captured_id
    assert draft["badcase_class"] != ""
    keywords_upper = [k.upper() for k in draft["keywords_for_retrieval"]]
    assert "ESCALATE" in keywords_upper[0] or "RCA_MISMATCH" in keywords_upper


def test_capture_idempotent_on_same_incident(tmp_path: Path):
    candidates = tmp_path / "candidates.jsonl"
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    inc = _make_incident(
        [{"step": "verify_reflection", "actions_taken": ["next_action=escalate"]}]
    )
    id1 = _run(maybe_capture_badcase(inc, registry, candidates_path=str(candidates)))
    id2 = _run(maybe_capture_badcase(inc, registry, candidates_path=str(candidates)))
    assert id1 == id2
    # file should have only one line (idempotent)
    lines = [l for l in candidates.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1


def test_capture_writes_audit_trail_entry(tmp_path: Path):
    """After maybe_capture_badcase, incident.context['audit_trail'] must contain
    a final entry with step == 'auto_badcase_captured' (spec §5.1 action list).
    """
    candidates = tmp_path / "candidates.jsonl"
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    inc = _make_incident(
        [{"step": "verify_reflection", "actions_taken": ["next_action=escalate"]}],
        verification={"rca_match_status": "mismatch"},
    )
    # Pre-condition: trail length known before capture
    pre_len = len(inc.context.get("audit_trail", []))
    captured_id = _run(maybe_capture_badcase(inc, registry, candidates_path=str(candidates)))
    assert captured_id is not None

    trail = inc.context.get("audit_trail", [])
    assert len(trail) == pre_len + 1, "exactly one audit_trail entry should be appended"
    last = trail[-1]
    assert last["step"] == "auto_badcase_captured"
    # trigger payload surfaces trigger names for downstream observers
    assert "triggers=" in last["trigger"]
    assert "REFLECT_ESCALATE" in last["trigger"]
    # actions_taken carries class + entry_id for traceability
    actions = " ".join(last["actions_taken"])
    assert "badcase_class=" in actions
    assert f"entry_id={captured_id}" in actions
    # outcome should mention the candidates path
    assert candidates.name in last["outcome"]