"""W7 audit_trail auto-capture hook — W7.5 data flywheel.

Reads incident.context["audit_trail"] + ["verification"] to detect signals
that indicate a badcase. Writes draft BadCaseEntry to candidates.jsonl
for human review (does NOT directly add to registry dev/test split).

Helper names are **public** (no leading underscore) so Task 3's RAG
endpoint and tests can import them directly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from app.models.badcase import BadCaseEntry
from app.models.incident import Incident
from app.services.badcase_registry_service import BadCaseRegistryService


class CaptureTrigger(str, Enum):
    REFLECT_ESCALATE = "REFLECT_ESCALATE"
    RCA_MISMATCH = "RCA_MISMATCH"
    PLAN_B_TRIGGERED = "PLAN_B_TRIGGERED"
    MONITOR_FEEDBACK_THRESHOLD = "MONITOR_FEEDBACK_THRESHOLD"
    INTENT_UNANSWERED = "INTENT_UNANSWERED"


def detect_triggers(incident: Incident) -> list[CaptureTrigger]:
    """Inspect incident.context to detect capture conditions.

    Returns the list of trigger types that fired. Empty list means no
    capture is needed.
    """
    triggers: list[CaptureTrigger] = []
    trail = incident.context.get("audit_trail", [])
    verification = incident.context.get("verification") or {}

    # 1. reflect_on_verify returned escalate
    for entry in trail:
        actions = entry.get("actions_taken", [])
        if any("next_action=escalate" in str(a) for a in actions):
            triggers.append(CaptureTrigger.REFLECT_ESCALATE)
            break

    # 2. RCA mismatch
    if verification.get("rca_match_status") == "mismatch":
        triggers.append(CaptureTrigger.RCA_MISMATCH)

    # 3. Plan B triggered
    if verification.get("plan_b_triggered") is True:
        triggers.append(CaptureTrigger.PLAN_B_TRIGGERED)

    return triggers


def classify_badcase_class(triggers: list[CaptureTrigger]) -> str:
    """Map triggers to a stable badcase class label."""
    if CaptureTrigger.RCA_MISMATCH in triggers:
        return "rca_predict_mismatch"
    if CaptureTrigger.PLAN_B_TRIGGERED in triggers:
        return "heal_plan_b_triggered"
    if CaptureTrigger.REFLECT_ESCALATE in triggers:
        return "reflect_escalated"
    return "unknown"


def extract_keywords(triggers: list[CaptureTrigger]) -> list[str]:
    """Build a deduped ordered list of retrieval keywords."""
    base = ["audit_trail", "verification", "w7_reflection"]
    base.extend(t.value.lower() for t in triggers)
    return list(dict.fromkeys(base))


def build_identified_flaw(incident: Incident, triggers: list[CaptureTrigger]) -> str:
    """Compose a one-paragraph human-readable flaw description."""
    trigger_str = ", ".join(t.value for t in triggers)
    return (
        f"Auto-captured from incident {incident.incident_id} due to: {trigger_str}. "
        f"Verify whether verify_phase reflection is sufficient."
    )


def build_suggestion(triggers: list[CaptureTrigger]) -> str:
    """Compose a default review action / lesson text."""
    return (
        "Inspect verify_phase: confirm whether mismatch/escalate signals warrant "
        "explicit guardrail, and check if prior similar cases were fixed. "
        f"Triggered by: {', '.join(t.value for t in triggers)}"
    )


async def maybe_capture_badcase(
    incident: Incident,
    registry: BadCaseRegistryService,
    candidates_path: str = "backend/data/badcases/candidates.jsonl",
) -> str | None:
    """If incident triggers any capture condition, write a draft BadCaseEntry.

    Returns the entry_id if captured, None otherwise. Idempotent on
    (incident_id, badcase_class).
    """
    triggers = detect_triggers(incident)
    if not triggers:
        return None

    badcase_class = classify_badcase_class(triggers)
    entry_id_seed = f"bc-{incident.incident_id}-{badcase_class}"
    # Stable id: first 12 hex of sha256 — matches BadCaseEntry.id pattern
    entry_id = "bc-" + hashlib.sha256(entry_id_seed.encode()).hexdigest()[:12]

    # Idempotency: check if candidate with same id already in file
    path = Path(candidates_path)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("id") == entry_id:
                return entry_id

    trail = incident.context.get("audit_trail", [])
    excerpt = trail[-5:] if trail else []

    entry = BadCaseEntry(
        id=entry_id,
        created_at=datetime.now(timezone.utc),
        badcase_class=badcase_class,
        incident_id=incident.incident_id,
        audit_trail_excerpt=excerpt,
        identified_flaw=build_identified_flaw(incident, triggers),
        keywords_for_retrieval=extract_keywords(triggers),
        suggestion_or_lesson=build_suggestion(triggers),
        severity="high" if CaptureTrigger.REFLECT_ESCALATE in triggers else "medium",
        eval_split="dev",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")

    # Spec §5.1: 写 audit_trail 一条 `auto_badcase_captured`。
    # Local import to avoid circular deps (reflection.py -> agent imports).
    from app.agents.reflection import append_audit_trail

    append_audit_trail(
        incident.context,
        step="auto_badcase_captured",
        trigger=f"triggers={[t.value for t in triggers]}",
        actions_taken=[
            f"badcase_class={badcase_class}",
            f"entry_id={entry_id}",
        ],
        outcome=f"draft written to {candidates_path}",
        duration_ms=0,
    )

    return entry_id