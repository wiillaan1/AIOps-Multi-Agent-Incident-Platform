"""Harness models — W7.5 data flywheel.

Models for the champion-challenger upgrade loop:
- ``PatchCandidate``  — operator's proposed fix for a badcase class
- ``EvalMetrics``     — extensible metric dict (precision / recall / f1 / …)
- ``HarnessRunRecord`` — one challenger run (with dev + test metrics + decision)
- ``Champion``        — currently promoted champion for a badcase_class
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Decision = Literal["pending", "running", "promote", "reject", "hold_for_human"]


class PatchCandidate(BaseModel):
    """A proposed fix for a single badcase class.

    Operators (or the auto-miner) submit a ``PatchCandidate`` to
    ``HarnessLoopEngine.run_sync``; the engine evaluates the patch on the
    hidden test split and either promotes, rejects, or holds for human
    review.
    """

    patch_id: str = Field(min_length=1, max_length=128)
    badcase_class: str = Field(min_length=3, max_length=64)
    diff_or_patch: str = Field(min_length=1)
    rationale: str = Field(min_length=10)
    proposed_by: str = Field(default="unknown")


class HarnessRunRecord(BaseModel):
    """One champion-challenger run.

    Persisted to ``backend/data/harness/runs.jsonl`` (append-only). A run
    has a unique ``run_id`` and ends in a single ``decision``; the engine
    also records dev vs test metrics separately so that test-set leakage
    is auditable.
    """

    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    badcase_class: str
    candidate: PatchCandidate
    base_champion_version: str = "v0.0.0"
    dev_metrics: dict[str, float] = Field(default_factory=dict)
    test_metrics: dict[str, float] = Field(default_factory=dict)
    decision: Decision = "pending"
    promoted_at: datetime | None = None
    notes: str = ""


class Champion(BaseModel):
    """Currently promoted champion for a single badcase class.

    Persisted to ``backend/data/champion.json`` keyed by ``class_name``
    — there is at most one champion per class.
    """

    class_name: str
    version: str
    promoted_at: datetime
    dev_metrics: dict[str, float]
    test_metrics: dict[str, float]
    source_run_id: str


# Backwards-compat alias used by Task 5 e2e fixtures.
EvalMetrics = dict[str, float]
