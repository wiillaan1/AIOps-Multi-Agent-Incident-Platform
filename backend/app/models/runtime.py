"""Investigation loop runtime models (W8 Task 2.2, P0 pillar 2/5)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class LoopStatus(str, Enum):
    """State machine status for an investigation loop."""

    RUNNING = "running"
    CONVERGED = "converged"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


class IterationRecord(BaseModel):
    """Per-iteration audit record written into the loop state."""

    iteration: int = Field(ge=1)
    scope_size: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    actions: list[str] = Field(default_factory=list)
    next_action: str | None = None
    gate_result: str | None = None


class InvestigationLoopState(BaseModel):
    """Mutable loop state container for one investigation case."""

    case_id: str
    tenant: str = "default"
    max_iterations: int = Field(default=10, ge=1, le=100)
    current_iteration: int = 0
    status: LoopStatus = LoopStatus.RUNNING
    records: list[IterationRecord] = Field(default_factory=list)