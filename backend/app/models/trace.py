"""Trace + Context 分层 models (W8 Task 2.3, P0 pillar 3/5).

Implements PPT P4 Trace + Context Layer:
- 5 元组: Thought / Action / Observation / Gate Result / State Snapshot
- 5 层 Context: Case Brief+System / Graph State+Topology / DFS Scope / Evidence Blocks / Working Memory
- 生命周期: GLOBAL_INVARIANT / PER_ITERATION / PER_SCOPE_CHANGE / ACCUMULATING / PER_ITERATION_REWRITE
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceEventType(str, Enum):
    """Identify one of the five runtime trace event types."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    GATE_RESULT = "gate_result"
    STATE_SNAPSHOT = "state_snapshot"


class ContextLayer(str, Enum):
    """Identify one of the five investigation context layers."""

    CASE_BRIEF_SYSTEM = "case_brief_system"
    GRAPH_STATE_TOPOLOGY = "graph_state_topology"
    DFS_SCOPE = "dfs_scope"
    EVIDENCE_BLOCKS = "evidence_blocks"
    WORKING_MEMORY = "working_memory"


Lifetime = Literal[
    "global_invariant",
    "per_iteration",
    "per_scope_change",
    "accumulating",
    "per_iteration_rewrite",
]

LAYER_LIFETIME: dict[ContextLayer, Lifetime] = {
    ContextLayer.CASE_BRIEF_SYSTEM: "global_invariant",
    ContextLayer.GRAPH_STATE_TOPOLOGY: "per_iteration",
    ContextLayer.DFS_SCOPE: "per_scope_change",
    ContextLayer.EVIDENCE_BLOCKS: "accumulating",
    ContextLayer.WORKING_MEMORY: "per_iteration_rewrite",
}


class TraceEvent(BaseModel):
    """Represent one append-only event in an investigation trace."""

    event_id: str = Field(pattern=r"^te-[a-f0-9]{8}$")
    event_type: TraceEventType
    iteration: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContextSlice(BaseModel):
    """Represent one context layer together with its size and lifetime."""

    layer: ContextLayer
    data: Any = None
    size_bytes: int = 0
    lifetime: Lifetime = "per_iteration"
