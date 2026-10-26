"""TraceRecorder — 5 元组 + 5 层 Context 记录器 (W8 Task 2.3)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from app.models.trace import (
    ContextLayer,
    ContextSlice,
    LAYER_LIFETIME,
    TraceEvent,
    TraceEventType,
)
from app.models.umodel import InvestigationGraph


class TraceRecorder:
    """Record investigation events and maintain the five context layers."""

    def __init__(self, base_path: str, case_id: str, tenant: str = "default") -> None:
        """Initialize an empty recorder for one tenant and case."""
        self.base_path = base_path
        self.case_id = case_id
        self.tenant = tenant
        self.events: list[TraceEvent] = []
        self.context: dict[ContextLayer, Any] = {}
        self._current_iteration = 0
        self._next_event_idx = 0

    def _gen_event_id(self) -> str:
        """Return the next monotonic hex event id."""
        self._next_event_idx += 1
        return f"te-{self._next_event_idx:08x}"

    def record(
        self,
        event_type: TraceEventType,
        payload: dict[str, Any],
        iteration: int | None = None,
    ) -> TraceEvent:
        """Append and return a trace event for the current or supplied iteration."""
        event = TraceEvent(
            event_id=self._gen_event_id(),
            event_type=event_type,
            iteration=iteration if iteration is not None else self._current_iteration,
            payload=payload,
        )
        self.events.append(event)
        return event

    def update_context(self, layer: ContextLayer, data: Any) -> None:
        """Update a context layer according to its lifecycle semantics."""
        if layer == ContextLayer.WORKING_MEMORY:
            self.context[layer] = data  # overwrite
        elif layer == ContextLayer.EVIDENCE_BLOCKS:
            existing: list[Any] = list(self.context.get(layer) or [])
            incoming = data if isinstance(data, list) else [data]
            for item in incoming:
                if item not in existing:
                    existing.append(item)
            self.context[layer] = existing
        else:
            self.context[layer] = data

    def context_slice(self, layer: ContextLayer) -> ContextSlice:
        """Return a typed view of a context layer with its UTF-8 JSON size."""
        data = self.context.get(layer)
        size = len(json.dumps(data, default=str).encode("utf-8")) if data is not None else 0
        return ContextSlice(
            layer=layer,
            data=data,
            size_bytes=size,
            lifetime=LAYER_LIFETIME[layer],
        )

    def advance_iteration(self) -> None:
        """Advance the default iteration used by subsequently recorded events."""
        self._current_iteration += 1

    def snapshot_for_iteration(self, iteration: int, graph: InvestigationGraph) -> None:
        """Persist graph state and events up to an iteration as a JSON snapshot."""
        target = self._trace_dir() / f"iter-{iteration:03d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "iteration": iteration,
            "graph": graph.model_dump(mode="json"),
            "events": [e.model_dump(mode="json") for e in self.events if e.iteration <= iteration],
        }
        target.write_text(json.dumps(snapshot, default=str, indent=2), encoding="utf-8")

    def total_size_bytes(self) -> int:
        """Return the sum of serialized sizes for all five context layers."""
        return sum(self.context_slice(layer).size_bytes for layer in ContextLayer)

    def persist(self) -> None:
        """Atomically persist all recorded events as newline-delimited JSON."""
        target = self._trace_dir() / "trace.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".trace.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for ev in self.events:
                    f.write(ev.model_dump_json() + "\n")
            os.replace(tmp_name, target)
        except Exception:
            if Path(tmp_name).exists():
                Path(tmp_name).unlink()
            raise

    def _trace_dir(self) -> Path:
        """Return the per-tenant, per-case trace directory."""
        return Path(self.base_path) / self.tenant / self.case_id
