"""Tests for TraceRecorder + Context 分层 (Task 2.3, P0 pillar 3/5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models.runtime import InvestigationLoopState
from app.models.trace import (
    ContextLayer,
    ContextSlice,
    TraceEvent,
    TraceEventType,
)
from app.models.umodel import InvestigationGraph
from app.services.trace_recorder import TraceRecorder


@pytest.fixture
def recorder(tmp_path: Path) -> TraceRecorder:
    return TraceRecorder(base_path=str(tmp_path / "trace"), case_id="INC-1", tenant="t1")


def test_record_thought_appends_event(recorder):
    recorder.record(TraceEventType.THOUGHT, {"text": "可能 db 慢"})
    assert len(recorder.events) == 1
    assert recorder.events[0].event_type == TraceEventType.THOUGHT


def test_record_action_includes_iteration(recorder):
    recorder.record(TraceEventType.ACTION, {"name": "query_prometheus"}, iteration=3)
    assert recorder.events[0].iteration == 3


def test_context_slice_case_brief_is_global(recorder):
    recorder.update_context(ContextLayer.CASE_BRIEF_SYSTEM, {"title": "checkout 503"})
    slice_ = recorder.context_slice(ContextLayer.CASE_BRIEF_SYSTEM)
    assert slice_.lifetime == "global_invariant"


def test_context_slice_working_memory_overwritten_each_iteration(recorder):
    recorder.update_context(ContextLayer.WORKING_MEMORY, {"scratch": "x"})
    recorder.advance_iteration()
    recorder.update_context(ContextLayer.WORKING_MEMORY, {"scratch": "y"})
    assert recorder.context_slice(ContextLayer.WORKING_MEMORY).data == {"scratch": "y"}


def test_context_slice_evidence_blocks_accumulating(recorder):
    recorder.update_context(ContextLayer.EVIDENCE_BLOCKS, [{"id": 1}])
    recorder.update_context(ContextLayer.EVIDENCE_BLOCKS, [{"id": 1}, {"id": 2}])
    assert len(recorder.context_slice(ContextLayer.EVIDENCE_BLOCKS).data) == 2


def test_snapshot_for_iteration_persists_graph_and_events(recorder, tmp_path):
    recorder.record(TraceEventType.THOUGHT, {"text": "x"})
    g = InvestigationGraph()
    g.iteration = 1
    recorder.snapshot_for_iteration(1, g)
    # check the file exists
    files = list((tmp_path / "trace" / "t1" / "INC-1").glob("iter-*.json"))
    assert len(files) == 1


def test_total_size_bytes_sums_all_context_layers(recorder):
    recorder.update_context(ContextLayer.CASE_BRIEF_SYSTEM, {"a": 1})
    recorder.update_context(ContextLayer.WORKING_MEMORY, {"b": 2})
    total = recorder.total_size_bytes()
    assert total > 0


def test_persist_to_jsonl_creates_file(recorder, tmp_path):
    recorder.record(TraceEventType.ACTION, {"name": "x"})
    recorder.persist()
    p = tmp_path / "trace" / "t1" / "INC-1" / "trace.jsonl"
    assert p.exists()
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "action"
