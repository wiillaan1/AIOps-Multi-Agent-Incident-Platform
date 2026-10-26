"""Test HarnessLoopEngine champion-challenger logic (Task 4 of W7.5 data flywheel)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.badcase import BadCaseEntry
from app.models.harness import Champion, HarnessRunRecord, PatchCandidate
from app.services.badcase_registry_service import BadCaseRegistryService
from app.services.harness_engine import HarnessLoopEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    id_suffix: str,
    badcase_class: str = "rca_predict_mismatch",
    split: str = "dev",
) -> BadCaseEntry:
    return BadCaseEntry(
        id=f"bc-{id_suffix}",
        created_at=datetime.now(timezone.utc),
        badcase_class=badcase_class,
        incident_id=f"INC-{id_suffix}",
        audit_trail_excerpt=[{"step": "verify_reflection"}],
        identified_flaw="Some identified flaw description with detail",
        keywords_for_retrieval=["rca", "low_confidence"],
        suggestion_or_lesson="Some suggestion description with detail",
        severity="medium",
        eval_split=split,
    )


def _make_candidate(badcase_class: str = "rca_predict_mismatch") -> PatchCandidate:
    return PatchCandidate(
        patch_id="patch-001",
        badcase_class=badcase_class,
        diff_or_patch="# some diff",
        rationale="Fix threshold check for confidence",
        proposed_by="operator_1",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_returns_record_with_dev_and_test_metrics(tmp_path: Path):
    """Engine.run_sync returns a HarnessRunRecord with dev + test metrics."""
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    for i in range(5):
        registry.create(_make_entry(id_suffix=f"{i:012x}"))
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
    )
    record = engine.run_sync(_make_candidate())
    assert isinstance(record, HarnessRunRecord)
    assert "precision" in record.dev_metrics
    assert "precision" in record.test_metrics
    assert "f1" in record.dev_metrics
    assert "f1" in record.test_metrics


def test_reject_when_any_test_metric_below_threshold(tmp_path: Path):
    """If a test metric falls below the configured threshold (here, -0.1), reject even if dev is good."""
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    for i in range(5):
        registry.create(_make_entry(id_suffix=f"{i:012x}"))
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        # Mock eval to return dev high, test containing a negative metric
        # (below the 0.0 default min_test_threshold) to trigger rejection
        # data-driven — not by a high default floor.
        _eval_fn=lambda candidate, entries, champion: (
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},  # dev
            {"precision": -0.1, "recall": 0.9, "f1": 0.9},  # test (precision < 0.0)
        ),
    )
    record = engine.run_sync(_make_candidate())
    assert record.decision == "reject"


def test_promote_when_all_metrics_exceed_champion(tmp_path: Path):
    """Pre-existing champion (low metrics) + challenger (high metrics) → promote."""
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    for i in range(5):
        registry.create(_make_entry(id_suffix=f"{i:012x}"))

    # Pre-seed champion with low scores
    champion_path = tmp_path / "champion.json"
    seed_champion = Champion(
        class_name="rca_predict_mismatch",
        version="v0.0.0",
        promoted_at=datetime.now(timezone.utc),
        dev_metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
        test_metrics={"precision": 0.5, "recall": 0.5, "f1": 0.5},
        source_run_id="baseline",
    )
    champion_path.write_text(
        json.dumps({"rca_predict_mismatch": seed_champion.model_dump(mode="json")})
    )

    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(champion_path),
        _eval_fn=lambda c, e, ch: (
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
        ),
    )
    record = engine.run_sync(_make_candidate())
    assert record.decision == "promote"

    new_champion_data = json.loads(champion_path.read_text())
    new_champion = Champion.model_validate(
        new_champion_data["rca_predict_mismatch"]
    )
    assert new_champion.source_run_id == record.run_id
    assert new_champion.dev_metrics["precision"] == 0.9


def test_hold_for_human_when_no_champion_exists(tmp_path: Path):
    """If no champion file exists yet, decision = hold_for_human."""
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    for i in range(5):
        registry.create(_make_entry(id_suffix=f"{i:012x}"))
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),  # does not exist
        _eval_fn=lambda c, e, ch: (
            {"precision": 0.7, "recall": 0.7, "f1": 0.7},
            {"precision": 0.7, "recall": 0.7, "f1": 0.7},
        ),
    )
    record = engine.run_sync(_make_candidate())
    assert record.decision == "hold_for_human"


def test_test_split_write_rejected_at_promotion(tmp_path: Path):
    """Harness optimizer must never write test entries — registry enforces it."""
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    test_entry = _make_entry(id_suffix="1" * 12, split="test")

    # Direct test split write must fail
    with pytest.raises(PermissionError):
        registry.create(test_entry)

    # Verify it was never written
    assert not (tmp_path / "badcases" / "badcase_test.jsonl").exists()

    # And the harness engine must not flip the flag anywhere
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        _eval_fn=lambda c, e, ch: (
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
        ),
    )
    record = engine.run_sync(_make_candidate())
    # Just running the engine does not enable test writes
    assert registry.allow_test_write is False
    # No test split file was created by the harness either
    assert not (tmp_path / "badcases" / "badcase_test.jsonl").exists()
    # And since no champion exists, the run is held for human
    assert record.decision == "hold_for_human"


def test_split_only_queries_dev_collection(tmp_path: Path):
    """`_split` must query ONLY the dev split — optimizer never reads the test split directly (spec §3.1)."""
    from app.models.badcase import BadCaseEntry
    from app.services.harness_engine import HarnessLoopEngine as _Engine

    # Build a fake entry to return from the mocked query_similar
    fake_entry = _make_entry(id_suffix="a" * 12, split="dev")

    # Build a registry-like stub that records every call to query_similar
    class _StubRegistry:
        def __init__(self):
            self.calls: list[dict] = []

        def query_similar(self, rationale, top_k, eval_split):
            self.calls.append(
                {"rationale": rationale, "top_k": top_k, "eval_split": eval_split}
            )
            return [fake_entry]

        def split_for_eval(self, entries, dev_ratio=0.7):
            # Deterministic: bucket the single entry by hashing its id.
            # With one entry, both dev and test buckets are derived
            # by split_for_eval's own logic — we don't need a fake impl.
            return entries[: max(1, int(len(entries) * dev_ratio))], entries[
                max(1, int(len(entries) * dev_ratio)) :
            ]

    stub = _StubRegistry()
    engine = _Engine(
        registry=stub,
        runs_path=str(tmp_path / "runs.jsonl"),
        champion_path=str(tmp_path / "champion.json"),
        _eval_fn=lambda c, e, ch: (
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
            {"precision": 0.9, "recall": 0.9, "f1": 0.9},
        ),
    )
    engine.run_sync(_make_candidate())

    # Exactly one call to query_similar, and it's for the dev split.
    assert len(stub.calls) == 1, f"expected 1 query_similar call, got {len(stub.calls)}"
    assert stub.calls[0]["eval_split"] == "dev"
    assert not any(c["eval_split"] == "test" for c in stub.calls)


def test_run_appends_to_runs_jsonl(tmp_path: Path):
    """Each run_sync appends one HarnessRunRecord to runs.jsonl."""
    registry = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    for i in range(3):
        registry.create(_make_entry(id_suffix=f"{i:012x}"))
    runs_path = tmp_path / "runs.jsonl"
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path=str(runs_path),
        champion_path=str(tmp_path / "champion.json"),
    )
    engine.run_sync(_make_candidate())
    assert runs_path.exists()
    lines = [line for line in runs_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    parsed = HarnessRunRecord.model_validate_json(lines[0])
    assert parsed.candidate.patch_id == "patch-001"
