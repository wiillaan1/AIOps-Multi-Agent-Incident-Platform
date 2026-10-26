"""Unit tests for BadCaseEntry + BadCaseRegistryService."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.badcase import BadCaseEntry, EvalSplit
from app.services.badcase_registry_service import BadCaseRegistryService


def _make_entry(
    id_suffix: str = "abcdef123456",
    badcase_class: str = "rca_self_critique_skip",
    eval_split: str = "dev",
    identified_flaw: str = "RCA Agent did not invoke _self_critique_and_recover when confidence was 0.4",
) -> BadCaseEntry:
    return BadCaseEntry(
        id=f"bc-{id_suffix}",
        created_at=datetime.now(timezone.utc),
        badcase_class=badcase_class,
        incident_id="INC-2026-001",
        audit_trail_excerpt=[{"step": "verify_reflection", "outcome": "confidence too low"}],
        identified_flaw=identified_flaw,
        keywords_for_retrieval=["rca", "low_confidence", "self_critique"],
        suggestion_or_lesson="Add hard threshold check before skipping self-critique",
        severity="high",
        eval_split=eval_split,
    )


def test_entry_id_format_validation():
    with pytest.raises(ValueError):
        BadCaseEntry(
            id="invalid",
            created_at=datetime.now(timezone.utc),
            badcase_class="x",
            incident_id="i",
            identified_flaw="too short flaw",  # min 10
            suggestion_or_lesson="too short lesson",  # min 10
        )


def test_keywords_normalized_to_lowercase():
    e = _make_entry(id_suffix="a" * 12)
    assert e.keywords_for_retrieval == ["rca", "low_confidence", "self_critique"]


def test_create_and_get(tmp_path: Path):
    svc = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    e = _make_entry()
    svc.create(e)
    got = svc.get(e.id)
    assert got is not None
    assert got.id == e.id
    assert got.identified_flaw == e.identified_flaw


def test_create_writes_to_correct_split_file(tmp_path: Path):
    svc = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    svc.allow_test_write = True  # allow test split so we can verify file routing
    dev_e = _make_entry(id_suffix="1" * 12, eval_split="dev")
    test_e = _make_entry(id_suffix="2" * 12, eval_split="test")
    svc.create(dev_e)
    svc.create(test_e)
    assert (tmp_path / "badcases" / "badcase_dev.jsonl").exists()
    assert (tmp_path / "badcases" / "badcase_test.jsonl").exists()


def test_query_similar_returns_related_entries(tmp_path: Path):
    svc = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    e1 = _make_entry(
        id_suffix="a" * 12,
        identified_flaw="RCA confidence was 0.4 below threshold causing skip",
    )
    e2 = _make_entry(
        id_suffix="b" * 12,
        identified_flaw="Heal Plan B selector failed due to empty candidates",
    )
    svc.create(e1)
    svc.create(e2)
    results = svc.query_similar("RCA confidence threshold problem", top_k=2)
    assert len(results) >= 1
    # e1 should be more relevant than e2 for "RCA confidence threshold"
    ids = [r.id for r in results]
    assert e1.id in ids


def test_query_similar_class_filter(tmp_path: Path):
    svc = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    svc.create(_make_entry(id_suffix="1" * 12, badcase_class="class_a"))
    svc.create(_make_entry(id_suffix="2" * 12, badcase_class="class_b"))
    results = svc.query_similar("anything", top_k=10, class_filter="class_a")
    assert all(r.badcase_class == "class_a" for r in results)


def test_split_for_eval_deterministic(tmp_path: Path):
    svc = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    entries = [_make_entry(id_suffix=f"{i:012x}") for i in range(20)]
    dev1, test1 = svc.split_for_eval(entries, dev_ratio=0.7)
    dev2, test2 = svc.split_for_eval(entries, dev_ratio=0.7)
    assert len(dev1) == len(dev2) and len(test1) == len(test2)
    assert {e.id for e in dev1} == {e.id for e in dev2}


def test_test_split_write_is_rejected(tmp_path: Path):
    svc = BadCaseRegistryService(
        chroma_path=str(tmp_path / "chroma"),
        jsonl_path=str(tmp_path / "badcases"),
    )
    svc.allow_test_write = False  # default
    test_e = _make_entry(id_suffix="1" * 12, eval_split="test")
    with pytest.raises(PermissionError, match="test split is read-only"):
        svc.create(test_e)