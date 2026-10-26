"""BadCase Registry Service — W7.5 data flywheel.

CRUD over JSONL + ChromaDB-indexed search.
Dev split: read+write (optimizer accesses).
Test split: read-only (hidden from optimizer).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from app.models.badcase import BadCaseEntry


class BadCaseRegistryService:
    def __init__(
        self,
        chroma_path: str = "backend/data/chromadb",
        jsonl_path: str = "backend/data/badcases",
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.allow_test_write = False  # CRITICAL: harness optimizer must not flip this
        self._chroma = self._init_chroma()

    def _init_chroma(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        return {
            "dev": client.get_or_create_collection("badcase_dev"),
            "test": client.get_or_create_collection("badcase_test"),
        }

    def _jsonl_path(self, eval_split: str) -> Path:
        return self.jsonl_path / f"badcase_{eval_split}.jsonl"

    def _check_test_split_readonly(self, entry: BadCaseEntry) -> None:
        split = entry.eval_split.value if hasattr(entry.eval_split, "value") else str(entry.eval_split)
        if split == "test" and not self.allow_test_write:
            raise PermissionError(
                f"Cannot write to test split: {entry.id} — test split is read-only"
            )

    def create(self, entry: BadCaseEntry) -> None:
        self._check_test_split_readonly(entry)
        split = entry.eval_split.value if hasattr(entry.eval_split, "value") else str(entry.eval_split)
        path = self._jsonl_path(split)
        with path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")
        self.add_to_chroma(entry)

    def get(self, entry_id: str) -> BadCaseEntry | None:
        for split in ("dev", "test"):
            path = self._jsonl_path(split)
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("id") == entry_id:
                    return BadCaseEntry(**data)
        return None

    def add_to_chroma(self, entry: BadCaseEntry) -> None:
        split = entry.eval_split.value if hasattr(entry.eval_split, "value") else str(entry.eval_split)
        collection = self._chroma[split]
        text = " ".join([
            entry.identified_flaw,
            entry.suggestion_or_lesson,
            " ".join(entry.keywords_for_retrieval),
        ])
        collection.upsert(
            ids=[entry.id],
            documents=[text],
            metadatas=[{"badcase_class": entry.badcase_class, "severity": entry.severity}],
        )

    def query_similar(
        self,
        text: str,
        top_k: int = 5,
        class_filter: str | None = None,
        eval_split: str = "dev",
    ) -> list[BadCaseEntry]:
        collection = self._chroma[eval_split]
        kwargs: dict = {"query_texts": [text], "n_results": min(top_k, 100)}
        if class_filter is not None:
            kwargs["where"] = {"badcase_class": class_filter}
        results = collection.query(**kwargs)
        ids = results.get("ids", [[]])[0]
        return [e for e in (self.get(i) for i in ids) if e is not None]

    def split_for_eval(
        self,
        entries: list[BadCaseEntry],
        dev_ratio: float = 0.7,
    ) -> tuple[list[BadCaseEntry], list[BadCaseEntry]]:
        """Deterministic split: hash(entry.id) % 100 < dev_ratio*100 → dev."""
        dev, test = [], []
        for e in entries:
            h = int(hashlib.sha256(e.id.encode()).hexdigest(), 16) % 100
            (dev if h < dev_ratio * 100 else test).append(e)
        return dev, test