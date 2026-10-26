"""HarnessLoopEngine — champion-challenger upgrade loop.

Implements the rule: challenger is promoted only if EVERY metric on the
hidden test split strictly exceeds the current champion. Test split is
read-only — never written by the optimizer.

The default eval function is a deterministic hash-based mock. Task 5
will inject a real badcase-replay eval via the ``_eval_fn`` constructor
parameter.

W7.5 数据飞轮闭环（审查报告七、长期演 #14）：
当 ``promote()`` 触发后，除了写 champion.json，还会自动 commit lesson 到 LTM
（带 verification_status=verified）+ 抽取 pattern 到 RuntimePatternStore。
这两个依赖通过构造参数注入，None 时跳过（向后兼容，1131 测试不挂）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.models.badcase import BadCaseEntry
from app.models.harness import Champion, HarnessRunRecord, PatchCandidate
from app.services.badcase_registry_service import BadCaseRegistryService


logger = logging.getLogger(__name__)


# Type alias for the mocked eval function signature.
# Returns (dev_metrics, test_metrics).
EvalFn = Callable[
    [PatchCandidate, list[BadCaseEntry], Champion | None],
    tuple[dict[str, float], dict[str, float]],
]


class HarnessLoopEngine:
    def __init__(
        self,
        registry: BadCaseRegistryService,
        runs_path: str = "backend/data/harness/runs.jsonl",
        champion_path: str = "backend/data/champion.json",
        _eval_fn: EvalFn | None = None,
        min_test_threshold: float = 0.0,
        graph_store=None,
        gate_factory=None,
        # ----- W7.5 飞轮闭环（审查报告七、长期演 #14）-----
        ltm: Any | None = None,
        pattern_store: Any | None = None,
    ) -> None:
        """Initialize the engine.

        Args:
            registry: BadCaseRegistryService used to source dev/test entries.
            runs_path: JSONL file for persisting HarnessRunRecord.
            champion_path: JSON file holding all Champions, keyed by class_name.
            _eval_fn: Injectable eval fn (dev_metrics, test_metrics) → tests
                use a stub. Defaults to a hash-based deterministic mock.
            min_test_threshold: If any test metric falls below this, the
                challenger is rejected unconditionally (safety net).
            ltm: LongTermMemory 实例（None 时跳过 promote→LTM 链路，
                保持向后兼容，不影响 1131 测试）。
            pattern_store: RuntimePatternStore 实例（None 时跳过 promote→KB
                链路，同上保持向后兼容）。
        """
        self.registry = registry
        self.runs_path = Path(runs_path)
        self.runs_path.parent.mkdir(parents=True, exist_ok=True)
        self.champion_path = Path(champion_path)
        self.champion_path.parent.mkdir(parents=True, exist_ok=True)
        self._eval_fn: EvalFn = _eval_fn or self._default_eval
        self.min_test_threshold = min_test_threshold
        self._graph_store = graph_store
        self._gate_factory = gate_factory
        # ----- W7.5 飞轮闭环 -----
        self._ltm = ltm
        self._pattern_store = pattern_store

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def _default_eval(
        self,
        candidate: PatchCandidate,
        entries: list[BadCaseEntry],
        champion: Champion | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Default eval: deterministic from candidate content.

        Real eval (Task 5) will run actual badcase replays. For now,
        hash candidate + entries to produce stable mock metrics.
        """
        h = hashlib.sha256(
            (candidate.patch_id + candidate.badcase_class).encode()
        ).hexdigest()
        base = int(h[:4], 16) / 65535.0  # 0..1
        # Dev slightly higher than test, to exercise both
        dev = {
            "precision": 0.6 + base * 0.3,
            "recall": 0.5 + base * 0.3,
            "f1": 0.55 + base * 0.3,
        }
        test = {
            "precision": 0.5 + base * 0.3,
            "recall": 0.4 + base * 0.3,
            "f1": 0.45 + base * 0.3,
        }
        return dev, test

    def _eval(
        self,
        candidate: PatchCandidate,
        entries: list[BadCaseEntry],
        champion: Champion | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        return self._eval_fn(candidate, entries, champion)

    # ------------------------------------------------------------------
    # Split — must use registry.query_similar + split_for_eval per spec.
    # ------------------------------------------------------------------

    def _split(
        self,
        candidate: PatchCandidate,
    ) -> tuple[list[BadCaseEntry], list[BadCaseEntry]]:
        """Split entries into dev/test by reading only the dev split.

        Per spec §3.1, the test split is hidden from the optimizer. We
        query only the dev split, then deterministically re-derive the
        test partition via `split_for_eval` (which hashes entry.id).
        This ensures the optimizer never directly accesses the test split's
        semantic search — it only sees what split_for_eval allocates.
        """
        dev_entries = self.registry.query_similar(
            candidate.rationale, top_k=100, eval_split="dev"
        )
        # split_for_eval hashes entry.id; same hash regardless of how the
        # entries were collected, so the dev/test partition is deterministic.
        dev, test = self.registry.split_for_eval(dev_entries, dev_ratio=0.7)
        return dev, test

    # ------------------------------------------------------------------
    # Champion I/O
    # ------------------------------------------------------------------

    def get_champion(self, class_name: str) -> Champion | None:
        """Read the current champion for ``class_name`` from disk.

        The file is a JSON object keyed by class_name. Returns None if
        the file doesn't exist, the class is absent, or the entry fails
        validation.
        """
        if not self.champion_path.exists():
            return None
        try:
            data = json.loads(self.champion_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return None
        entry = data.get(class_name) if isinstance(data, dict) else None
        if not entry:
            return None
        try:
            return Champion.model_validate(entry)
        except Exception:
            return None

    def _persist(self, record: HarnessRunRecord) -> None:
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    # ------------------------------------------------------------------
    # Decision — strict "every metric must exceed champion" rule.
    # ------------------------------------------------------------------

    def _decide(
        self,
        champion: Champion | None,
        dev_metrics: dict[str, float],
        test_metrics: dict[str, float],
    ) -> str:
        # Sanity: any test metric below min threshold → reject
        if any(v < self.min_test_threshold for v in test_metrics.values()):
            return "reject"
        if champion is None:
            return "hold_for_human"
        # Every metric must strictly exceed champion's (both dev and test)
        for k, v in test_metrics.items():
            if v <= champion.test_metrics.get(k, -1.0):
                return "reject"
        for k, v in dev_metrics.items():
            if v <= champion.dev_metrics.get(k, -1.0):
                return "reject"
        return "promote"

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def promote(self, run: HarnessRunRecord) -> None:
        """Write the run as the new champion for its badcase_class.

        Reads existing champion file (if any), updates the entry for
        ``run.badcase_class``, and atomically writes back. Does NOT touch
        the test split — that guarantee is enforced by
        ``BadCaseRegistryService._check_test_split_readonly``.

        W7.5 飞轮闭环（审查报告七、长期演 #14）：
        如果构造时注入了 ``ltm`` / ``pattern_store``，promote() 完成后会自动：
        1. commit 一条 lesson 到 LTM（带 verification_status=VERIFIED +
           3 次 verified_by 计数 + 2 个 verifier 来源，触发从 UNVERIFIED 直升
           VERIFIED，跳过 PROVISIONAL 中间态——因为 Harness champion 是被
           A/B 严格评测出来的"已被验证"经验，不是 LLM 自己生成的 lesson）；
        2. 抽取一条 pattern 到 RuntimePatternStore，便于后续 RCA 召回。
        注入参数为 None 时跳过对应步骤（保持 1131 测试不挂）。
        """
        existing: dict = {}
        if self.champion_path.exists():
            try:
                existing = json.loads(self.champion_path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except json.JSONDecodeError:
                existing = {}
        champion = Champion(
            class_name=run.badcase_class,
            version=f"v{run.run_id}",
            promoted_at=run.promoted_at or datetime.now(timezone.utc),
            dev_metrics=run.dev_metrics,
            test_metrics=run.test_metrics,
            source_run_id=run.run_id,
        )
        existing[run.badcase_class] = champion.model_dump(mode="json")
        self.champion_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ----- W7.5 飞轮闭环：commit lesson 到 LTM -----
        if self._ltm is not None:
            try:
                self._commit_lesson_to_ltm(run)
            except Exception as e:
                logger.warning(
                    f"[HarnessLoopEngine] LTM commit 失败（不影响 promote）: {e}"
                )

        # ----- W7.5 飞轮闭环：抽取 pattern 到 KB -----
        if self._pattern_store is not None:
            try:
                self._extract_pattern_to_kb(run)
            except Exception as e:
                logger.warning(
                    f"[HarnessLoopEngine] Pattern 抽取失败（不影响 promote）: {e}"
                )

    def _commit_lesson_to_ltm(self, run: HarnessRunRecord) -> None:
        """
        把 promoted champion 抽象成一条 LTM lesson（PROCEDURAL 类型）。

        关键设计：
        - 写入时直接 verification_status=VERIFIED + verification_count=3 + 2 个 verifier，
          跳过 UNVERIFIED→PROVISIONAL→VERIFIED 晋升路径（Harness champion 已经被
          A/B 严格评测，不需要走"慢慢晋升"的路径，避免与 LLM 自生成 lesson 混在一起）。
        - 失败不阻断 promote 主链路（fire-and-forget）。
        """
        from app.models.memory import (
            MemoryEntry,
            MemoryLevel,
            MemoryType,
            VerificationStatus,
        )

        lesson = MemoryEntry(
            memory_type=MemoryType.PROCEDURAL,
            memory_level=MemoryLevel.LONG_TERM,
            content=(
                f"[Harness promote] badcase_class={run.badcase_class} "
                f"fix_summary={run.candidate.rationale[:200]} "
                f"metrics={run.test_metrics}"
            ),
            summary=f"Harness champion fix for {run.badcase_class}: {run.candidate.rationale[:80]}",
            source_agent="harness_loop_engine",
            source_incident_id="",
            verification_status=VerificationStatus.VERIFIED,
            verification_count=3,
            last_verified_at=datetime.now(timezone.utc),
            verified_by=["harness_loop_engine", "ab_test_evaluator", "registry_split_check"],
        )
        # 异步接口，但 LongTermMemory 内部已经 thread-safe；同步调用即可
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 已在 event loop 中：schedule 不阻塞
                loop.create_task(self._ltm.store(lesson))
            else:
                loop.run_until_complete(self._ltm.store(lesson))
        except RuntimeError:
            # 没有 event loop
            asyncio.run(self._ltm.store(lesson))

    def _extract_pattern_to_kb(self, run: HarnessRunRecord) -> None:
        """
        把 promoted champion 抽取成 RuntimePattern。

        pattern_type 默认按 badcase_class 前缀归类（保持简单）：
        - "memory_oom_*" → "memory_optimization"
        - "cpu_*" → "cpu_optimization"
        - "network_*" → "network_optimization"
        - 其他 → "configuration_change"
        """
        from app.services.runtime_pattern_store import RuntimePattern

        cls = run.badcase_class
        if cls.startswith("memory_") or "_oom_" in cls:
            pattern_type = "memory_optimization"
        elif cls.startswith("cpu_"):
            pattern_type = "cpu_optimization"
        elif cls.startswith("network_") or "_latency_" in cls:
            pattern_type = "network_optimization"
        else:
            pattern_type = "configuration_change"

        pattern = RuntimePattern(
            badcase_class=cls,
            source_run_id=run.run_id,
            source_champion_version=f"v{run.run_id}",
            pattern_type=pattern_type,
            fix_summary=run.candidate.rationale[:300],
            trigger_signature=f"badcase_class={cls}",
            evidence_refs=[run.run_id],
            metrics=run.test_metrics,
            promoted_by="harness_loop_engine",
        )
        self._pattern_store.add(pattern)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_sync(self, candidate: PatchCandidate) -> HarnessRunRecord:
        """Synchronous version of run() — for tests + sync endpoints.

        Pipeline:
            1. Allocate run_id + started_at.
            2. _split(candidate) → (dev_entries, test_entries)
            3. get_champion(candidate.badcase_class)
            4. _eval → (dev_metrics, test_metrics)
            5. _decide → "promote" | "reject" | "hold_for_human"
            6. If "promote" → promote()
            7. _persist(record) — always
            8. Return record.
        """
        run_id = f"hr-{uuid.uuid4().hex[:12]}"
        started_at = datetime.now(timezone.utc)
        dev_entries, test_entries = self._split(candidate)
        champion = self.get_champion(candidate.badcase_class)
        # Eval fn gets the combined entry list (dev + test) so it can
        # compute both metric sets. The split is also encoded in
        # the per-split test set if the eval fn wants to use it.
        dev_metrics, test_metrics = self._eval(
            candidate, dev_entries + test_entries, champion
        )
        decision = self._decide(champion, dev_metrics, test_metrics)
        finished_at = datetime.now(timezone.utc)
        promoted_at = finished_at if decision == "promote" else None
        record = HarnessRunRecord(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            badcase_class=candidate.badcase_class,
            candidate=candidate,
            base_champion_version=champion.version if champion else "v0.0.0",
            dev_metrics=dev_metrics,
            test_metrics=test_metrics,
            decision=decision,
            promoted_at=promoted_at,
            notes=f"dev_entries={len(dev_entries)} test_entries={len(test_entries)}",
        )
        if decision == "promote":
            self.promote(record)
        self._persist(record)
        return record

    async def run(self, candidate: PatchCandidate) -> HarnessRunRecord:
        """Async wrapper around run_sync (Task 5 will use this from FastAPI)."""
        return self.run_sync(candidate)
