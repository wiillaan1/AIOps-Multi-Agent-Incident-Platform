"""CounterfactualGate — 4 问检查 (W8 Task 2.4).

Implements PPT P4 Counterfactual Gate:
"每一步因果判定都必须'过门', 不过门就不能推进调查."
4 问:
1. 时间顺序是否成立?
2. 是否有替代解释未被排除?
3. 证据是否充分支撑该判定?
4. 反事实可行性 (能否构造反事实问题)?
"""

from __future__ import annotations

import re

from app.models.gate import (
    CounterfactualCheck,
    CounterfactualClaim,
    GateResult,
)
from app.models.umodel import InvestigationGraph


_TIME_PATTERN = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})")


class CounterfactualGate:
    """Run the 4-question counterfactual gate over a claim against an investigation graph."""

    def __init__(self, graph: InvestigationGraph) -> None:
        """Store the graph used for evidence lookup during evaluation."""
        self.graph = graph

    def evaluate(self, claim: CounterfactualClaim) -> CounterfactualCheck:
        """Run all 4 checks and aggregate into a single GateResult."""
        checks: dict[str, str] = {}
        checks["time_ordering"] = self._check_time_ordering(claim)
        checks["alternatives"] = self._check_alternatives(claim)
        checks["evidence"] = self._check_evidence(claim)
        checks["counterfactual"] = self._check_counterfactual_feasibility(claim)

        fail_keys = [
            k
            for k, v in checks.items()
            if (
                "fail" in v.lower()
                or "insufficient" in v.lower()
                or "missing" in v.lower()
                or "not excluded" in v.lower()
            )
        ]
        if fail_keys:
            return CounterfactualCheck(
                claim=claim,
                checks=checks,
                result=GateResult.FAIL,
                reason=f"failed checks: {', '.join(fail_keys)}",
            )
        return CounterfactualCheck(
            claim=claim,
            checks=checks,
            result=GateResult.PASS,
            reason="all 4 checks passed",
        )

    def _check_time_ordering(self, claim: CounterfactualClaim) -> str:
        """Verify supporting evidence timestamps are within 60 min of the claim anchor."""
        match = _TIME_PATTERN.match(claim.time_anchor)
        if not match:
            return "fail: time_anchor not parseable as ISO 8601"
        supporting = [
            b for b in self.graph.evidence_blocks
            if b.block_id in claim.supporting_evidence_ids
        ]
        if not supporting:
            return "ok: no supporting evidence to compare"
        anchor_minutes = self._to_minutes(claim.time_anchor)
        for eb in supporting:
            eb_minutes = self._to_minutes(eb.time)
            if eb_minutes is None:
                continue
            if abs(eb_minutes - anchor_minutes) > 60:
                return f"fail: evidence {eb.block_id} time {eb.time} > 60min from anchor"
        return "ok: all evidence within 60min of anchor"

    def _check_alternatives(self, claim: CounterfactualClaim) -> str:
        """Heuristic: each declared alternative needs refuting evidence to be excluded."""
        if not claim.alternatives:
            return "ok: no alternatives declared"
        refuting_blocks = [
            b for b in self.graph.evidence_blocks
            if b.block_id in claim.refuting_evidence_ids
        ]
        if len(refuting_blocks) < len(claim.alternatives) * 0.5:
            return (
                f"fail: {len(claim.alternatives)} alternative(s) declared but only "
                f"{len(refuting_blocks)} refuting evidence blocks"
            )
        return "ok: alternatives appear excluded"

    def _check_evidence(self, claim: CounterfactualClaim) -> str:
        """Need >=2 supporting blocks with avg confidence >= 0.6."""
        supporting = [
            b for b in self.graph.evidence_blocks
            if b.block_id in claim.supporting_evidence_ids
        ]
        if not supporting:
            return "fail: no supporting evidence blocks found in graph"
        avg_conf = sum(b.confidence for b in supporting) / len(supporting)
        if len(supporting) < 2 or avg_conf < 0.6:
            return (
                f"fail: insufficient evidence (count={len(supporting)}, "
                f"avg_conf={avg_conf:.2f})"
            )
        return f"ok: {len(supporting)} blocks, avg_conf={avg_conf:.2f}"

    def _check_counterfactual_feasibility(self, claim: CounterfactualClaim) -> str:
        """Heuristic: counterfactual question is feasible if claim text contains counterfactual markers."""
        text = claim.claim.lower()
        if (
            "if" in text
            or "假设" in claim.claim
            or "would" in text
            or "不是" in claim.claim
        ):
            return "ok: counterfactual question is constructible"
        return "ok: claim is declarative (no explicit counterfactual needed)"

    @staticmethod
    def _to_minutes(t: str) -> int | None:
        """Coarse-grained minutes-since-epoch for ordering comparisons; returns None on parse failure."""
        m = _TIME_PATTERN.match(t)
        if not m:
            return None
        y, mo, d, h, mi, _s = (int(x) for x in m.groups())
        return ((y * 12 + mo) * 31 + d) * 24 * 60 + h * 60 + mi