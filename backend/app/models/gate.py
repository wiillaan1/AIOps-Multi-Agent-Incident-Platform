"""Gate models — Counterfactual Gate (W8 Task 2.4, P0 pillar 4/5)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GateResult(str, Enum):
    """Outcome of a gate evaluation."""

    PASS = "pass"
    FAIL = "fail"
    NEEDS_HUMAN = "needs_human"


class CounterfactualClaim(BaseModel):
    """A causal claim to be checked by the counterfactual gate."""

    claim: str = Field(min_length=1, max_length=512)
    mechanism: str = Field(min_length=1, max_length=128)
    time_anchor: str = Field(min_length=1, max_length=64)
    alternatives: list[str] = Field(default_factory=list, max_length=20)
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    refuting_evidence_ids: list[str] = Field(default_factory=list, max_length=50)


class CounterfactualCheck(BaseModel):
    """The result of running the 4 counterfactual checks on a claim."""

    claim: CounterfactualClaim
    checks: dict[str, str] = Field(default_factory=dict)
    result: GateResult
    reason: str = ""