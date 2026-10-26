"""Rule and checklist validation models for investigation planning."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.umodel import EntityType


class ChecklistItem(BaseModel):
    """Describe one required investigation dimension."""

    entity_type: EntityType
    dimension: str = Field(min_length=1, max_length=64)
    required: bool = True
    rationale: str = ""


class Rule(BaseModel):
    """Describe an evidence-triggered observation plan."""

    rule_id: str = Field(min_length=1, max_length=64)
    trigger_entity_type: EntityType
    condition_evidence_pattern: str = Field(min_length=1, max_length=128)
    produces_observation_plan: list[str] = Field(default_factory=list, max_length=20)
    priority: int = Field(default=10, ge=0, le=100)


class RuleMatch(BaseModel):
    """Represent a rule and the evidence that activated it."""

    rule: Rule
    matched_evidence_ids: list[str] = Field(default_factory=list)
    produced_observations: list[str] = Field(default_factory=list)
