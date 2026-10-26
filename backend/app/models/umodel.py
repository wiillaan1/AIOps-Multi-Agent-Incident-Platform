"""UModel — unified entity model + Investigation Graph (W8).

P1 计划的基类。所有 Runtime 支柱 (P0) / 协议外壳 (P2) 都基于此。
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class EntityType(str, Enum):
    SERVICE = "service"
    POD = "pod"
    TOPIC = "topic"
    DB = "db"
    NODE = "node"
    RUNBOOK = "runbook"
    CI_CD = "ci_cd"
    CHANGE_EVENT = "change_event"


class LinkType(str, Enum):
    CALLS = "calls"
    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"
    RUNS_ON = "runs_on"
    SAME_AS = "same_as"
    DEPENDS_ON = "depends_on"
    SCHEDULED_ON = "scheduled_on"
    HOSTED_ON = "hosted_on"
    DEPLOYED = "deployed"
    READS = "reads"
    WRITES = "writes"


class EntityState(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


def entity_id(entity_type: EntityType | str, raw_id: str) -> str:
    """Combine entity_type and raw id to a single graph-unique key.

    >>> entity_id(EntityType.SERVICE, "checkout-api")
    'service:checkout-api'
    """
    prefix = entity_type.value if isinstance(entity_type, EntityType) else entity_type
    return f"{prefix}:{raw_id}"


class EntitySet(BaseModel):
    """A single entity (service/pod/db/...) tracked in the investigation graph."""

    entity_type: EntityType
    entity_id: str = Field(min_length=1, max_length=128)
    attributes: dict[str, Any] = Field(default_factory=dict)
    state: EntityState = EntityState.UNKNOWN
    labels: list[str] = Field(default_factory=list, max_length=20)

    @property
    def id(self) -> str:
        """Return the graph-unique key ``type:entity_id``."""
        return entity_id(self.entity_type, self.entity_id)


class EntitySetLink(BaseModel):
    """A directed edge between two entity keys."""

    src: str = Field(min_length=1)
    dst: str = Field(min_length=1)
    link_type: LinkType
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _src_dst_distinct(self) -> EntitySetLink:
        """Reject self-loops where src equals dst."""
        if self.src == self.dst:
            raise ValueError("EntitySetLink src and dst must be distinct")
        return self


class EvidenceBlock(BaseModel):
    """One piece of evidence supporting or refuting a hypothesis.

    Mirrors PPT 4-Counterfactual EvidenceBlock (Object/Time/Observation/Mechanism/Confidence)
    with an extra ``is_evidence_against_hypothesis`` list for counter-evidence.
    """

    block_id: str = Field(pattern=r"^eb-[a-zA-Z0-9_-]{1,32}$")
    object_ref: str = Field(min_length=1, max_length=128)
    time: str = Field(min_length=1, max_length=64)
    observation: str = Field(min_length=1, max_length=2048)
    mechanism: str = Field(min_length=1, max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    is_evidence_against_hypothesis: list[str] = Field(default_factory=list, max_length=8)


class InvestigationGraph(BaseModel):
    """Runtime state container for an investigation.

    Holds entities / links / evidence blocks / scope boundary / iteration counter.
    Per PPT p15: "图不是画图, 而是状态容器".
    """

    entities: dict[str, EntitySet] = Field(default_factory=dict)
    links: list[EntitySetLink] = Field(default_factory=list, max_length=10_000)
    evidence_blocks: list[EvidenceBlock] = Field(default_factory=list, max_length=10_000)
    scope_boundary: set[str] = Field(default_factory=set)
    iteration: int = 0
    iteration_history: list[dict[str, Any]] = Field(default_factory=list, max_length=1000)
    case_brief: dict[str, Any] = Field(default_factory=dict)

    def add_entity(self, entity: EntitySet) -> None:
        """Insert or replace an entity keyed by its graph-unique id."""
        self.entities[entity.id] = entity

    def add_link(self, link: EntitySetLink) -> None:
        """Append a link, requiring both endpoints to already exist."""
        if link.src not in self.entities:
            raise ValueError(f"link src {link.src} not in graph entities")
        if link.dst not in self.entities:
            raise ValueError(f"link dst {link.dst} not in graph entities")
        self.links.append(link)
