"""AgentProtocol + W7 5 角色 adapter (W8 Task 3.1, P2 计划).

Implements PPT P5 protocol-first multi-agent pattern:
"协议优先于角色. Agent 之间交换 Evidence Block, 不自由聊天."

W7 现有 5 角色 (Intent/RCA/Heal/Monitor/Orchestrator) 内部实现字节不变,
只通过本 adapter 披一层协议外壳.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol as TypingProtocol

from app.models.gate import CounterfactualCheck, CounterfactualClaim
from app.models.protocol import AgentMessage
from app.models.umodel import EvidenceBlock, InvestigationGraph
from app.services.counterfactual_gate import CounterfactualGate
from app.services.scope import DFSScope
from app.services.umodel_store import InvestigationGraphStore


class AgentProtocol(TypingProtocol):
    """Protocol every W7 agent must satisfy (structural typing)."""

    graph: InvestigationGraph
    hypotheses: list[dict[str, Any]]

    def submit_evidence(self, block: EvidenceBlock) -> None: ...
    def request_next_hop(self, current: str) -> list[str]: ...
    def cross_gate(self, claim: CounterfactualClaim) -> CounterfactualCheck: ...
    def record_hypothesis(self, agent: str, text: str, confidence: float) -> None: ...
    def send_message(
        self, from_agent: str, to_agent: str, message_type: str, payload: dict[str, Any]
    ) -> AgentMessage: ...
    def persist(self) -> None: ...


class W7AgentAdapter:
    """Wrap W7 5 角色 with protocol-based Evidence Block exchange."""

    def __init__(
        self,
        store: InvestigationGraphStore,
        graph: InvestigationGraph,
        case_id: str,
        tenant: str = "default",
        k_hop: int = 2,
        w7_inner_orchestrator: Any | None = None,
    ) -> None:
        """Initialize adapter; bind store/graph/case and optional inner W7 orchestrator reference."""
        self.store = store
        self.graph = graph
        self.case_id = case_id
        self.tenant = tenant
        self.scope_engine = DFSScope(store, k_hop=k_hop)
        self.hypotheses: list[dict[str, Any]] = []
        self.messages: list[AgentMessage] = []
        self._w7_inner = w7_inner_orchestrator

    def submit_evidence(self, block: EvidenceBlock) -> None:
        """Append evidence block to graph and expand scope to admit its object_ref."""
        if block.object_ref not in self.graph.entities:
            self.graph.add_entity(_entity_from_evidence(block))
        self.graph.evidence_blocks.append(block)
        self.scope_engine.expand_scope_on_evidence(self.graph, block)

    def request_next_hop(self, current: str) -> list[str]:
        """Return neighbors of ``current`` constrained to the live scope boundary."""
        nbrs = self.store.neighbors(self.graph, current, k_hop=self.scope_engine.k_hop)
        return [n for n in nbrs if n in self.graph.scope_boundary or n == current]

    def cross_gate(self, claim: CounterfactualClaim) -> CounterfactualCheck:
        """Evaluate ``claim`` against graph state via the counterfactual gate."""
        return CounterfactualGate(self.graph).evaluate(claim)

    def record_hypothesis(self, agent: str, text: str, confidence: float) -> None:
        """Append a hypothesis entry attributed to ``agent``."""
        self.hypotheses.append(
            {
                "agent": agent,
                "hypothesis": text,
                "confidence": confidence,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: dict[str, Any],
    ) -> AgentMessage:
        """Construct an AgentMessage, append to history, and return it."""
        msg = AgentMessage(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,  # type: ignore[arg-type]
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.messages.append(msg)
        return msg

    def persist(self) -> None:
        """Atomically save the current graph to the store under (tenant, case_id)."""
        self.store.save(self.tenant, self.case_id, self.graph)

    def invoke_w7_orchestrator(self, *args: Any, **kwargs: Any) -> Any:
        """Passthrough to W7 Orchestrator — does NOT mutate inner method bytes."""
        if self._w7_inner is None:
            return None
        return self._w7_inner.run_sequential_with_reflection(*args, **kwargs)


def _entity_from_evidence(block: EvidenceBlock) -> Any:
    """Best-effort: derive EntitySet from object_ref, fallback to generic service entity."""
    from app.models.umodel import EntitySet, EntityState, EntityType

    raw = block.object_ref
    if ":" in raw:
        prefix, _, tail = raw.partition(":")
        try:
            return EntitySet(
                entity_type=EntityType(prefix),
                entity_id=tail,
                state=EntityState.UNKNOWN,
            )
        except ValueError:
            pass
    return EntitySet(
        entity_type=EntityType.SERVICE,
        entity_id=raw,
        state=EntityState.UNKNOWN,
    )
