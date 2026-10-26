"""AgentProtocol models (W8 Task 3.1, P2 计划)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


MessageType = Literal["evidence", "hypothesis", "claim", "action_request", "gate_request"]


class AgentMessage(BaseModel):
    """Structured message exchanged between agents per the protocol-first contract."""

    from_agent: str = Field(min_length=1, max_length=64)
    to_agent: str = Field(min_length=1, max_length=64)
    message_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""
