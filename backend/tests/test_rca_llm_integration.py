"""RCA 主链路 LLM 接入契约测试。"""
from __future__ import annotations

import pytest

from app.agents.rca_agent import RCAAgent, RCAInput
from app.core.llm_client import LLMCallResult
from app.models.agent import AgentExecutionContext
from app.models.events import AlertEvent, SeverityLevel


class FakeLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str:
        return "available"

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return LLMCallResult(
            content=self.content,
            model="glm-5.3",
            provider="openai",
            latency_ms=12,
        )


@pytest.mark.asyncio
async def test_rca_process_calls_llm_and_records_provenance(monkeypatch):
    fake = FakeLLMClient(
        '{"root_cause":"traffic_spike","confidence":0.91,'
        '"contributing_factors":["high_cpu"],'
        '"recommended_actions":["scale_up_resources"],'
        '"evidence_refs":["alert.metric","alert.value"]}'
    )
    monkeypatch.setattr("app.agents.rca_agent.get_llm_client", lambda: fake)
    agent = RCAAgent()
    alert = AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.CRITICAL,
    )

    result = await agent.process(
        RCAInput(alert=alert, incident_id="inc-llm-001"),
        AgentExecutionContext(incident_id="inc-llm-001"),
    )

    assert result.success is True
    assert len(fake.calls) == 1
    assert fake.calls[0]["kwargs"]["incident_id"] == "inc-llm-001"
    rca = result.output_data["rca_event"]
    assert rca["root_cause"] == "traffic_spike"
    assert rca["evidence"]["llm"]["provider"] == "openai"
    assert rca["evidence"]["llm"]["model"] == "glm-5.3"
    assert rca["evidence"]["llm"]["via_fallback"] is False


@pytest.mark.asyncio
async def test_rca_process_falls_back_when_llm_unavailable(monkeypatch):
    class Unavailable:
        def is_available(self):
            return False

        def availability_reason(self):
            return "LLM_API_KEY 未配置"

    monkeypatch.setattr("app.agents.rca_agent.get_llm_client", lambda: Unavailable())
    agent = RCAAgent()
    alert = AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.CRITICAL,
    )

    result = await agent.process(
        RCAInput(alert=alert, incident_id="inc-fallback-001"),
        AgentExecutionContext(incident_id="inc-fallback-001"),
    )

    assert result.success is True
    rca = result.output_data["rca_event"]
    assert rca["evidence"]["llm"]["via_fallback"] is True
    assert "LLM_API_KEY" in rca["evidence"]["llm"]["fallback_reason"]
