"""Focused regression coverage for RCA reflection-context propagation."""

from __future__ import annotations

import pytest

from app.agents.rca_agent import RCAAgent, RCAInput
from app.models.agent import AgentExecutionContext
from app.models.events import AlertEvent, SeverityLevel


@pytest.mark.asyncio
async def test_process_passes_reflection_context_to_synthesis(monkeypatch):
    agent = RCAAgent()
    reflection_context = {"evidence_boost": "log", "bayesian_penalty": 0.15}
    observed: dict[str, object] = {}

    async def no_memory(*_args, **_kwargs):
        return []

    async def no_logs(*_args, **_kwargs):
        return {
            "available": False,
            "entries_count": 0,
            "cause_hits": {},
            "boosted_cause": None,
            "confidence_boost": 0.0,
            "samples": {},
            "source": "unavailable",
        }

    def synthesize(*_args, **kwargs):
        observed["reflection_context"] = kwargs.get("reflection_context")
        return "resource_exhaustion", 0.8, {
            "log_corroboration": False,
            "log_divergence": False,
        }

    monkeypatch.setattr(agent, "_search_historical_incidents", no_memory)
    monkeypatch.setattr(agent, "_query_log_evidence", no_logs)
    monkeypatch.setattr(agent, "_synthesize_analysis", synthesize)

    result = await agent.process(
        RCAInput(
            alert=AlertEvent(
                service="order-service",
                metric="cpu_usage_percent",
                value=95.0,
                threshold=80.0,
                severity=SeverityLevel.HIGH,
            ),
            incident_id="inc-1",
            reflection_context=reflection_context,
        ),
        AgentExecutionContext(incident_id="inc-1"),
    )

    assert result.success is True
    assert observed["reflection_context"] == reflection_context


def test_reflection_context_alias_remains_compatible():
    context = {"evidence_boost": "log"}

    input_data = RCAInput.model_validate(
        {
            "alert": {
                "service": "order-service",
                "metric": "cpu_usage_percent",
                "value": 95.0,
                "threshold": 80.0,
                "severity": "high",
            },
            "_reflection_context": context,
        }
    )

    assert input_data.reflection_context == context
    assert input_data._reflection_context == context
