"""
LangfuseTracer 测试 + LLMClient 集成测试

覆盖：
1. LangfuseTracer 启用/禁用路径
2. LangfuseTracer.trace_chat 调用 Langfuse SDK
3. LangfuseTracer.score 调用 Langfuse SDK
4. Langfuse 失败不阻断主链路
5. LLMClient 集成 Langfuse（成功/失败/熔断 都 trace）
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import LangfuseConfig, LLMConfig
from app.core.circuit_breaker import CircuitBreaker
from app.core.llm_client import LLMClient
from app.observability.langfuse_tracer import LangfuseTracer, reset_langfuse_tracer


# =====================================================================
# LangfuseTracer 测试
# =====================================================================

class TestLangfuseTracerInit:
    def test_disabled_when_enabled_false(self):
        config = LangfuseConfig(enabled=False, public_key="pk", secret_key="sk")
        tracer = LangfuseTracer(config)
        assert tracer.is_enabled is False

    def test_disabled_when_key_missing(self):
        config = LangfuseConfig(enabled=True, public_key="", secret_key="sk")
        tracer = LangfuseTracer(config)
        assert tracer.is_enabled is False

    def test_enabled_when_configured(self):
        config = LangfuseConfig(
            enabled=True,
            public_key="pk-test",
            secret_key="sk-test",
            host="https://cloud.langfuse.com",
        )
        with patch("langfuse.Langfuse") as mock_langfuse:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client
            tracer = LangfuseTracer(config)
            assert tracer.is_enabled is True
            mock_langfuse.assert_called_once()

    def test_init_failure_disables(self):
        config = LangfuseConfig(
            enabled=True, public_key="pk", secret_key="sk"
        )
        with patch("langfuse.Langfuse") as mock_langfuse:
            mock_langfuse.side_effect = RuntimeError("connection failed")
            tracer = LangfuseTracer(config)
            assert tracer.is_enabled is False

    def test_status_dict(self):
        config = LangfuseConfig(enabled=False)
        tracer = LangfuseTracer(config)
        status = tracer.status()
        assert status["enabled"] is False
        assert status["client_active"] is False


class TestLangfuseTracerTraceChat:
    def test_trace_chat_returns_none_when_disabled(self):
        config = LangfuseConfig(enabled=False)
        tracer = LangfuseTracer(config)
        assert tracer.trace_chat(messages=[{"role": "user", "content": "x"}]) is None

    def test_trace_chat_success(self):
        config = LangfuseConfig(
            enabled=True, public_key="pk", secret_key="sk"
        )
        with patch("langfuse.Langfuse") as mock_langfuse_cls:
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_trace.id = "trace-123"
            mock_client.trace.return_value = mock_trace
            mock_langfuse_cls.return_value = mock_client

            tracer = LangfuseTracer(config)
            result = tracer.trace_chat(
                messages=[{"role": "user", "content": "hi"}],
                response="hello",
                model="gpt-4o-mini",
                provider="openai",
                latency_ms=200,
                incident_id="inc-1",
            )
            assert result == "trace-123"
            mock_client.trace.assert_called_once()
            mock_trace.generation.assert_called_once()

    def test_trace_chat_with_error(self):
        config = LangfuseConfig(enabled=True, public_key="pk", secret_key="sk")
        with patch("langfuse.Langfuse") as mock_langfuse_cls:
            mock_client = MagicMock()
            mock_trace = MagicMock()
            mock_trace.id = "trace-err"
            mock_client.trace.return_value = mock_trace
            mock_langfuse_cls.return_value = mock_client

            tracer = LangfuseTracer(config)
            result = tracer.trace_chat(
                messages=[{"role": "user", "content": "x"}],
                response=None,
                error="connection timeout",
            )
            assert result == "trace-err"
            # 确认 generation.update(level="ERROR") 被调用
            mock_trace.generation.return_value.update.assert_called_once_with(
                level="ERROR", status_message="connection timeout"
            )

    def test_trace_chat_failure_does_not_raise(self):
        """Langfuse SDK 失败不应抛异常"""
        config = LangfuseConfig(enabled=True, public_key="pk", secret_key="sk")
        with patch("langfuse.Langfuse") as mock_langfuse_cls:
            mock_client = MagicMock()
            mock_client.trace.side_effect = RuntimeError("network error")
            mock_langfuse_cls.return_value = mock_client

            tracer = LangfuseTracer(config)
            # 不抛异常，返回 None
            result = tracer.trace_chat(messages=[{"role": "user", "content": "x"}])
            assert result is None


class TestLangfuseTracerScore:
    def test_score_returns_false_when_disabled(self):
        config = LangfuseConfig(enabled=False)
        tracer = LangfuseTracer(config)
        assert tracer.score("trace-1", "accuracy", 0.95) is False

    def test_score_returns_false_when_trace_id_none(self):
        config = LangfuseConfig(enabled=True, public_key="pk", secret_key="sk")
        with patch("langfuse.Langfuse") as mock_langfuse_cls:
            mock_langfuse_cls.return_value = MagicMock()
            tracer = LangfuseTracer(config)
            assert tracer.score(None, "accuracy", 0.95) is False

    def test_score_success(self):
        config = LangfuseConfig(enabled=True, public_key="pk", secret_key="sk")
        with patch("langfuse.Langfuse") as mock_langfuse_cls:
            mock_client = MagicMock()
            mock_langfuse_cls.return_value = mock_client

            tracer = LangfuseTracer(config)
            result = tracer.score("trace-1", "accuracy", 0.95, "good response")
            assert result is True
            mock_client.score.assert_called_once_with(
                trace_id="trace-1", name="accuracy", value=0.95, comment="good response"
            )

    def test_score_failure_returns_false(self):
        config = LangfuseConfig(enabled=True, public_key="pk", secret_key="sk")
        with patch("langfuse.Langfuse") as mock_langfuse_cls:
            mock_client = MagicMock()
            mock_client.score.side_effect = RuntimeError("api error")
            mock_langfuse_cls.return_value = mock_client

            tracer = LangfuseTracer(config)
            result = tracer.score("trace-1", "accuracy", 0.95)
            assert result is False


# =====================================================================
# LLMClient 集成测试
# =====================================================================

class TestLLMClientLangfuseIntegration:
    @pytest.mark.asyncio
    async def test_chat_does_not_call_langfuse_when_not_provided(self):
        cfg = LLMConfig(
            provider="openai",
            api_key="sk-test12345678901234567890",
        )
        breaker = CircuitBreaker("test", failure_threshold=1, timeout_seconds=60)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

        client = LLMClient(cfg, breaker=breaker, langfuse_tracer=None)
        with pytest.raises(Exception):
            await client.chat([{"role": "user", "content": "hi"}], incident_id="inc-1")
        # 不抛异常就 OK（无 tracer）

    @pytest.mark.asyncio
    async def test_chat_calls_langfuse_on_circuit_open(self):
        cfg = LLMConfig(
            provider="openai",
            api_key="sk-test12345678901234567890",
        )
        breaker = CircuitBreaker("test", failure_threshold=1, timeout_seconds=60)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

        mock_tracer = MagicMock()
        mock_tracer.is_enabled = True

        client = LLMClient(cfg, breaker=breaker, langfuse_tracer=mock_tracer)
        with pytest.raises(Exception):
            await client.chat([{"role": "user", "content": "hi"}], incident_id="inc-1")

        # 确认 trace_chat 被调用（circuit_open）
        mock_tracer.trace_chat.assert_called_once()
        call_kwargs = mock_tracer.trace_chat.call_args.kwargs
        assert call_kwargs["error"] == "circuit_open"
        assert call_kwargs["incident_id"] == "inc-1"

    @pytest.mark.asyncio
    async def test_chat_calls_langfuse_disabled_tracer_noop(self):
        cfg = LLMConfig(provider="openai", api_key="sk-test12345678901234567890")
        breaker = CircuitBreaker("test", failure_threshold=1, timeout_seconds=60)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

        mock_tracer = MagicMock()
        mock_tracer.is_enabled = False  # 禁用

        client = LLMClient(cfg, breaker=breaker, langfuse_tracer=mock_tracer)
        with pytest.raises(Exception):
            await client.chat([{"role": "user", "content": "hi"}])

        # disabled 时不应调用 trace_chat
        mock_tracer.trace_chat.assert_not_called()
