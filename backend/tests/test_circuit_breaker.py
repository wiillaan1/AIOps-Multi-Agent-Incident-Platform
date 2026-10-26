"""
Circuit Breaker + LLM Client + Fallback 单元测试

覆盖：
1. CircuitBreaker 三态转换（CLOSED → OPEN → HALF_OPEN → CLOSED）
2. CircuitBreaker 短路过熔断期
3. CircuitBreaker 统计正确性
4. LLMClient.is_available() 在 key 缺失/熔断 OPEN 时返回 False
5. LLMFallback 三类降级输出格式
6. safe_chat 自动降级路径
"""
from __future__ import annotations

import time

import pytest

from app.config import LLMConfig
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)
from app.core.llm_client import LLMClient, LLMUnavailable, LLMCallResult
from app.core.llm_fallback import (
    summarize_alert,
    rca_root_cause,
    change_risk_narrative,
    safe_chat,
)


# =====================================================================
# CircuitBreaker 测试
# =====================================================================

class TestCircuitBreaker:
    """CircuitBreaker 状态机测试"""

    def test_closed_to_open_after_threshold_failures(self):
        """3 次连续失败 → OPEN"""
        cb = CircuitBreaker("test", failure_threshold=3, timeout_seconds=60)

        def fail():
            raise RuntimeError("simulated failure")

        for i in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)
            assert cb.state == CircuitState.CLOSED

        # 第 3 次失败触发熔断
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state == CircuitState.OPEN

    def test_open_blocks_subsequent_calls(self):
        """OPEN 状态下后续调用直接短路（不调函数）"""
        cb = CircuitBreaker("test", failure_threshold=2, timeout_seconds=60)

        def fail():
            raise RuntimeError("fail")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN

        call_count = 0

        def should_not_be_called():
            nonlocal call_count
            call_count += 1
            return "ok"

        with pytest.raises(CircuitBreakerOpen):
            cb.call(should_not_be_called)
        assert call_count == 0  # 短路成功

    def test_open_to_half_open_after_timeout(self):
        """OPEN 超时后下次调用转 HALF_OPEN"""
        cb = CircuitBreaker("test", failure_threshold=1, timeout_seconds=0.2)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

        time.sleep(0.25)
        # 下次调用触发懒转换
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        """HALF_OPEN 连续成功 → CLOSED"""
        cb = CircuitBreaker(
            "test", failure_threshold=1, success_threshold=2, timeout_seconds=0.1
        )

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        time.sleep(0.15)

        # 探测成功 1 次（仍 HALF_OPEN）
        cb.call(lambda: "ok")
        assert cb.state == CircuitState.HALF_OPEN

        # 第 2 次成功 → CLOSED
        cb.call(lambda: "ok")
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        """HALF_OPEN 失败 → 重新 OPEN"""
        cb = CircuitBreaker("test", failure_threshold=1, timeout_seconds=0.1)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        time.sleep(0.15)
        # 探测失败
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

    def test_success_in_closed_reduces_failure_count(self):
        """CLOSED 状态下成功会渐进减少失败计数（防毛刺）"""
        cb = CircuitBreaker("test", failure_threshold=5, timeout_seconds=60)

        # 失败 4 次（未达阈值）
        for _ in range(4):
            with pytest.raises(RuntimeError):
                cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.stats().failure_count == 4

        # 成功 1 次 → 失败计数减 1
        cb.call(lambda: "ok")
        assert cb.stats().failure_count == 3

    def test_reset_manual(self):
        """手动 reset 回到 CLOSED"""
        cb = CircuitBreaker("test", failure_threshold=1, timeout_seconds=60)

        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.stats().failure_count == 0

    def test_stats_accuracy(self):
        """stats() 统计字段正确性"""
        cb = CircuitBreaker("test", failure_threshold=3, timeout_seconds=60)

        cb.call(lambda: "ok")
        cb.call(lambda: "ok")
        with pytest.raises(RuntimeError):
            cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

        stats = cb.stats()
        assert stats.name == "test"
        assert stats.total_calls == 3
        assert stats.total_failures == 1
        assert stats.total_short_circuits == 0
        assert stats.last_failure_time is not None


# =====================================================================
# LLMClient 测试
# =====================================================================

class TestLLMClientAvailability:
    """LLMClient.is_available() 行为"""

    pytestmark = pytest.mark.allow_llm

    def test_unavailable_when_key_missing(self):
        cfg = LLMConfig(provider="openai", api_key="")
        client = LLMClient(cfg)
        assert client.is_available() is False
        assert "未配置" in client.availability_reason()

    def test_available_when_key_present_and_closed(self):
        cfg = LLMConfig(
            provider="openai",
            api_key="sk-test12345678901234567890",
        )
        client = LLMClient(cfg)
        assert client.is_available() is True

    def test_unavailable_when_breaker_open(self):
        cfg = LLMConfig(
            provider="openai",
            api_key="sk-test12345678901234567890",
        )
        client = LLMClient(cfg, breaker=CircuitBreaker(
            "test", failure_threshold=1, timeout_seconds=60
        ))
        # 触发熔断
        with pytest.raises(RuntimeError):
            client.breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert client.is_available() is False
        assert "熔断" in client.availability_reason()


class TestLLMClientChatFails:
    """LLMClient.chat() 失败行为（不需要真调 LLM）"""

    @pytest.mark.asyncio
    async def test_chat_raises_when_key_missing(self):
        cfg = LLMConfig(provider="openai", api_key="")
        client = LLMClient(cfg)
        with pytest.raises(LLMUnavailable, match="未配置"):
            await client.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_chat_raises_when_breaker_open(self):
        cfg = LLMConfig(
            provider="openai",
            api_key="sk-test12345678901234567890",
        )
        breaker = CircuitBreaker("test", failure_threshold=1, timeout_seconds=60)
        # 触发熔断
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("x")))

        client = LLMClient(cfg, breaker=breaker)
        with pytest.raises(LLMUnavailable, match="熔断器 OPEN"):
            await client.chat([{"role": "user", "content": "hi"}])


# =====================================================================
# LLMFallback 测试
# =====================================================================

class TestFallbackSummarize:
    def test_summarize_critical_alert(self):
        result = summarize_alert({
            "service": "order-service",
            "metric_name": "cpu_usage_percent",
            "severity": "critical",
            "value": 95,
            "threshold": 80,
        })
        assert "🚨" in result.headline
        assert result.severity == "critical"
        assert result.deviation_pct == pytest.approx(18.8, abs=0.1)

    def test_summarize_missing_optional_fields(self):
        result = summarize_alert({"service": "auth", "metric_name": "error_rate"})
        assert "auth" in result.headline
        assert result.deviation_pct is None
        assert "规则引擎摘要" in result.notes[0]


class TestFallbackRCA:
    def test_rca_uses_bayesian_when_available(self):
        result = rca_root_cause(
            {"service": "order", "metric_name": "cpu"},
            {"bayesian_top": {"hypothesis": "resource_saturation", "posterior": 0.85}},
        )
        assert result["root_cause"] == "resource_saturation"
        assert result["confidence"] == 0.85
        assert result["method"] == "rules_engine+bayesian"
        assert result["needs_human_review"] is False

    def test_rca_low_confidence_needs_review(self):
        result = rca_root_cause(
            {"service": "order", "metric_name": "cpu"},
            {"bayesian_top": {"hypothesis": "x", "posterior": 0.3}},
        )
        assert result["needs_human_review"] is True

    def test_rca_falls_back_to_history(self):
        result = rca_root_cause(
            {"service": "order", "metric_name": "cpu"},
            {"history_similar": [{"id": "inc-1", "root_cause": "db_slow", "similarity": 0.75}]},
        )
        assert result["root_cause"] == "db_slow"
        assert result["method"] == "rules_engine+history"

    def test_rca_heuristic_by_metric_name(self):
        for metric, expected in [
            ("cpu_usage", "CPU resource saturation"),
            ("memory_usage", "Memory leak"),
            ("p99_latency_ms", "Slow downstream"),
            ("error_rate", "Application exception"),
            ("db_pool_active", "Connection pool"),
        ]:
            result = rca_root_cause({"service": "x", "metric_name": metric})
            assert expected in result["root_cause"]

    def test_rca_unknown_metric(self):
        result = rca_root_cause({"service": "x", "metric_name": "weird_thing"})
        assert result["confidence"] == 0.2
        assert result["needs_human_review"] is True


class TestFallbackChangeRisk:
    def test_high_risk(self):
        text = change_risk_narrative(
            0.8, {"impact": 0.7, "complexity": 0.5}, "order-service"
        )
        assert "高风险" in text
        assert "0.80" in text
        assert "impact" in text

    def test_low_risk(self):
        text = change_risk_narrative(
            0.2, {"impact": 0.1}, "order-service"
        )
        assert "低风险" in text

    def test_multi_service_blast_radius_warning(self):
        text = change_risk_narrative(
            0.6, {"impact": 0.5}, "order", blast_radius="multi"
        )
        assert "跨多个服务" in text


# =====================================================================
# safe_chat 自动降级
# =====================================================================

class TestSafeChat:
    pytestmark = [pytest.mark.allow_llm, pytest.mark.asyncio]
    async def test_returns_fallback_when_unavailable(self):
        cfg = LLMConfig(provider="openai", api_key="")
        client = LLMClient(cfg)
        fallback = LLMCallResult(
            content="规则版答案",
            model="rules-engine",
            provider="fallback",
            latency_ms=0,
        )
        result = await safe_chat(client, [{"role": "user", "content": "x"}], fallback)
        assert result.via_fallback is True
        assert result.content == "规则版答案"
        assert "未配置" in result.fallback_reason

    @pytest.mark.asyncio
    async def test_raises_when_unavailable_and_no_fallback(self):
        cfg = LLMConfig(provider="openai", api_key="")
        client = LLMClient(cfg)
        with pytest.raises(LLMUnavailable):
            await safe_chat(client, [{"role": "user", "content": "x"}])
