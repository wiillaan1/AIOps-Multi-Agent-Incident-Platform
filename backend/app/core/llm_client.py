"""
LLM Client - 带熔断 + Fallback 的 LLM 客户端

设计原则（来自调研）：
1. LLM 是"渐进式增强"不是替代层 —— 调用失败必须能 fallback
2. 所有 LLM call 都过熔断器 —— 连续失败触发熔断，保护上游
3. 每次调用前做 health check —— API key / 熔断器状态
4. 调用前后写审计（Phase A3 接入）

当前 provider 支持：
  - openai / deepseek / mimimax / ollama：统一走 langchain ChatOpenAI（OpenAI 兼容协议）
  - anthropic：langchain ChatAnthropic
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.audit import AuditAction, get_audit_logger
from app.config import LLMConfig, LangfuseConfig
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
from app.core.llm_cost import (
    LLMUsageRecord,
    compute_cost_usd,
    extract_usage_metadata,
    get_llm_usage_recorder,
)
from app.observability import LangfuseTracer, get_langfuse_tracer


logger = logging.getLogger(__name__)


class LLMUnavailable(Exception):
    """LLM 不可用（key 缺失 / 熔断打开 / 调用失败）"""
    def __init__(self, reason: str, breaker_state: CircuitState | None = None):
        self.reason = reason
        self.breaker_state = breaker_state
        msg = f"LLM 不可用: {reason}"
        if breaker_state:
            msg += f" (熔断器: {breaker_state.value})"
        super().__init__(msg)


@dataclass
class LLMCallResult:
    """LLM 调用结果（统一封装）"""
    content: str
    model: str
    provider: str
    latency_ms: int
    token_in: int = 0
    token_out: int = 0
    cost_usd: float = 0.0
    currency: str = "USD"
    via_fallback: bool = False
    fallback_reason: str | None = None


class LLMClient:
    """
    LLM 客户端（线程安全 + 异步安全）。

    用法：
        client = LLMClient(config)
        if client.is_available():
            result = await client.chat([HumanMessage(content="...")])
        else:
            fallback = get_fallback_result(...)
    """

    def __init__(
        self,
        config: LLMConfig,
        breaker: CircuitBreaker | None = None,
        langfuse_tracer: LangfuseTracer | None = None,
    ):
        self.config = config
        self.breaker = breaker or CircuitBreaker(
            name=f"llm-{config.provider}",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=60.0,
            half_open_max_calls=1,
        )
        self.langfuse_tracer = langfuse_tracer

    # ---------- 健康检查 ----------
    def is_available(self) -> bool:
        """
        快速健康检查（不需要实际调用 LLM）。
        返回 False 时调用方应直接走 fallback。
        """
        if not self.config.is_key_present():
            return False
        return self.breaker.state != CircuitState.OPEN

    def availability_reason(self) -> str:
        """返回不可用的具体原因（用于诊断/告警）"""
        if not self.config.is_key_present():
            return "LLM_API_KEY 未配置"
        state = self.breaker.state
        if state == CircuitState.OPEN:
            return f"熔断器 OPEN（最近失败次数过多，等待 {self.config.timeout_seconds}s 后探测）"
        if state == CircuitState.HALF_OPEN:
            return "熔断器 HALF_OPEN（探测中）"
        return "可用"

    # ---------- 实际调用 ----------
    async def chat(
        self,
        messages: list[Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        incident_id: str | None = None,
        caller_agent: str | None = None,
        **kwargs,
    ) -> LLMCallResult:
        """
        异步调用 LLM chat completion。

        抛出：
            LLMUnavailable: 不可用时（熔断/key 缺失/调用失败）

        入参：
            incident_id: 可选，关联到 incident 用于审计追溯
            caller_agent: 可选，调用方身份（"rca_agent"/"reflect_agent"/"harness" 等），
                          写入 aiops_model_invocations 表用于按 Agent 维度核算成本
        """
        if not self.config.is_key_present():
            raise LLMUnavailable("LLM_API_KEY 未配置")

        state = self.breaker.state
        if state == CircuitState.OPEN:
            # 审计：熔断打开事件
            get_audit_logger().record(
                actor=f"system:llm_client:{self.config.provider}",
                action=AuditAction.LLM_CIRCUIT_OPEN,
                target_type="llm",
                target_id=self.config.model,
                input_data={"incident_id": incident_id, "messages_count": len(messages)},
                output_data=None,
                details={
                    "incident_id": incident_id,
                    "failure_count": self.breaker._failure_count,
                    "timeout_seconds": self.config.timeout_seconds,
                },
            )
            # Langfuse trace（熔断事件也算）
            self._langfuse_trace(
                messages=messages, response=None, error="circuit_open",
                incident_id=incident_id, latency_ms=0,
            )
            raise LLMUnavailable(
                "熔断器 OPEN，连续失败次数过多",
                breaker_state=state,
            )

        start = time.time()
        try:
            sync_result = await asyncio.get_event_loop().run_in_executor(
                None, self._sync_chat, messages, temperature, max_tokens, kwargs
            )
            # 兼容旧版 _sync_chat 只返回 content 字符串、新版返回 (content, usage_meta) tuple
            if isinstance(sync_result, tuple) and len(sync_result) == 2:
                content, usage_meta = sync_result
            else:
                content = sync_result
                usage_meta = {"token_in": 0, "token_out": 0}
            self.breaker._on_success()  # noqa: SLF001 — 受控访问
            latency = int((time.time() - start) * 1000)

            # 计算成本（审查报告七、中期补 #7）
            token_in = int(usage_meta.get("token_in", 0) or 0)
            token_out = int(usage_meta.get("token_out", 0) or 0)
            cost_usd = compute_cost_usd(
                token_in=token_in,
                token_out=token_out,
                input_cost_per_1k=self.config.input_cost_per_1k,
                output_cost_per_1k=self.config.output_cost_per_1k,
            )

            # 审计：LLM 调用成功
            get_audit_logger().record(
                actor=f"system:llm_client:{self.config.provider}",
                action=AuditAction.LLM_CALLED,
                target_type="llm",
                target_id=self.config.model,
                input_data={"incident_id": incident_id, "messages_count": len(messages)},
                output_data={"content_preview": content[:500] if content else ""},
                details={
                    "incident_id": incident_id,
                    "latency_ms": latency,
                    "temperature": temperature or self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_tokens,
                    "token_in": token_in,
                    "token_out": token_out,
                    "cost_usd": cost_usd,
                    "caller_agent": caller_agent,
                },
            )
            # Langfuse trace
            self._langfuse_trace(
                messages=messages, response=content, incident_id=incident_id,
                latency_ms=latency,
            )

            # 持久化 LLM 调用记录（fire-and-forget）
            try:
                get_llm_usage_recorder().record(
                    LLMUsageRecord(
                        incident_id=incident_id,
                        model=self.config.model,
                        provider=self.config.provider,
                        token_in=token_in,
                        token_out=token_out,
                        cost_usd=cost_usd,
                        currency=self.config.cost_currency,
                        latency_ms=latency,
                        via_fallback=False,
                        caller_agent=caller_agent,
                        prompt_preview=str(messages[0])[:200] if messages else None,
                        response_preview=(content or "")[:200],
                    )
                )
            except Exception as e:
                logger.warning(f"[LLMClient] cost 持久化失败（不影响主链路）: {e}")

            return LLMCallResult(
                content=content,
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency,
                token_in=token_in,
                token_out=token_out,
                cost_usd=cost_usd,
                currency=self.config.cost_currency,
            )
        except CircuitBreakerOpen as e:
            raise LLMUnavailable(str(e), breaker_state=e.state) from e
        except Exception as e:
            self.breaker._on_failure()  # noqa: SLF001
            # 审计：LLM 调用失败
            get_audit_logger().record(
                actor=f"system:llm_client:{self.config.provider}",
                action=AuditAction.LLM_CALLED,
                target_type="llm",
                target_id=self.config.model,
                input_data={"incident_id": incident_id, "messages_count": len(messages)},
                output_data=None,
                details={
                    "status": "failed",
                    "exception_type": type(e).__name__,
                    "exception_msg": str(e)[:500],
                },
            )
            # Langfuse trace（失败也记）
            self._langfuse_trace(
                messages=messages, response=None,
                error=f"{type(e).__name__}: {e}"[:200],
                incident_id=incident_id, latency_ms=int((time.time() - start) * 1000),
            )
            raise LLMUnavailable(f"调用失败: {type(e).__name__}: {e}") from e

    def _langfuse_trace(
        self,
        messages: list[Any],
        response: str | None,
        incident_id: str | None,
        latency_ms: int,
        error: str | None = None,
    ) -> None:
        """Langfuse trace 封装（失败不抛异常）"""
        if self.langfuse_tracer is None or not self.langfuse_tracer.is_enabled:
            return
        try:
            self.langfuse_tracer.trace_chat(
                messages=messages,
                response=response,
                model=self.config.model,
                provider=self.config.provider,
                latency_ms=latency_ms,
                incident_id=incident_id,
                error=error,
            )
        except Exception as e:  # 兜底：trace 失败不影响主链路
            logger.warning(f"[LLMClient] Langfuse trace 兜底失败: {e}")

    def _sync_chat(
        self,
        messages: list[Any],
        temperature: float | None,
        max_tokens: int | None,
        kwargs: dict,
    ) -> tuple[str, dict[str, int]]:
        """同步调用（在线程池跑）

        返回：(content, usage_metadata)
        - content: 响应文本
        - usage_metadata: {"token_in": int, "token_out": int}，找不到时全 0

        兼容：测试 mock 可能直接返回 string，这种情况下 usage_metadata 会是 {0, 0}。
        """
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        # 转换消息为 langchain 格式
        lc_messages = []
        for m in messages:
            if isinstance(m, str):
                lc_messages.append(HumanMessage(content=m))
            elif hasattr(m, "content"):
                lc_messages.append(m)
            elif isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(HumanMessage(content=content))

        provider = self.config.provider
        if provider in ("openai", "deepseek", "mimimax", "ollama"):
            from langchain_openai import ChatOpenAI
            llm_kwargs = {
                "model": self.config.model,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
                "timeout": self.config.timeout_seconds,
                "max_retries": self.config.max_retries,
                "api_key": self.config.api_key,
            }
            base_url = self.config.base_url or self._default_base_url(provider)
            if base_url:
                llm_kwargs["base_url"] = base_url
            llm = ChatOpenAI(**llm_kwargs)
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(
                model=self.config.model,
                temperature=temperature if temperature is not None else self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                timeout=self.config.timeout_seconds,
                max_retries=self.config.max_retries,
                api_key=self.config.api_key,
                **(kwargs or {}),
            )
        else:
            raise LLMUnavailable(f"未知 provider: {provider}")

        response = llm.invoke(lc_messages)
        content = response.content if hasattr(response, "content") else str(response)
        usage_meta = extract_usage_metadata(response)
        return content, usage_meta

    @staticmethod
    def _default_base_url(provider: str) -> str | None:
        """OpenAI 兼容 provider 的默认 base_url"""
        defaults = {
            "openai": None,  # 用官方默认
            "deepseek": "https://api.deepseek.com/v1",
            "ollama": "http://localhost:11434/v1",
            "mimimax": None,  # 用户必须配
        }
        return defaults.get(provider)

    # ---------- 同步入口（兼容旧代码） ----------
    def chat_sync(self, messages: list[Any], **kwargs) -> LLMCallResult:
        """同步调用入口（包装 async）"""
        return asyncio.run(self.chat(messages, **kwargs))


# ---------- 全局单例 ----------
_client: LLMClient | None = None


def get_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """获取全局 LLM 客户端（懒加载）"""
    global _client
    if _client is None:
        from app.config import get_config
        cfg = config or get_config().llm
        _client = LLMClient(cfg)
    return _client


def reset_llm_client() -> None:
    """重置全局客户端（用于配置热更新）"""
    global _client
    _client = None
