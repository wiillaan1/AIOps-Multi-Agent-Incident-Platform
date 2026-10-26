"""
AIOps Agent Platform - Intent LLM Fallback Layer

为 IntentClassifier 提供 LLM 兜底。设计原则：

1. **规则主 + LLM 兜底**：高置信度正则直接返回；低置信度才调 LLM
2. **抽象 Provider**：`LLMProvider` 接口 + 默认无操作的 `RuleOnlyProvider` + 测试用 `MockLLMProvider`
3. **降级**：LLM 调用失败/超时不阻塞主流程，回退到原规则结果
4. **可审计**：每次 LLM 决策写 reasoning 进 `UserIntent.sub_intent`
5. **不引入重依赖**：避免 langchain 等；HTTP 调用走 aiohttp 或外部注入

后续接入真实 LLM（OpenAI / DeepSeek / Ollama）只需实现 `LLMProvider.classify()`。
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMIntentResult:
    """LLM 分类结果。"""

    intent: str  # 必须命中 IntentType 之一的 value
    confidence: float
    reasoning: str = ""  # 用于审计


class LLMProvider(abc.ABC):
    """LLM Provider 抽象基类。"""

    name: str = "abstract"

    @abc.abstractmethod
    async def classify(
        self,
        query: str,
        rule_intent: str,
        rule_confidence: float,
        rule_clarification_questions: list[str],
    ) -> LLMIntentResult:
        """
        让 LLM 基于规则结果 + 追问列表重新分类。

        Args:
            query: 原始用户输入
            rule_intent: 规则命中的意图（可能是 NEED_CLARIFICATION / GENERAL_QUESTION）
            rule_confidence: 规则置信度
            rule_clarification_questions: 规则已生成的追问列表

        Returns:
            LLMIntentResult: LLM 重新分类的结果
        """
        raise NotImplementedError


class RuleOnlyProvider(LLMProvider):
    """默认 Provider：不调任何外部服务，直接返回规则结果。"""

    name = "rule_only"

    async def classify(
        self,
        query: str,
        rule_intent: str,
        rule_confidence: float,
        rule_clarification_questions: list[str],
    ) -> LLMIntentResult:
        # 不调用 LLM，直接透传规则结果
        return LLMIntentResult(
            intent=rule_intent,
            confidence=rule_confidence,
            reasoning="rule_only_provider_no_llm_call",
        )


class MockLLMProvider(LLMProvider):
    """测试用 Provider：可注入预设响应。"""

    name = "mock"

    def __init__(
        self,
        response: LLMIntentResult | None = None,
        raise_on_call: Exception | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.response = response
        self.raise_on_call = raise_on_call
        self.delay_seconds = delay_seconds
        self.call_count = 0

    async def classify(
        self,
        query: str,
        rule_intent: str,
        rule_confidence: float,
        rule_clarification_questions: list[str],
    ) -> LLMIntentResult:
        self.call_count += 1
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.raise_on_call:
            raise self.raise_on_call
        if self.response:
            return self.response
        # 默认 mock：直接透传规则
        return LLMIntentResult(
            intent=rule_intent,
            confidence=rule_confidence,
            reasoning="mock_default_passthrough",
        )


# 触发 LLM 兜底的阈值（规则置信度低于此才调 LLM）
DEFAULT_LLM_TRIGGER_THRESHOLD = 0.5
# LLM 调用的超时时间（秒）
DEFAULT_LLM_TIMEOUT_SECONDS = 3.0


class IntentLLMGateway:
    """意图识别 LLM 网关：决定何时调 LLM、如何降级、如何审计。"""

    def __init__(
        self,
        provider: LLMProvider,
        trigger_threshold: float = DEFAULT_LLM_TRIGGER_THRESHOLD,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        self.provider = provider
        self.trigger_threshold = trigger_threshold
        self.timeout_seconds = timeout_seconds
        # 统计
        self.llm_call_count = 0
        self.llm_fallback_count = 0

    def should_call_llm(
        self,
        rule_intent: str,
        rule_confidence: float,
    ) -> bool:
        """决定是否要调用 LLM 兜底。

        规则：
        - 规则已经命中 NEED_CLARIFICATION → 不调（追问是更精准的动作）
        - 规则置信度 >= threshold → 不调（规则够用）
        - 否则 → 调 LLM
        """
        if rule_intent == "need_clarification":
            return False
        if rule_confidence >= self.trigger_threshold:
            return False
        return True

    async def maybe_refine(
        self,
        query: str,
        rule_intent: str,
        rule_confidence: float,
        rule_clarification_questions: list[str],
    ) -> dict[str, Any]:
        """
        可能调用 LLM，返回审计字段：
        - llm_called: bool
        - llm_intent: str | None
        - llm_confidence: float | None
        - llm_reasoning: str | None
        - provider: str
        - latency_ms: int
        - fallback_used: bool  # 是否触发了降级
        """
        result: dict[str, Any] = {
            "llm_called": False,
            "llm_intent": None,
            "llm_confidence": None,
            "llm_reasoning": None,
            "provider": self.provider.name,
            "latency_ms": 0,
            "fallback_used": False,
        }

        if not self.should_call_llm(rule_intent, rule_confidence):
            return result

        started = time.time()
        self.llm_call_count += 1
        result["llm_called"] = True  # 已决定要调，无论成功失败都标记
        try:
            llm_result = await asyncio.wait_for(
                self.provider.classify(
                    query, rule_intent, rule_confidence, rule_clarification_questions
                ),
                timeout=self.timeout_seconds,
            )
            result["llm_intent"] = llm_result.intent
            result["llm_confidence"] = llm_result.confidence
            result["llm_reasoning"] = llm_result.reasoning
            logger.info(
                "Intent LLM refinement applied",
                query=query[:50],
                rule_intent=rule_intent,
                rule_confidence=rule_confidence,
                llm_intent=llm_result.intent,
                llm_confidence=llm_result.confidence,
            )
        except asyncio.TimeoutError:
            self.llm_fallback_count += 1
            result["fallback_used"] = True
            logger.warning(
                "Intent LLM timeout, falling back to rule result",
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            self.llm_fallback_count += 1
            result["fallback_used"] = True
            logger.warning(
                "Intent LLM call failed, falling back",
                error=str(exc),
            )
        finally:
            result["latency_ms"] = int((time.time() - started) * 1000)

        return result

    def merge_into(
        self,
        rule_intent: str,
        rule_confidence: float,
        rule_clarification_questions: list[str],
        rule_sub_intent: str,
        llm_result: dict[str, Any],
    ) -> dict[str, Any]:
        """把 LLM 兜底结果合并到最终输出。

        Returns:
            dict 含 final_intent / final_confidence / clarification_questions / sub_intent / llm_meta
        """
        if not llm_result["llm_called"] or llm_result["fallback_used"]:
            return {
                "final_intent": rule_intent,
                "final_confidence": rule_confidence,
                "clarification_questions": rule_clarification_questions,
                "sub_intent": rule_sub_intent,
                "llm_meta": llm_result,
            }
        return {
            "final_intent": llm_result["llm_intent"] or rule_intent,
            "final_confidence": max(
                rule_confidence,
                llm_result["llm_confidence"] or 0.0,
            ),
            "clarification_questions": rule_clarification_questions,
            "sub_intent": (
                f"llm_refined:{llm_result['llm_reasoning']}"
                if llm_result["llm_reasoning"]
                else rule_sub_intent
            ),
            "llm_meta": llm_result,
        }
