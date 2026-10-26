"""
AIOps Agent Platform - Symptom Parser (Phase 1b)

把运维/客服/监控系统转述的"自然语言故障描述"解析为结构化症状，
让用户首次交互就能直接进 RCA，不用反复追问"哪个服务"。

示例：
  "客户反馈下单后页面一直转圈，等了 30 秒"
  → {
      "service": "order-service",
      "symptom": "high_latency",
      "urgency": "high",
      "time_hint": "now",
      "duration_seconds": 30,
      "user_facing": True,
      "raw_query": "客户反馈下单后页面一直转圈，等了 30 秒"
  }

设计原则：
1. 规则主 + LLM 兜底（与意图识别一致的双轨）
2. 字段提取全部受控（白名单 service / symptom 候选）
3. LLM 失败时降级到规则 + EntityExtractor
4. 可审计：所有抽取结果附带 reasoning
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# 已知服务白名单（解析时约束 LLM 输出）
SERVICE_WHITELIST = {
    "order-service", "payment-service", "user-service",
    "inventory-service", "api-gateway", "mysql-primary",
    "redis-cache", "kafka", "elasticsearch",
}

# 症状白名单
SYMPTOM_WHITELIST = {
    "high_latency", "timeout", "high_error_rate", "service_down",
    "high_cpu", "high_memory", "oom_killed", "oom",
    "connection_refused", "disk_full",
}

# 紧急度枚举
URGENCY_WHITELIST = {"low", "medium", "high", "critical"}


@dataclass
class ParsedSymptom:
    """症状解析结果。"""

    service: str = ""
    symptom: str = ""
    urgency: str = "medium"
    time_hint: str = "now"
    duration_seconds: int | None = None
    user_facing: bool = False
    raw_query: str = ""
    # 审计
    extraction_method: str = ""  # "rule" / "llm" / "rule_fallback"
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "symptom": self.symptom,
            "urgency": self.urgency,
            "time_hint": self.time_hint,
            "duration_seconds": self.duration_seconds,
            "user_facing": self.user_facing,
            "raw_query": self.raw_query,
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    def is_complete(self) -> bool:
        """是否包含 RCA 所需的最小字段集。"""
        return bool(self.service) and bool(self.symptom)


class SymptomParser:
    """
    症状自然语言解析器。

    规则流程：
    1. 规则尝试：用 EntityExtractor 抽取 service / symptom / time_range
    2. 时间正则：提取"X 秒/分钟/小时" → duration_seconds
    3. 紧急度规则：含"卡/慢" → high；含"挂/崩" → critical
    4. LLM 兜底：规则失败 / 字段不全时调用 LLM
    5. 校验：LLM 输出必须落在白名单内，否则丢弃
    """

    def __init__(self, llm_provider=None) -> None:
        from app.nlu.intent_llm import LLMProvider, RuleOnlyProvider
        self.llm_provider = llm_provider or RuleOnlyProvider()

    def parse(self, query: str) -> ParsedSymptom:
        """主入口：先规则后 LLM。"""
        if not query or not query.strip():
            return ParsedSymptom(raw_query=query, extraction_method="empty")

        # 阶段 1: 规则尝试
        rule_result = self._parse_by_rules(query)
        if rule_result.is_complete() and rule_result.confidence >= 0.6:
            return rule_result

        # 阶段 2: LLM 兜底
        llm_result = self._parse_by_llm(query, fallback=rule_result)
        if llm_result and llm_result.is_complete():
            return llm_result

        # 阶段 3: 用规则最佳结果兜底
        rule_result.extraction_method = "rule_fallback"
        rule_result.reasoning = (
            f"LLM 兜底未提取完整字段，回退到规则结果。"
            f"rule: {rule_result.reasoning}"
        )
        return rule_result

    # ==================== 规则阶段 ====================

    def _parse_by_rules(self, query: str) -> ParsedSymptom:
        """用 EntityExtractor + 时间正则 + 紧急度规则抽取。"""
        from app.nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        entity_result = extractor.extract(query)

        result = ParsedSymptom(
            raw_query=query,
            extraction_method="rule",
            time_hint=entity_result.time_range or "now",
            urgency=entity_result.urgency or "medium",
            user_facing=self._detect_user_facing(query),
        )
        if entity_result.services:
            result.service = entity_result.services[0]
        if entity_result.symptoms:
            result.symptom = entity_result.symptoms[0]

        # 时间正则：提取"X 秒/分钟/小时" → duration_seconds
        result.duration_seconds = self._extract_duration_seconds(query)

        # 紧急度规则（覆盖 EntityExtractor 的弱规则）
        result.urgency = self._refine_urgency(query, result.urgency)

        # 置信度评估
        filled = sum(bool(x) for x in (result.service, result.symptom))
        result.confidence = 0.3 + 0.35 * filled  # 0.3 / 0.65 / 1.0

        result.reasoning = (
            f"rule: service={result.service or '?'}, "
            f"symptom={result.symptom or '?'}, "
            f"urgency={result.urgency}"
        )
        return result

    def _extract_duration_seconds(self, query: str) -> int | None:
        # 支持阿拉伯数字（30 秒 / 30秒）和中文数字（一小时 / 一个小时）
        chinese_numbers = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
                          "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

        def _resolve(num_str: str) -> int | None:
            if num_str.isdigit():
                return int(num_str)
            if num_str in chinese_numbers:
                return chinese_numbers[num_str]
            return None

        # 按从长到短匹配：先匹配"小时"再匹配"分钟"再匹配"秒"（避免误匹配）
        m = re.search(r"(\d+|[一二两二三四五六七八九十])\s*小时", query)
        if m:
            v = _resolve(m.group(1))
            if v is not None:
                return v * 3600
        m = re.search(r"(\d+|[一二两二三四五六七八九十])\s*分钟?", query)
        if m:
            v = _resolve(m.group(1))
            if v is not None:
                return v * 60
        m = re.search(r"(\d+|[一二两二三四五六七八九十])\s*秒", query)
        if m:
            v = _resolve(m.group(1))
            if v is not None:
                return v
        return None

    def _refine_urgency(self, query: str, base: str) -> str:
        """基于关键字提升紧急度（仅升不降）。"""
        critical_words = ["挂", "崩", "全挂", "宕", "瘫", "完蛋"]
        high_words = ["卡", "慢", "超时", "严重", "大量", "批量"]
        if any(w in query for w in critical_words):
            return "critical"
        if any(w in query for w in high_words):
            return "high" if base in {"medium", "low"} else base
        return base

    def _detect_user_facing(self, query: str) -> bool:
        """检测是否用户报障（vs 系统自检）。"""
        indicators = ["客户", "用户", "反馈", "投诉", "报障", "工单"]
        return any(w in query for w in indicators)

    # ==================== LLM 兜底 ====================

    def _parse_by_llm(
        self,
        query: str,
        fallback: ParsedSymptom,
    ) -> ParsedSymptom | None:
        """调用 LLM 做结构化抽取（受白名单约束）。"""
        import asyncio

        from app.nlu.intent_llm import LLMIntentResult

        prompt = self._build_extraction_prompt(query, fallback)

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                llm_resp: LLMIntentResult = loop.run_until_complete(
                    self.llm_provider.classify(
                        query=query,
                        rule_intent=fallback.symptom or "unknown",
                        rule_confidence=fallback.confidence,
                        rule_clarification_questions=[],
                    )
                )
            finally:
                loop.close()
        except Exception as exc:
            logger.warning("Symptom LLM extraction failed", error=str(exc))
            return None

        # 解析 LLM 输出的 JSON（约定：reasoning 字段含 JSON）
        parsed = self._parse_llm_output(llm_resp, query)
        if parsed is None:
            return None

        # 白名单校验
        if parsed.service and parsed.service not in SERVICE_WHITELIST:
            # 未知服务名 → 不接受 LLM 输出
            parsed.service = ""
        if parsed.symptom and parsed.symptom not in SYMPTOM_WHITELIST:
            parsed.symptom = ""
        if parsed.urgency not in URGENCY_WHITELIST:
            parsed.urgency = "medium"
        return parsed

    def _build_extraction_prompt(self, query: str, fallback: ParsedSymptom) -> str:
        """构造 LLM prompt（在 MockLLMProvider 里被忽略，仅真实 Provider 用到）。"""
        return (
            f"Extract structured info from: {query}\n"
            f"Services allowed: {sorted(SERVICE_WHITELIST)}\n"
            f"Symptoms allowed: {sorted(SYMPTOM_WHITELIST)}\n"
            f"Output JSON: {json.dumps({'service': '', 'symptom': '', 'urgency': 'medium'})}"
        )

    def _parse_llm_output(
        self,
        llm_resp,
        query: str,
    ) -> ParsedSymptom | None:
        """
        MockLLMProvider 的 reasoning 字段约定为 JSON 字符串（用于测试）。
        真实 LLM Provider 应输出标准 JSON。
        """
        if not llm_resp or not llm_resp.reasoning:
            return None
        try:
            data = json.loads(llm_resp.reasoning)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(data, dict):
            return None

        return ParsedSymptom(
            service=str(data.get("service", "") or "").strip(),
            symptom=str(data.get("symptom", "") or "").strip(),
            urgency=str(data.get("urgency", "medium") or "medium").strip(),
            time_hint=str(data.get("time_hint", "now") or "now").strip(),
            duration_seconds=data.get("duration_seconds"),
            user_facing=bool(data.get("user_facing", False)),
            raw_query=query,
            extraction_method="llm",
            confidence=0.85,  # LLM 抽取默认高置信度
            reasoning=f"llm_extracted: {llm_resp.reasoning[:100]}",
        )
