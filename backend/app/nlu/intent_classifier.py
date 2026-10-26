"""
AIOps Agent Platform - Intent Classifier

识别用户自然语言问题的意图类型。
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """用户意图类型"""
    FAULT_DIAGNOSIS = "fault_diagnosis"      # 故障诊断："为什么下单这么慢？"
    METRIC_QUERY = "metric_query"             # 指标查询："CPU 使用率多少？"
    HEAL_REQUEST = "heal_request"             # 修复请求："帮我重启一下订单服务"
    HISTORY_LOOKUP = "history_lookup"         # 历史查询："上次类似的故障怎么处理的？"
    BUSINESS_CHECK = "business_check"         # 业务检查："有没有重复扣款？"
    NEED_CLARIFICATION = "need_clarification" # 需要追问（反思结果）
    GENERAL_QUESTION = "general_question"     # 一般问题


# 各意图的必填字段（用于反思追问）
# 注意：HEAL_REQUEST 的 "action" 不在必需字段中 — 动作由用户点"确认"按钮时再确认
INTENT_REQUIRED_FIELDS: dict[IntentType, list[str]] = {
    IntentType.FAULT_DIAGNOSIS: ["service"],
    IntentType.METRIC_QUERY: ["service"],
    IntentType.HEAL_REQUEST: ["service"],
    IntentType.HISTORY_LOOKUP: ["service"],
    IntentType.BUSINESS_CHECK: ["business_domain"],
    IntentType.NEED_CLARIFICATION: [],
    IntentType.GENERAL_QUESTION: [],
}


class UserIntent(BaseModel):
    """用户意图识别结果"""
    intent: IntentType = Field(description="识别到的意图类型")
    confidence: float = Field(default=0.0, description="置信度")
    sub_intent: str = Field(default="", description="子意图")
    raw_query: str = Field(default="", description="原始查询")
    # W1b 反思产物
    needs_clarification: bool = Field(default=False, description="是否需要追问")
    clarification_questions: list[str] = Field(
        default_factory=list,
        description="追问问题列表（仅当 needs_clarification=True 时填充）",
    )
    reflection_reason: str = Field(
        default="",
        description="触发反思的原因（缺字段 / 低置信度 / 无匹配）",
    )


# 意图匹配规则（正则 + 关键词）
# 注意：query 已被转为小写（query_lower），所有模式用小写
INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.FAULT_DIAGNOSIS: [
        r"为什么.*(慢|卡|挂|不行|出错|报错|超时|崩|异常|故障)",
        r"(是不是|是不是已经|已经).*(挂|崩|坏|死|宕|瘫|不行|有问题|出问题|故障)",
        r".*(出问题|出故障|出异常|不行)了",
        r".*(挂了|崩了|坏了|死了|宕了|瘫了)",
        r"帮我看下.*怎么回事",
        r"排查.*",
        r"什么原因.*",
        r"怎么.*(这么慢|这么卡|报错|挂了|回事)",
        r"(慢|卡|超时|错误|异常|故障|挂了|崩了)$",
        r".*(一直|老是|总是|经常).*(转圈|卡|慢|超时|报错)",
    ],
    IntentType.METRIC_QUERY: [
        r".*(cpu|内存|延迟|qps|tps|吞吐|磁盘|网络).*多少",
        r".*(cpu|内存|延迟|qps|tps|吞吐|磁盘|网络).*(多少|怎么样|什么|如何|查询|查看|看看|看下)",
        r"查.*指标",
        r".*当前.*(状态|值|情况)",
        r"看一下.*(cpu|内存|延迟)",
    ],
    IntentType.HEAL_REQUEST: [
        r"帮我.*(修|重启|扩容|回滚|恢复|处理|解决|改)",
        r"怎么.*(处理|解决|修|恢复|修好)",
        r"(重启|回滚|扩容|缩容).*",
        r".*执行.*(重启|回滚|扩容)",
    ],
    IntentType.HISTORY_LOOKUP: [
        r"上次.*(故障|问题|异常).*(怎么|如何).*",
        r"最近.*(告警|故障|异常)",
        r"历史.*(记录|故障|告警)",
        r"以前.*类似.*",
        r".*以前.*(怎么|如何).*处理",
    ],
    IntentType.BUSINESS_CHECK: [
        r"有没有.*(重复扣款|超卖|对账不平|优惠券|卡单|多扣|少扣)",
        r"检查.*(业务|交易|订单|支付|库存|扣款)",
        r".*(扣.*两次|多.*扣|少.*扣|重复.*扣)",
        r".*(超卖|少货|库存不对|对账不平)",
        r".*业务.*(异常|问题|故障|错误)",
    ],
}


class IntentClassifier:
    """
    用户意图分类器。

    规则主 + LLM 兜底：
    - 高置信度正则直接返回（不调 LLM，零延迟零成本）
    - 低置信度时调用 LLM 重新分类（可降级回规则结果）
    - 不依赖 LLM 可在离线环境运行（默认 RuleOnlyProvider）
    """

    def __init__(self, llm_gateway=None) -> None:
        """
        Args:
            llm_gateway: IntentLLMGateway 实例；为 None 时使用 RuleOnlyProvider
        """
        from app.nlu.intent_llm import IntentLLMGateway, RuleOnlyProvider
        self.llm_gateway = llm_gateway or IntentLLMGateway(
            provider=RuleOnlyProvider(),
        )

    # 意图优先级加权关键词（解决歧义）
    INTENT_BOOST_KEYWORDS: dict[IntentType, list[str]] = {
        IntentType.HISTORY_LOOKUP: [
            "上次", "以前", "历史", "最近", "之前", "曾经", "过去", "前几天",
        ],
        IntentType.BUSINESS_CHECK: [
            "有没有", "检查一下", "超卖", "重复扣款", "对账不平",
        ],
        IntentType.FAULT_DIAGNOSIS: [
            "为什么", "是不是", "怎么回事", "什么原因", "排查",
        ],
    }

    def classify(self, query: str) -> UserIntent:
        """
        对用户查询进行意图分类。

        W1b + W6.7 升级：规则主 + LLM 兜底
        1. 规则匹配 + 关键词加权（W1.0）
        2. 低置信度 → 反思"是不是漏字段了？"（W1b）
        3. 命中意图但缺必填字段 → 反思"该追问什么？"（W1b）
        4. 规则仍不确定 → 调用 LLM 兜底重分类（W6.7）

        Args:
            query: 用户自然语言查询

        Returns:
            UserIntent: 分类结果（含反思产物 + LLM 兜底审计）
        """
        # 导入放在函数内避免循环依赖
        from app.agents.reflection import reflect_missing_info

        if not query or not query.strip():
            return UserIntent(
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.0,
                raw_query=query,
                reflection_reason="empty_query",
            )

        query_lower = query.lower().strip()
        scores: dict[IntentType, float] = {}

        for intent, patterns in INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1.0
            # 归一化
            scores[intent] = score / max(len(patterns), 1)

        # 关键词加权（解决意图歧义）
        for intent, keywords in self.INTENT_BOOST_KEYWORDS.items():
            if intent in scores:
                boost = sum(1.0 for kw in keywords if kw in query_lower)
                scores[intent] += boost * 0.15  # 每个关键词 +0.15

        # 选出最高分意图
        if not scores:
            return UserIntent(
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.0,
                raw_query=query,
                reflection_reason="no_pattern_matched",
            )

        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]

        if best_score < 0.1:
            return UserIntent(
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.0,
                raw_query=query,
                reflection_reason="all_scores_below_threshold",
            )

        confidence = min(best_score, 1.0)

        # ===== W1b 反思：检查必填字段 =====
        required = INTENT_REQUIRED_FIELDS.get(best_intent, [])
        parsed = self._extract_fields(query_lower)
        missing_questions = reflect_missing_info(
            intent=best_intent.value,
            parsed_fields=parsed,
            required_fields=required,
        )

        # ===== W6.7 LLM 兜底：规则不够确定时让 LLM 重分类 =====
        # 仅对"规则命中但置信度低"的场景调 LLM；NEED_CLARIFICATION 已经够精准不调
        rule_clarification = missing_questions
        rule_sub_intent = ""

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            in_event_loop = loop.is_running()
        except RuntimeError:
            in_event_loop = False

        if in_event_loop:
            # 当前在事件循环里：跳过 LLM（避免嵌套），只用规则结果
            llm_meta: dict[str, Any] = {
                "llm_called": False,
                "provider": self.llm_gateway.provider.name,
                "latency_ms": 0,
                "fallback_used": False,
                "skipped_reason": "already_in_event_loop",
            }
        else:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                llm_meta = loop.run_until_complete(
                    self.llm_gateway.maybe_refine(
                        query=query,
                        rule_intent=(
                            IntentType.NEED_CLARIFICATION.value
                            if missing_questions
                            else best_intent.value
                        ),
                        rule_confidence=confidence,
                        rule_clarification_questions=rule_clarification,
                    )
                )
            finally:
                loop.close()

        # 合并 LLM 结果
        merged = self.llm_gateway.merge_into(
            rule_intent=(
                IntentType.NEED_CLARIFICATION.value
                if missing_questions
                else best_intent.value
            ),
            rule_confidence=confidence,
            rule_clarification_questions=rule_clarification,
            rule_sub_intent=rule_sub_intent,
            llm_result=llm_meta,
        )

        if missing_questions:
            # 规则已生成追问，NEED_CLARIFICATION 不被 LLM 覆盖
            return UserIntent(
                intent=IntentType.NEED_CLARIFICATION,
                confidence=merged["final_confidence"],
                raw_query=query,
                needs_clarification=True,
                clarification_questions=rule_clarification,
                sub_intent=merged["sub_intent"],
                reflection_reason="missing_required_fields",
            )

        return UserIntent(
            intent=IntentType(merged["final_intent"]),
            confidence=merged["final_confidence"],
            raw_query=query,
            sub_intent=merged["sub_intent"],
            reflection_reason="",
        )

    def _extract_fields(self, query_lower: str) -> dict[str, str]:
        """从 query 里粗略提取 service / symptom / action / business_domain。

        不要求高精度（这里是反思层兜底，RCA 阶段会有更结构化的抽取）。
        复用项目里既有的服务名 / 症状 / 业务域同义词映射（与 NLU 阶段共享）。
        """
        from app.nlu.entity_extractor import EntityExtractor

        extractor = EntityExtractor()
        result = extractor.extract(query_lower)

        fields: dict[str, str] = {}
        if result.services:
            fields["service"] = result.services[0]
        if result.symptoms:
            fields["symptom"] = result.symptoms[0]
        if result.business_domain:
            fields["business_domain"] = result.business_domain
        return fields
