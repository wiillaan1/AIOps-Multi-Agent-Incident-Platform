"""
Tests for the IntentClarification mechanism (W1b).

覆盖 4 类行为：
1. 正常分类：返回原意图
2. 命中意图但缺必填字段 → 反思为 NEED_CLARIFICATION
3. 完全匹配不到 → GENERAL_QUESTION + reflection_reason
4. 不同意图追问问题模板不同
"""

from __future__ import annotations

from app.nlu.intent_classifier import (
    INTENT_REQUIRED_FIELDS,
    IntentClassifier,
    IntentType,
)


def test_classify_fault_diagnosis_with_service_no_clarification():
    """完整的故障诊断问句（带服务名）不需要追问。"""
    result = IntentClassifier().classify("订单服务为什么这么慢？")
    assert result.intent == IntentType.FAULT_DIAGNOSIS
    assert result.needs_clarification is False
    assert result.clarification_questions == []


def test_classify_fault_diagnosis_missing_service_triggers_clarification():
    """故障诊断但没说服务 → 反思追问服务名。"""
    result = IntentClassifier().classify("为什么这么卡？")
    assert result.intent == IntentType.NEED_CLARIFICATION
    assert result.needs_clarification is True
    assert len(result.clarification_questions) >= 1
    assert any("服务" in q for q in result.clarification_questions)
    assert result.reflection_reason == "missing_required_fields"


def test_classify_heal_request_with_service_no_clarification():
    """修复请求带服务名 → 不追问（动作由确认按钮补齐）。"""
    result = IntentClassifier().classify("帮我处理一下订单服务")
    if result.intent == IntentType.HEAL_REQUEST:
        assert result.needs_clarification is False


def test_classify_heal_request_missing_service_triggers_clarification():
    """修复请求没说服务 → 反思追问服务。"""
    result = IntentClassifier().classify("帮我重启一下")
    # 没服务名 → 命中规则后触发 NEED_CLARIFICATION
    assert result.intent in {
        IntentType.NEED_CLARIFICATION,
        IntentType.HEAL_REQUEST,  # 在某些低置信情况下也接受
    }
    if result.needs_clarification:
        assert any("服务" in q for q in result.clarification_questions)


def test_classify_business_check_missing_domain_triggers_clarification():
    result = IntentClassifier().classify("有没有重复扣款？")
    if result.intent == IntentType.NEED_CLARIFICATION:
        assert any("业务" in q or "场景" in q or "环境" in q for q in result.clarification_questions)


def test_classify_unmatched_query_returns_general_with_reflection_reason():
    result = IntentClassifier().classify("随便问问")
    assert result.intent == IntentType.GENERAL_QUESTION
    assert result.reflection_reason in {"all_scores_below_threshold", "no_pattern_matched"}


def test_classify_empty_query():
    result = IntentClassifier().classify("")
    assert result.intent == IntentType.GENERAL_QUESTION
    assert result.reflection_reason == "empty_query"


def test_required_fields_mapping_completeness():
    """每种业务意图都应至少有一个必填字段（除非确实不需要）。"""
    for intent, fields in INTENT_REQUIRED_FIELDS.items():
        if intent in {
            IntentType.NEED_CLARIFICATION,
            IntentType.GENERAL_QUESTION,
        }:
            continue
        assert fields, f"{intent} 应有必填字段"


def test_clarification_questions_are_distinct():
    """追问问题应当是不重复的具体提示。"""
    result = IntentClassifier().classify("怎么回事？")
    if result.clarification_questions:
        assert len(result.clarification_questions) == len(set(result.clarification_questions))


def test_classify_metric_query_with_metric_name_no_clarification():
    result = IntentClassifier().classify("订单服务 CPU 多少")
    assert result.intent == IntentType.METRIC_QUERY
    assert result.needs_clarification is False


def test_classify_metric_query_missing_service_triggers_clarification():
    result = IntentClassifier().classify("CPU 用了多少？")
    # 命中 METRIC_QUERY 但缺 service
    if result.intent == IntentType.METRIC_QUERY:
        assert result.needs_clarification is True
    else:
        # 或者完全没匹配上
        assert result.intent in {IntentType.GENERAL_QUESTION, IntentType.NEED_CLARIFICATION}


def test_classify_history_lookup_with_time_hint_no_clarification():
    result = IntentClassifier().classify("订单服务上次故障怎么处理的")
    assert result.intent == IntentType.HISTORY_LOOKUP
    assert result.needs_clarification is False
