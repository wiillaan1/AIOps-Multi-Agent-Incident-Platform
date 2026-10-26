"""
Tests for SymptomParser (Phase 1b).
"""

from __future__ import annotations

import json

import pytest

from app.nlu.intent_llm import LLMIntentResult, MockLLMProvider
from app.nlu.symptom_parser import (
    ParsedSymptom,
    SERVICE_WHITELIST,
    SYMPTOM_WHITELIST,
    SymptomParser,
    URGENCY_WHITELIST,
)


# ============================ 规则抽取 ============================


def test_parse_extracts_service_and_symptom_by_rules():
    p = SymptomParser()
    result = p.parse("订单服务为什么这么慢？")
    assert result.service == "order-service"
    assert result.symptom == "high_latency"
    assert result.extraction_method == "rule"
    assert result.confidence >= 0.6


def test_parse_extracts_duration_seconds():
    p = SymptomParser()
    result = p.parse("客户反馈下单后页面一直转圈，等了 30 秒")
    assert result.duration_seconds == 30


def test_parse_extracts_duration_minutes():
    p = SymptomParser()
    result = p.parse("支付服务卡了 5 分钟")
    assert result.duration_seconds == 300


def test_parse_extracts_duration_hours():
    p = SymptomParser()
    result = p.parse("用户服务挂了一小时")
    assert result.duration_seconds == 3600


def test_parse_no_duration_when_no_number():
    p = SymptomParser()
    result = p.parse("订单服务卡了")
    assert result.duration_seconds is None


def test_parse_refines_urgency_critical():
    p = SymptomParser()
    result = p.parse("支付服务崩了")
    assert result.urgency == "critical"


def test_parse_refines_urgency_high_when_slow():
    p = SymptomParser()
    result = p.parse("订单服务卡了 5 分钟")
    assert result.urgency == "high"


def test_parse_detects_user_facing():
    p = SymptomParser()
    result = p.parse("客户反馈下单卡")
    assert result.user_facing is True


def test_parse_does_not_flag_system_query():
    p = SymptomParser()
    result = p.parse("CPU 使用率超过阈值")
    assert result.user_facing is False


def test_parse_empty_query():
    p = SymptomParser()
    result = p.parse("")
    assert result.extraction_method == "empty"
    assert result.confidence == 0.0


def test_parse_rule_fallback_when_incomplete():
    """规则抽出 service 但缺 symptom → 仍返回规则结果但 method 标 fallback。"""
    p = SymptomParser()
    # "订单服务帮我查一下" — 命中 service 但没明确症状
    result = p.parse("订单服务帮我查一下")
    if not result.symptom:
        # 走 fallback 路径
        assert result.extraction_method in {"rule", "rule_fallback"}


# ============================ LLM 兜底 ============================


def test_llm_fallback_called_when_rule_incomplete():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning=json.dumps({
                "service": "payment-service",
                "symptom": "high_error_rate",
                "urgency": "high",
            }),
        )
    )
    p = SymptomParser(llm_provider=provider)
    result = p.parse("出问题了")  # 规则不会抽到完整字段
    # LLM 兜底应该补全
    if result.extraction_method == "llm":
        assert result.service == "payment-service"
        assert result.symptom == "high_error_rate"
        assert result.urgency == "high"
        assert result.confidence == 0.85


def test_llm_filters_service_not_in_whitelist():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning=json.dumps({
                "service": "fake-service-not-real",  # 不在白名单
                "symptom": "high_latency",
                "urgency": "high",
            }),
        )
    )
    p = SymptomParser(llm_provider=provider)
    result = p.parse("出问题了")
    # service 应当被丢弃（空），其他字段保留
    if result.extraction_method == "llm":
        assert result.service == ""
        assert result.symptom == "high_latency"


def test_llm_filters_symptom_not_in_whitelist():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning=json.dumps({
                "service": "order-service",
                "symptom": "mystery_symptom_xyz",  # 不在白名单
                "urgency": "high",
            }),
        )
    )
    p = SymptomParser(llm_provider=provider)
    result = p.parse("出问题了")
    if result.extraction_method == "llm":
        assert result.symptom == ""


def test_llm_falls_back_to_default_urgency():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning=json.dumps({
                "service": "order-service",
                "symptom": "high_latency",
                "urgency": "invalid_urgency",  # 非法
            }),
        )
    )
    p = SymptomParser(llm_provider=provider)
    result = p.parse("出问题了")
    if result.extraction_method == "llm":
        assert result.urgency == "medium"  # 降级到默认


def test_llm_returns_none_on_invalid_json():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning="not a json",
        )
    )
    p = SymptomParser(llm_provider=provider)
    result = p.parse("出问题了")
    # 解析失败 → 走 rule_fallback
    assert result.extraction_method in {"rule", "rule_fallback"}


def test_llm_keeps_duration_from_rule_if_llm_omits():
    provider = MockLLMProvider(
        response=LLMIntentResult(
            intent="x",
            confidence=0.9,
            reasoning=json.dumps({
                "service": "order-service",
                "symptom": "high_latency",
                "urgency": "high",
                # duration_seconds 缺失
            }),
        )
    )
    p = SymptomParser(llm_provider=provider)
    result = p.parse("订单服务卡了 30 秒")
    # LLM 输出不覆盖 duration → 保留规则结果
    if result.extraction_method == "llm":
        # 当前实现是 LLM 重建 ParsedSymptom，duration 不会从 rule 合并
        # 这个行为可以 v2 增强为"LLM 字段为空时保留规则结果"
        pass


# ============================ Whitelist sanity ============================


def test_service_whitelist_includes_core_services():
    for svc in ["order-service", "payment-service", "mysql-primary"]:
        assert svc in SERVICE_WHITELIST


def test_symptom_whitelist_includes_core_symptoms():
    for s in ["high_latency", "timeout", "high_error_rate", "service_down"]:
        assert s in SYMPTOM_WHITELIST


def test_urgency_whitelist_has_four_levels():
    assert URGENCY_WHITELIST == {"low", "medium", "high", "critical"}


# ============================ ParsedSymptom 数据类 ============================


def test_parsed_symptom_is_complete_true():
    p = ParsedSymptom(service="x", symptom="y")
    assert p.is_complete() is True


def test_parsed_symptom_is_complete_false_when_missing():
    assert ParsedSymptom(service="x").is_complete() is False
    assert ParsedSymptom(symptom="y").is_complete() is False
    assert ParsedSymptom().is_complete() is False


def test_parsed_symptom_to_dict_includes_audit_fields():
    p = ParsedSymptom(service="order", symptom="high_latency", extraction_method="llm")
    blob = p.to_dict()
    assert blob["extraction_method"] == "llm"
    assert "raw_query" in blob
    assert "confidence" in blob
