"""
LLM Fallback - LLM 不可用时的规则引擎降级

设计原则（Zenjoy AWS Agent "程序计算，AI 总结"）：
- LLM 失败/熔断/key 缺失时，自动降级到规则引擎
- 输出格式尽量贴近 LLM 的 schema，方便上层无感切换
- 降级路径必须记录到审计（reason 字段）

支持的降级场景：
1. 告警摘要（summarize_alert）
2. RCA 根因（rca_root_cause）
3. 变更风险叙述（change_risk_narrative）
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

from app.core.llm_client import LLMUnavailable, LLMCallResult


logger = logging.getLogger(__name__)


@dataclass
class AlertSummary:
    """告警摘要的统一 schema"""
    headline: str
    severity: str
    affected_service: str
    metric: str
    trigger_value: float | None
    threshold: float | None
    deviation_pct: float | None
    notes: list[str]


def summarize_alert(alert: dict[str, Any]) -> AlertSummary:
    """
    规则引擎版告警摘要（无 LLM）。

    输入 alert 字段约定：
        service / metric_name / severity / value / threshold / window
    """
    service = alert.get("service", "unknown")
    metric = alert.get("metric_name", alert.get("metric", "unknown"))
    severity = alert.get("severity", "unknown")
    value = alert.get("value")
    threshold = alert.get("threshold")
    notes: list[str] = []

    deviation_pct = None
    if value is not None and threshold not in (None, 0):
        deviation_pct = round((float(value) - float(threshold)) / float(threshold) * 100, 1)
        if deviation_pct > 0:
            notes.append(f"超出阈值 {deviation_pct}%")

    if severity == "critical":
        headline = f"🚨 {service} {metric} 严重告警"
    elif severity == "high":
        headline = f"⚠️  {service} {metric} 高优先级告警"
    elif severity == "medium":
        headline = f"📊 {service} {metric} 中等告警"
    else:
        headline = f"ℹ️  {service} {metric} 通知"

    if value is not None:
        headline += f"（当前值 {value}）"

    return AlertSummary(
        headline=headline,
        severity=severity,
        affected_service=service,
        metric=metric,
        trigger_value=float(value) if value is not None else None,
        threshold=float(threshold) if threshold is not None else None,
        deviation_pct=deviation_pct,
        notes=notes or ["规则引擎摘要（LLM 未启用）"],
    )


def rca_root_cause(alert: dict[str, Any], evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    规则引擎版根因推断。

    evidence 字段（可选）：
        history_similar: list of {id, root_cause, similarity}
        bayesian_top: {hypothesis, posterior, likelihoods}
        rag_docs: list of {id, content, score}

    返回 schema：
        {
            "root_cause": str,
            "confidence": float,
            "evidence_refs": list[str],
            "method": "rules_engine",
            "needs_human_review": bool,
        }
    """
    ev = evidence or {}
    history = ev.get("history_similar") or []
    bayesian = ev.get("bayesian_top") or {}

    # 优先级 1: 贝叶斯后验概率
    if bayesian:
        hyp = bayesian.get("hypothesis", "unknown")
        posterior = float(bayesian.get("posterior", 0.0))
        return {
            "root_cause": hyp,
            "confidence": round(posterior, 3),
            "evidence_refs": [f"bayesian:{hyp}"],
            "method": "rules_engine+bayesian",
            "needs_human_review": posterior < 0.6,
        }

    # 优先级 2: 历史相似事故
    if history:
        top = max(history, key=lambda h: h.get("similarity", 0))
        return {
            "root_cause": top.get("root_cause", "unknown"),
            "confidence": round(float(top.get("similarity", 0.0)) * 0.8, 3),  # 折扣
            "evidence_refs": [f"history:{top.get('id', '?')}"],
            "method": "rules_engine+history",
            "needs_human_review": True,
        }

    # 优先级 3: 启发式（基于 metric 类型）
    metric = (alert.get("metric_name") or alert.get("metric") or "").lower()
    if "cpu" in metric:
        guess, conf = "CPU resource saturation", 0.4
    elif "memory" in metric or "mem" in metric:
        guess, conf = "Memory leak / resource exhaustion", 0.4
    elif "latency" in metric or "p99" in metric:
        guess, conf = "Slow downstream dependency / DB", 0.4
    elif "error" in metric:
        guess, conf = "Application exception spike", 0.35
    elif "pool" in metric or "connection" in metric:
        guess, conf = "Connection pool exhaustion", 0.4
    else:
        guess, conf = "Unknown pattern (insufficient evidence)", 0.2

    return {
        "root_cause": guess,
        "confidence": conf,
        "evidence_refs": [],
        "method": "rules_engine+heuristic",
        "needs_human_review": True,
    }


def change_risk_narrative(
    risk_score: float,
    factor_scores: dict[str, float],
    service: str,
    blast_radius: str = "single",
) -> str:
    """
    规则引擎版变更风险叙述。

    risk_score: 0-1
    factor_scores: {"impact": 0.3, "complexity": 0.2, ...}
    """
    if risk_score >= 0.7:
        level = "高风险"
    elif risk_score >= 0.4:
        level = "中风险"
    else:
        level = "低风险"

    top_factor = max(factor_scores.items(), key=lambda kv: kv[1]) if factor_scores else (None, 0)

    narrative = (
        f"{service} 变更评估为{level}（综合评分 {risk_score:.2f}）。"
    )
    if blast_radius == "multi":
        narrative += "⚠️ 影响范围跨多个服务，需逐项确认。"
    if top_factor[0]:
        narrative += f"主要风险因素：{top_factor[0]}（{top_factor[1]:.2f}）。"

    narrative += "（规则引擎生成，LLM 未启用）"
    return narrative


# ---------- 统一封装：自动 fallback ----------
async def safe_chat(
    client,
    messages: list[Any],
    fallback_result: LLMCallResult | None = None,
    **kwargs,
) -> LLMCallResult:
    """
    带自动降级的 LLM 调用封装。

    流程：
      1. 检查 client.is_available() → False 则直接返回 fallback_result
      2. 调用 client.chat(messages) → 失败则返回 fallback_result（标记 via_fallback=True）
      3. 成功则返回 LLM 真实结果

    用法：
        result = await safe_chat(
            client,
            messages=[{"role": "user", "content": "..."}],
            fallback_result=LLMCallResult(
                content="规则版摘要...",
                model="rules-engine",
                provider="fallback",
                latency_ms=0,
            ),
        )
    """
    if not client.is_available():
        reason = client.availability_reason()
        logger.warning("[safe_chat] LLM 不可用，降级到规则引擎: %s", reason)
        if fallback_result is None:
            raise LLMUnavailable(reason)
        return LLMCallResult(
            content=fallback_result.content,
            model=fallback_result.model,
            provider="fallback",
            latency_ms=0,
            via_fallback=True,
            fallback_reason=reason,
        )

    try:
        return await client.chat(messages, **kwargs)
    except LLMUnavailable as e:
        logger.warning("[safe_chat] LLM 调用失败，降级: %s", e)
        if fallback_result is None:
            raise
        return LLMCallResult(
            content=fallback_result.content,
            model=fallback_result.model,
            provider="fallback",
            latency_ms=0,
            via_fallback=True,
            fallback_reason=str(e),
        )
