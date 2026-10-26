"""
AIOps Agent Platform - Reflection Layer

让 Orchestrator / RCA / Heal / Intent 四个子系统具备"反思"能力：
不是把决策权交给 LLM，而是结构化地枚举反思分支、记录反思原因、
控制重试次数。每次反思都写入 `incident.context["audit_trail"]`
便于事后回溯。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ===== 反思原因枚举（防止 LLM 自由发挥）=====

REFLECTION_REASONS = {
    "low_confidence",            # 置信度低
    "incomplete_recovery",       # 症状恢复不完整
    "no_match",                  # 匹配不到
    "tool_failure",              # 工具调用失败
    "inconsistent_evidence",     # 证据不一致
    "missing_info",              # 信息缺失
    "max_retry_exceeded",        # 重试超限
    "all_candidates_too_risky",  # 所有候选风险过高
}

# 永久错误 — 不重试
PERMANENT_ERRORS = {
    "permission_denied",
    "invalid_input",
    "service_not_found",
}


# ===== 反思数据结构 =====


@dataclass
class ReflectionResult:
    """反思决策的输出。"""

    next_action: str  # "retry" / "escalate" / "complete" / "broaden_search" / "clarify"
    reason: str       # REFLECTION_REASONS 中的一个
    reason_detail: str = ""
    confidence: float = 0.0
    suggestions: list[str] = field(default_factory=list)  # 给下一轮的具体建议
    duration_ms: int = 0


@dataclass
class VerifyEvidence:
    """verify 节点的证据输入。"""

    confidence: float = 0.0          # 自愈/RCA 综合置信度
    symptoms_resolved: int = 0       # 已恢复的症状数
    symptoms_total: int = 0          # 总症状数
    tool_failures: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ===== Verify 反思 =====


def reflect_on_verify(
    evidence: VerifyEvidence,
    retry_count: int = 0,
    max_retry: int = 2,
) -> ReflectionResult:
    """
    verify 节点的反思：
    - 置信度低 → escalate
    - 症状恢复比例 < 50% → retry
    - 工具持续失败 → escalate
    - 重试超限 → escalate
    - 否则 → complete

    Args:
        evidence: verify 阶段收集的证据
        retry_count: 已经重试的次数
        max_retry: 最大重试次数（默认 2）

    Returns:
        ReflectionResult: 决策与原因
    """
    started = time.time()

    # 规则 1: 工具持续失败 → 升级
    if len(evidence.tool_failures) >= 3:
        return ReflectionResult(
            next_action="escalate",
            reason="tool_failure",
            reason_detail=f"已累计 {len(evidence.tool_failures)} 次工具失败",
            confidence=0.0,
            suggestions=["建议人工介入检查工具链可达性"],
            duration_ms=_elapsed(started),
        )

    # 规则 2: 重试超限 → 升级
    if retry_count >= max_retry:
        return ReflectionResult(
            next_action="escalate",
            reason="max_retry_exceeded",
            reason_detail=f"已重试 {retry_count}/{max_retry} 仍未达标",
            confidence=evidence.confidence,
            suggestions=["自动恢复失败，转人工处置"],
            duration_ms=_elapsed(started),
        )

    # 规则 3: 症状恢复不完整 → 重试
    if evidence.symptoms_total > 0:
        recovery_ratio = evidence.symptoms_resolved / evidence.symptoms_total
        if recovery_ratio < 0.5:
            return ReflectionResult(
                next_action="retry",
                reason="incomplete_recovery",
                reason_detail=(
                    f"症状恢复比例 {recovery_ratio:.0%} "
                    f"({evidence.symptoms_resolved}/{evidence.symptoms_total}) 不足 50%"
                ),
                confidence=evidence.confidence,
                suggestions=[
                    "检查自愈动作是否真的执行",
                    "考虑放宽到不同 severity 的 playbook",
                ],
                duration_ms=_elapsed(started),
            )

    # 规则 4: 置信度不足 → 升级（不重试以免死循环）
    if evidence.confidence < 0.6:
        return ReflectionResult(
            next_action="escalate",
            reason="low_confidence",
            reason_detail=f"综合置信度 {evidence.confidence:.0%} 低于 60%",
            confidence=evidence.confidence,
            suggestions=["人工确认根因后再决策"],
            duration_ms=_elapsed(started),
        )

    # 通过
    return ReflectionResult(
        next_action="complete",
        reason="low_confidence",  # 此处实际是"通过"
        reason_detail="置信度与症状恢复均达标",
        confidence=evidence.confidence,
        duration_ms=_elapsed(started),
    )


# ===== 工具调用反思 =====


def reflect_on_tool_failure(
    tool_name: str,
    error: str,
    attempt: int,
    max_attempts: int = 3,
) -> ReflectionResult:
    """
    工具调用失败时的反思：
    - 永久错误 → 不重试
    - 临时错误 → 重试
    - 重试超限 → 升级
    """
    started = time.time()
    permanent = any(perm in error.lower() for perm in PERMANENT_ERRORS)

    if permanent:
        return ReflectionResult(
            next_action="escalate",
            reason="tool_failure",
            reason_detail=f"工具 {tool_name} 返回永久错误: {error[:200]}",
            suggestions=["检查 RBAC / 输入参数 / 服务可达性"],
            duration_ms=_elapsed(started),
        )

    if attempt >= max_attempts:
        return ReflectionResult(
            next_action="escalate",
            reason="max_retry_exceeded",
            reason_detail=f"工具 {tool_name} 已重试 {attempt}/{max_attempts}",
            suggestions=["切换到 fallback 数据源或人工查询"],
            duration_ms=_elapsed(started),
        )

    return ReflectionResult(
        next_action="retry",
        reason="tool_failure",
        reason_detail=f"工具 {tool_name} 第 {attempt + 1} 次尝试",
        suggestions=[
            "检查参数是否完整",
            "如果是网络超时，增大 timeout",
            "如果是临时错误，自动重试",
        ],
        duration_ms=_elapsed(started),
    )


# ===== 信息缺失反思（用于 Intent / RCA 上下文补全）=====


def reflect_missing_info(
    intent: str,
    parsed_fields: dict[str, Any],
    required_fields: list[str],
) -> list[str]:
    """
    反思：基于意图与已解析字段，生成追问问题。

    Args:
        intent: 已识别的意图
        parsed_fields: 已解析出的字段 {service, symptom, time_window, ...}
        required_fields: 该意图必需字段

    Returns:
        list[str]: 追问问题列表（空表示无需追问）
    """
    questions: list[str] = []
    missing = [f for f in required_fields if not parsed_fields.get(f)]
    if not missing:
        return questions

    prompts_by_field = {
        "service": "请问是哪个服务出现了问题？例如：订单、支付、登录、网关",
        "symptom": "具体表现是什么？是慢、报错、还是完全不可用？",
        "time_window": "问题是从什么时候开始的？最近 1 小时还是今天？",
        "alert_id": "您希望分析哪条告警？",
        "environment": "是生产环境还是测试环境？",
    }
    for field in missing:
        if field in prompts_by_field:
            questions.append(prompts_by_field[field])
    return questions


# ===== 工具 =====


def _elapsed(started: float) -> int:
    return int((time.time() - started) * 1000)


def append_audit_trail(
    incident_context: dict[str, Any],
    step: str,
    trigger: str,
    actions_taken: list[str],
    outcome: str,
    duration_ms: int = 0,
) -> None:
    """把每次反思写入 incident.context.audit_trail，便于事后回溯。"""
    trail = incident_context.setdefault("audit_trail", [])
    trail.append({
        "step": step,
        "trigger": trigger,
        "actions_taken": actions_taken,
        "outcome": outcome,
        "duration_ms": duration_ms,
        "timestamp": time.time(),
    })
    # 防止无限增长
    if len(trail) > 100:
        incident_context["audit_trail"] = trail[-100:]
