"""
证据四分类 + 根因门禁

修复审查报告（中期补 #6）：
- 把每条证据按"强度"分四类：STRONG / WEAK / REFUTE / INSUFFICIENT
- 反事实前置打分（基于已有 _check_causality.counterfactual_score）
- 根因门禁：没有 STRONG 证据 + 至少有 1 条 REFUTE → 强制降级为 INSUFFICIENT
- INSUFFICIENT 状态：promoted_confidence=0, allowed_to_heal=False

调用方：RCAAgent._synthesize_analysis 末尾用 apply_root_cause_gate() 覆盖
返回的置信度；orchestrator 端用 GateResult.allowed_to_heal 决定是否走 Heal。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class EvidenceStrength(str, enum.Enum):
    """证据强度四分类。"""

    STRONG = "strong"            # 因果校验通过 + 反事实 score ≥ 0.7
    WEAK = "weak"                # 仅时间相关或仅拓扑相关，因果未通过
    REFUTE = "refute"            # 健康对照差异 ≤ 5% 或反事实 score ≤ 0.2（说明不是根因）
    INSUFFICIENT = "insufficient"  # 没有可用证据


# 反事实分数阈值（与 rca_agent._check_causality 配合使用）
COUNTERFACTUAL_STRONG_THRESHOLD = 0.7
COUNTERFACTUAL_REFUTE_THRESHOLD = 0.2
HEALTHY_DIFF_REFUTE_THRESHOLD = 0.05  # 健康对照差异 ≤ 5% 视为反证


@dataclass
class EvidenceItem:
    """单条证据输入（与 RCA 各源兼容的最小化表示）。"""

    source: str  # "bayesian" / "rag" / "memory" / "log" / "topology" / "change"
    target: str  # 候选根因（如 "order-service.cpu_high"）
    score: float = 0.0           # 源得分（0-1）
    causality: dict[str, Any] | None = None  # _check_causality 返回的 dict
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateResult:
    """根因门禁结果。

    Attributes:
        strength: 最终证据强度（取所有证据项的最弱门控分类）
        promoted_confidence: 经过门禁调整后的置信度（0-1）
        allowed_to_heal: 是否允许进入自愈链路
        reason: 决策原因（用于审计/可解释性）
        breakdown: 各分类证据条数（strong/weak/refute/insufficient）
    """

    strength: EvidenceStrength
    promoted_confidence: float
    allowed_to_heal: bool
    reason: str
    breakdown: dict[str, int] = field(default_factory=dict)


def classify_evidence(item: EvidenceItem) -> EvidenceStrength:
    """把单条证据按四分类规则分类。

    规则优先级（从强到弱）：
    1. REFUTE：健康对照差异 ≤ 5% 或反事实 score ≤ 0.2 → 强制反证
    2. STRONG：反事实 score ≥ 0.7 且因果校验通过 → 强证据
    3. WEAK：源得分 ≥ 0.3 但不满足 STRONG → 弱证据
    4. INSUFFICIENT：源得分 < 0.3 且无因果信息
    """
    causality = item.causality or {}

    # 1. 反证优先：健康对照差异小 + 反事实 score 低 = 反证
    counterfactual = causality.get("counterfactual_score")
    healthy_diff = causality.get("healthy_baseline_diff")
    if (
        counterfactual is not None
        and counterfactual <= COUNTERFACTUAL_REFUTE_THRESHOLD
    ) or (
        healthy_diff is not None
        and healthy_diff <= HEALTHY_DIFF_REFUTE_THRESHOLD
    ):
        return EvidenceStrength.REFUTE

    # 2. STRONG：反事实 score ≥ 0.7 + 因果校验通过（score 太低 0.1 视为噪声不算）
    is_causal = causality.get("is_causal", False)
    if (
        counterfactual is not None
        and counterfactual >= COUNTERFACTUAL_STRONG_THRESHOLD
        and is_causal
        and item.score >= 0.1
    ):
        return EvidenceStrength.STRONG

    # 3. WEAK：源得分 ≥ 0.3
    if item.score >= 0.3:
        return EvidenceStrength.WEAK

    # 4. INSUFFICIENT
    return EvidenceStrength.INSUFFICIENT


def apply_root_cause_gate(
    evidence_items: list[EvidenceItem],
    raw_confidence: float = 0.0,
) -> GateResult:
    """根因门禁。

    规则：
    - 任何 1 条 REFUTE 出现 → 强制降级为 INSUFFICIENT（不让 heal 瞎执行）
    - 至少 1 条 STRONG OR (≥3 条 WEAK) → 通过门禁，allowed_to_heal=True
    - 否则 INSUFFICIENT，promoted_confidence=0，allowed_to_heal=False
    """
    if not evidence_items:
        return GateResult(
            strength=EvidenceStrength.INSUFFICIENT,
            promoted_confidence=0.0,
            allowed_to_heal=False,
            reason="no_evidence",
            breakdown={"strong": 0, "weak": 0, "refute": 0, "insufficient": 0},
        )

    classifications = [classify_evidence(item) for item in evidence_items]
    breakdown = {
        "strong": classifications.count(EvidenceStrength.STRONG),
        "weak": classifications.count(EvidenceStrength.WEAK),
        "refute": classifications.count(EvidenceStrength.REFUTE),
        "insufficient": classifications.count(EvidenceStrength.INSUFFICIENT),
    }

    # 任何 REFUTE → 强制降级
    if breakdown["refute"] > 0:
        return GateResult(
            strength=EvidenceStrength.INSUFFICIENT,
            promoted_confidence=0.0,
            allowed_to_heal=False,
            reason=f"refute_present ({breakdown['refute']} refuting evidence)",
            breakdown=breakdown,
        )

    # 至少 1 STRONG 或 ≥3 WEAK → 通过
    if breakdown["strong"] >= 1 or breakdown["weak"] >= 3:
        # 通过门禁：promoted_confidence = raw * min(1, strong + weak*0.3)
        strength_factor = min(1.0, breakdown["strong"] + breakdown["weak"] * 0.3)
        promoted = round(min(1.0, raw_confidence * strength_factor), 4)
        return GateResult(
            strength=EvidenceStrength.STRONG if breakdown["strong"] >= 1 else EvidenceStrength.WEAK,
            promoted_confidence=promoted,
            allowed_to_heal=True,
            reason=f"gate_passed (strong={breakdown['strong']}, weak={breakdown['weak']})",
            breakdown=breakdown,
        )

    # 否则 INSUFFICIENT
    return GateResult(
        strength=EvidenceStrength.INSUFFICIENT,
        promoted_confidence=0.0,
        allowed_to_heal=False,
        reason=f"insufficient ({breakdown['weak']} weak, {breakdown['strong']} strong)",
        breakdown=breakdown,
    )
