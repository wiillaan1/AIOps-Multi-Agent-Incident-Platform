"""
测试 evidence_classifier + RCAAgent._apply_root_cause_gate

覆盖：
- 四分类边界（STRONG / WEAK / REFUTE / INSUFFICIENT）
- 门禁规则（REFUTE 强制降级、1 STRONG 通过、3 WEAK 通过、<3 WEAK 不通过）
- RCA agent 集成（_apply_root_cause_gate 返回正确 GateResult）
"""

from __future__ import annotations

from app.agents.evidence_classifier import (
    COUNTERFACTUAL_REFUTE_THRESHOLD,
    COUNTERFACTUAL_STRONG_THRESHOLD,
    HEALTHY_DIFF_REFUTE_THRESHOLD,
    EvidenceItem,
    EvidenceStrength,
    GateResult,
    apply_root_cause_gate,
    classify_evidence,
)
from app.agents.rca_agent import RCAAgent


# ==================== classify_evidence 边界测试 ====================


def test_strong_when_counterfactual_high_and_causal():
    """STRONG：counterfactual >= 0.7 + is_causal=True"""
    item = EvidenceItem(
        source="bayesian",
        target="order.cpu",
        score=0.9,
        causality={
            "is_causal": True,
            "counterfactual_score": COUNTERFACTUAL_STRONG_THRESHOLD,
            "healthy_baseline_diff": 0.3,
        },
    )
    assert classify_evidence(item) == EvidenceStrength.STRONG


def test_weak_when_score_high_but_causal_not_passed():
    """WEAK：score >= 0.3 但因果未通过"""
    item = EvidenceItem(
        source="rag",
        target="order.cpu",
        score=0.6,
        causality={
            "is_causal": False,
            "counterfactual_score": 0.4,
            "healthy_baseline_diff": 0.2,
        },
    )
    assert classify_evidence(item) == EvidenceStrength.WEAK


def test_refute_when_counterfactual_very_low():
    """REFUTE：counterfactual <= 0.2"""
    item = EvidenceItem(
        source="log",
        target="order.cpu",
        score=0.5,
        causality={
            "is_causal": False,
            "counterfactual_score": COUNTERFACTUAL_REFUTE_THRESHOLD - 0.05,
            "healthy_baseline_diff": 0.4,
        },
    )
    assert classify_evidence(item) == EvidenceStrength.REFUTE


def test_refute_when_healthy_diff_very_small():
    """REFUTE：healthy_baseline_diff <= 5%"""
    item = EvidenceItem(
        source="topology",
        target="order.cpu",
        score=0.8,
        causality={
            "is_causal": False,
            "counterfactual_score": 0.6,
            "healthy_baseline_diff": HEALTHY_DIFF_REFUTE_THRESHOLD - 0.01,
        },
    )
    assert classify_evidence(item) == EvidenceStrength.REFUTE


def test_insufficient_when_score_low_and_no_causality():
    """INSUFFICIENT：score < 0.3 且无 causality"""
    item = EvidenceItem(
        source="rag",
        target="order.cpu",
        score=0.1,
        causality=None,
    )
    assert classify_evidence(item) == EvidenceStrength.INSUFFICIENT


def test_insufficient_when_score_low_with_causality():
    """INSUFFICIENT：score < 0.1（噪声阈值）且有 causality 仍算不足"""
    item = EvidenceItem(
        source="rag",
        target="order.cpu",
        score=0.05,
        causality={
            "is_causal": True,
            "counterfactual_score": 0.8,
            "healthy_baseline_diff": 0.3,
        },
    )
    # score 0.05 < 噪声阈值 0.1，即便因果通过也归 INSUFFICIENT
    assert classify_evidence(item) == EvidenceStrength.INSUFFICIENT


def test_refute_priority_over_strong():
    """REFUTE 优先级最高：即便其他条件满足，只要 refuting 就 REFUTE"""
    item = EvidenceItem(
        source="memory",
        target="order.cpu",
        score=0.95,
        causality={
            "is_causal": True,
            "counterfactual_score": 0.95,
            "healthy_baseline_diff": 0.01,  # 健康对照反证
        },
    )
    assert classify_evidence(item) == EvidenceStrength.REFUTE


def test_causality_none_falls_through_to_score():
    """causality=None 时只按 score 分类"""
    assert classify_evidence(EvidenceItem("a", "x", 0.8)) == EvidenceStrength.WEAK
    assert classify_evidence(EvidenceItem("a", "x", 0.1)) == EvidenceStrength.INSUFFICIENT


# ==================== apply_root_cause_gate 门禁测试 ====================


def test_gate_empty_evidence():
    """空证据 → INSUFFICIENT"""
    g = apply_root_cause_gate([], raw_confidence=0.8)
    assert g.strength == EvidenceStrength.INSUFFICIENT
    assert g.promoted_confidence == 0.0
    assert g.allowed_to_heal is False
    assert "no_evidence" in g.reason


def test_gate_one_strong_passes():
    """1 条 STRONG → 通过门禁"""
    items = [
        EvidenceItem(
            "bayesian", "x.cpu", 0.9,
            causality={"is_causal": True, "counterfactual_score": 0.8},
        ),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.7)
    assert g.strength == EvidenceStrength.STRONG
    assert g.allowed_to_heal is True
    assert g.promoted_confidence > 0


def test_gate_three_weak_passes():
    """3 条 WEAK → 通过门禁"""
    items = [
        EvidenceItem("rag", "x.cpu", 0.5),
        EvidenceItem("memory", "x.cpu", 0.4),
        EvidenceItem("log", "x.cpu", 0.6),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.6)
    assert g.strength == EvidenceStrength.WEAK
    assert g.allowed_to_heal is True
    assert g.breakdown["weak"] == 3
    assert g.breakdown["strong"] == 0


def test_gate_two_weak_insufficient():
    """2 条 WEAK（不够 3 条）→ INSUFFICIENT"""
    items = [
        EvidenceItem("rag", "x.cpu", 0.5),
        EvidenceItem("memory", "x.cpu", 0.4),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.6)
    assert g.strength == EvidenceStrength.INSUFFICIENT
    assert g.allowed_to_heal is False
    assert g.promoted_confidence == 0.0


def test_gate_refute_forces_downgrade():
    """任何 1 条 REFUTE → 强制 INSUFFICIENT，即便有 STRONG"""
    items = [
        EvidenceItem(  # STRONG
            "bayesian", "x.cpu", 0.9,
            causality={"is_causal": True, "counterfactual_score": 0.9},
        ),
        EvidenceItem(  # REFUTE
            "topology", "x.cpu", 0.7,
            causality={"counterfactual_score": 0.1},
        ),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.8)
    assert g.strength == EvidenceStrength.INSUFFICIENT
    assert g.allowed_to_heal is False
    assert g.promoted_confidence == 0.0
    assert "refute_present" in g.reason
    assert g.breakdown["refute"] == 1
    assert g.breakdown["strong"] == 1  # STRONG 仍然计入


def test_gate_promoted_confidence_capped_at_one():
    """promoted_confidence 不会超过 1.0"""
    items = [
        EvidenceItem("a", "x", 0.9, causality={"is_causal": True, "counterfactual_score": 0.95}),
        EvidenceItem("b", "x", 0.9, causality={"is_causal": True, "counterfactual_score": 0.95}),
        EvidenceItem("c", "x", 0.9, causality={"is_causal": True, "counterfactual_score": 0.95}),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.95)
    assert g.promoted_confidence <= 1.0


def test_gate_breakdown_counts_correctly():
    """breakdown 统计各类证据条数"""
    items = [
        EvidenceItem("a", "x", 0.9, causality={"is_causal": True, "counterfactual_score": 0.9}),
        EvidenceItem("b", "x", 0.5),
        EvidenceItem("c", "x", 0.4),
        EvidenceItem("d", "x", 0.2),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.7)
    assert g.breakdown["strong"] == 1
    assert g.breakdown["weak"] == 2
    assert g.breakdown["refute"] == 0
    assert g.breakdown["insufficient"] == 1


# ==================== RCAAgent._apply_root_cause_gate 集成测试 ====================


def test_rca_agent_gate_method_returns_gateresult():
    """RCAAgent._apply_root_cause_gate 返回 GateResult（集成测试）"""
    agent = RCAAgent()
    items = [
        EvidenceItem(
            "bayesian", "order.cpu", 0.9,
            causality={"is_causal": True, "counterfactual_score": 0.85},
        ),
    ]
    result = agent._apply_root_cause_gate(items, raw_confidence=0.75)
    assert isinstance(result, GateResult)
    assert result.allowed_to_heal is True
    assert result.promoted_confidence > 0


def test_rca_agent_gate_method_refute_downgrades():
    """RCAAgent._apply_root_cause_gate：REFUTE 触发降级"""
    agent = RCAAgent()
    items = [
        EvidenceItem(  # STRONG
            "bayesian", "x", 0.9,
            causality={"is_causal": True, "counterfactual_score": 0.85},
        ),
        EvidenceItem(  # REFUTE
            "topology", "x", 0.7,
            causality={"counterfactual_score": 0.05},
        ),
    ]
    result = agent._apply_root_cause_gate(items, raw_confidence=0.8)
    assert result.allowed_to_heal is False
    assert result.promoted_confidence == 0.0


# ==================== 实战场景测试（贴近真实 RCA 输出）====================


def test_real_world_scenario_cpu_oom_causal_confirmed():
    """实战场景：CPU OOM，三源都指向同一根因 + 因果通过 → STRONG"""
    items = [
        EvidenceItem("bayesian", "order.cpu_oom", 0.85,
                     causality={"is_causal": True, "counterfactual_score": 0.82}),
        EvidenceItem("rag", "order.cpu_oom", 0.7),
        EvidenceItem("log", "order.cpu_oom", 0.9),
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.8)
    assert g.allowed_to_heal is True
    assert g.strength == EvidenceStrength.STRONG


def test_real_world_scenario_healthy_baseline_disagrees():
    """实战场景：贝叶斯/记忆/RAG 都指向 X，但健康对照显示无差异 → REFUTE 主导"""
    items = [
        EvidenceItem("bayesian", "x.cpu", 0.9,
                     causality={"is_causal": True, "counterfactual_score": 0.85}),
        EvidenceItem("memory", "x.cpu", 0.8),
        EvidenceItem("topology", "x.cpu", 0.6,
                     causality={"healthy_baseline_diff": 0.02}),  # 健康对照几乎无差异
    ]
    g = apply_root_cause_gate(items, raw_confidence=0.85)
    assert g.allowed_to_heal is False
    assert g.breakdown["refute"] == 1
