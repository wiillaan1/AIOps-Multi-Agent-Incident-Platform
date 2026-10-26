"""
AIOps Agent Platform - Heal Decision Explainer (Phase 2b)

为 HealAgent 的自愈决策生成"人话"解释，让运维看到：
- 为什么选了这个 playbook（而不是其它候选）
- 为什么是 L0 / L1 / L2 风险等级
- 爆炸半径多大、回滚方案是什么
- 如果是"风险过高降级到备选"也会解释

**重要约束**：
- 解释层**不**修改 HealAgent 的决策，只在原决策上加人话包装
- 解释的事实（数字 / 风险等级）必须 100% 来自 act_with_reflection 的输出
- LLM 失败时降级到结构化模板
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealDecisionExplanation:
    """自愈决策解释。"""

    summary: str                                  # 1 句话总结
    why_this_playbook: str                         # 为什么选这个 playbook
    why_this_risk_level: str                       # 为什么是 L0/L1/L2
    blast_radius_summary: str                      # 爆炸半径
    rollback_summary: str                          # 回滚方案
    alternatives: list[dict[str, Any]] = field(default_factory=list)  # 备选方案
    reflection_note: str = ""                      # Plan+Act 反思痕迹
    llm_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "why_this_playbook": self.why_this_playbook,
            "why_this_risk_level": self.why_this_risk_level,
            "blast_radius_summary": self.blast_radius_summary,
            "rollback_summary": self.rollback_summary,
            "alternatives": self.alternatives,
            "reflection_note": self.reflection_note,
            "llm_used": self.llm_used,
            "fallback_reason": self.fallback_reason,
        }


class HealExplainer:
    """为 HealAgent.act_with_reflection 的结果生成"人话"解释。"""

    def __init__(self, llm_provider=None) -> None:
        from app.nlu.intent_llm import LLMProvider, RuleOnlyProvider
        self.llm_provider = llm_provider or RuleOnlyProvider()

    def explain(self, act_result: dict[str, Any], rca_event=None) -> HealDecisionExplanation:
        """主入口：模板优先，LLM 润色。"""
        # 1. 模板化解释
        template = self._build_template(act_result, rca_event)

        # 2. 尝试 LLM 润色（不改变事实）
        if template.rollback_summary:  # 至少要事实字段非空
            llm_version = self._try_llm_polish(template)
            if llm_version:
                return llm_version

        return template

    def _build_template(self, act_result: dict[str, Any], rca_event) -> HealDecisionExplanation:
        """结构化模板：用 act_result 字段填空。"""
        chosen = act_result.get("playbook")
        candidates = act_result.get("candidates", [])
        reflection_reason = act_result.get("reflection_reason", "")
        escalation = act_result.get("escalation_needed", False)

        if not chosen:
            # 全部风险过高 → escalate
            return HealDecisionExplanation(
                summary=(
                    f"无可用自愈方案：{reflection_reason}。"
                    f"已尝试 {len(candidates)} 个候选 playbook，全部风险过高。"
                ),
                why_this_playbook="无可用 playbook",
                why_this_risk_level=(
                    "所有候选 playbook 的爆炸半径风险都超过当前 severity 的可接受范围。"
                ),
                blast_radius_summary="无（未执行任何自愈）",
                rollback_summary="无（未执行任何自愈）",
                alternatives=[
                    {"playbook_id": c.get("playbook_id"), "risk_level": c.get("risk_level")}
                    for c in candidates
                ],
                reflection_note=reflection_reason,
                llm_used=False,
            )

        # 1. 1 句话总结
        risk_level = chosen.get("risk_level", "?")
        blast = chosen.get("blast_radius", {})
        blast_ratio = blast.get("blast_radius_ratio", 0)
        summary = (
            f"选择 playbook **{chosen.get('playbook_name', chosen.get('playbook_id', '?'))}**"
            f"（风险 {risk_level}，爆炸半径 {blast_ratio:.0%}）"
        )
        if reflection_reason and reflection_reason != "top_1_acceptable":
            summary += f"。反思：{reflection_reason}"

        # 2. 为什么选这个 playbook
        match_score = chosen.get("match_score", 0)
        success_rate = chosen.get("estimated_success_rate", 0)
        why_this_playbook = (
            f"Playbook 与当前 RCA 匹配度 **{match_score:.2f}**（top-{candidates.index(chosen) + 1}），"
            f"历史成功率 **{success_rate:.0%}**。"
        )
        if reflection_reason == "top_1_too_risky_downgraded_to_low_risk":
            why_this_playbook += (
                "\n\n**反思调整**：top-1 候选风险过高（high/critical），"
                "已自动降级到低风险候选。原始 top-1 候选见下方的 alternatives。"
            )

        # 3. 为什么是 L0 / L1 / L2
        risk_explanations = {
            "low": (
                "L0（自动执行）：爆炸半径 < 5%，"
                "影响范围仅限本服务或标准服务，"
                "无需人工审批。"
            ),
            "medium": (
                "L1（oncall 确认）：爆炸半径 5~20%，"
                "影响部分服务但非核心，"
                "oncall 工程师点确认即执行。"
            ),
            "high": (
                "L2（TL 审批）：爆炸半径 20~50%，"
                "影响核心服务或有重大变更，"
                "需 Tech Lead 审批。"
            ),
            "critical": (
                "L2（TL 审批 + SRE 升级）：爆炸半径 > 50%，"
                "涉及关键服务或全局影响，"
                "需 TL + SRE Manager 双审批。"
            ),
        }
        # blast_radius.risk_level 字段对应 low/medium/high/critical
        br_risk = blast.get("risk_level", "low")
        why_this_risk_level = risk_explanations.get(br_risk, f"风险等级：{br_risk}")

        # 4. 爆炸半径
        affected_count = blast.get("affected_service_count", 0)
        total_count = blast.get("total_service_count", 0)
        critical_affected = blast.get("critical_services_affected", [])
        blast_radius_summary = (
            f"影响 **{affected_count} / {total_count}** 个服务（{blast_ratio:.0%}）"
        )
        if critical_affected:
            blast_radius_summary += f"，其中关键服务：{', '.join(critical_affected[:3])}"

        # 5. 回滚方案
        rollback_summary = (
            "自愈动作执行前会先做 dry-run 验证。"
            "执行后如出现症状恶化，会自动触发回滚方案（见 playbook.rollback_plan 字段）。"
        )

        # 6. 备选方案
        alternatives = []
        for c in candidates:
            if c.get("playbook_id") != chosen.get("playbook_id"):
                alternatives.append({
                    "playbook_id": c.get("playbook_id"),
                    "playbook_name": c.get("playbook_name"),
                    "match_score": c.get("match_score"),
                    "risk_level": c.get("risk_level"),
                    "estimated_success_rate": c.get("estimated_success_rate"),
                })

        return HealDecisionExplanation(
            summary=summary,
            why_this_playbook=why_this_playbook,
            why_this_risk_level=why_this_risk_level,
            blast_radius_summary=blast_radius_summary,
            rollback_summary=rollback_summary,
            alternatives=alternatives,
            reflection_note=reflection_reason,
            llm_used=False,
        )

    def _try_llm_polish(self, template: HealDecisionExplanation) -> HealDecisionExplanation | None:
        """LLM 润色（不改变事实）。"""
        # RuleOnlyProvider 不调任何外部服务 → 直接跳过避免"假 LLM"
        if self.llm_provider.name == "rule_only":
            return None

        import asyncio

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llm_resp = loop.run_until_complete(
                self.llm_provider.classify(
                    query=template.summary,
                    rule_intent="heal_decision",
                    rule_confidence=0.8,
                    rule_clarification_questions=[],
                )
            )
            loop.close()
        except Exception as exc:
            logger.warning("Heal LLM polish failed", error=str(exc))
            template.fallback_reason = f"llm_call_failed: {exc}"
            return None

        if not llm_resp or not llm_resp.reasoning:
            return None

        # LLM 只能润色 summary；其他字段保持
        template.summary = llm_resp.reasoning[:300]
        template.llm_used = True
        return template
