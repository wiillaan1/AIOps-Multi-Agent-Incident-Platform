"""
AIOps Agent Platform - RCA Natural Language Explainer (Phase 2a)

把 RCA 输出的"机器分数"翻译成"人能读懂"的分析说明。

输入：RCAEvent（含 evidence、impact_chain、recommended_actions、log_evidence）
输出：自然语言段落，关键事实必须 100% 来自 evidence（不能编造）

设计原则（最严格的可解释性约束）：
1. **零编造**：只能描述 RCAEvent.evidence 里已经存在的事实
2. **可追溯**：每一句结论都标注证据来源（如"贝叶斯后验 0.45"、"日志样本 3 条"）
3. **不确定性显式化**：置信度低时显式说明"证据不足"
4. **LLM 失败不阻塞**：返回结构化模板版本
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# v15 M4：5 维指纹（借鉴 Troubleshooter §3.1）
# 用于检索相似历史 case，弥补文字匹配的局限
# ============================================================
@dataclass(frozen=True)
class _FingerprintKey:
    """5 维故障指纹 - 内部使用，外部通过 matcher.add() 接入"""
    service: str
    alert_name: str
    metric_signature: str
    log_signature: str
    error_signature: str


_EXCEPTION_PREFIX_RE = re.compile(
    r"^[a-zA-Z_][\w]*(\.[a-zA-Z_][\w]*)*\.([A-Z][\w]*):?\s*"
)


def _normalize_log_internal(raw_log: str) -> str:
    """归一化日志签名（内部）"""
    if not raw_log:
        return ""
    return _EXCEPTION_PREFIX_RE.sub("", raw_log).strip().lower()


def _normalize_error_internal(raw_error: str) -> str:
    """归一化错误签名（内部）"""
    if not raw_error:
        return ""
    match = re.match(r"([A-Z][\w]+)", raw_error)
    return match.group(1) if match else raw_error.split()[0]


def normalize_fingerprint(
    service: str,
    raw_alert: str = "",
    raw_metric: str = "",
    raw_log: str = "",
    raw_error: str = "",
) -> _FingerprintKey:
    """从原始告警/日志/错误归一化出 5 维指纹（公开 API）

    服务于 RCAExplainer / PlaybookMatcher。
    """
    alert = ""
    keywords = [
        "HighErrorRate", "HighCPU", "HighMemory", "HighLatency",
        "PodCrash", "ServiceDown", "DiskFull", "NetworkTimeout",
    ]
    raw_upper = raw_alert.upper().replace(" ", "")
    for kw in keywords:
        if kw.upper() in raw_upper:
            alert = kw
            break
    if not alert:
        m = re.search(r"\b([A-Z][a-zA-Z]+)\b", raw_alert)
        alert = m.group(1) if m else raw_alert.strip()

    # 指标：统一为 "metric>threshold"
    metric = raw_metric.strip()
    mm = re.match(r"(\w+)\s*([<>=!]+)\s*([\d.]+)", raw_metric)
    if mm:
        metric = f"{mm.group(1)}>{mm.group(3)}"

    return _FingerprintKey(
        service=service.strip().lower(),
        alert_name=alert,
        metric_signature=metric,
        log_signature=_normalize_log_internal(raw_log),
        error_signature=_normalize_error_internal(raw_error),
    )


@dataclass
class _MatchEntry:
    key: str
    fingerprint: _FingerprintKey
    historical_score: float = 0.0


class _FingerprintMatcher:
    """5 维指纹匹配器（内部）"""

    WEIGHTS: dict[str, float] = {
        "service": 0.15,
        "alert_name": 0.25,
        "metric_signature": 0.20,
        "log_signature": 0.20,
        "error_signature": 0.20,
    }

    def __init__(self) -> None:
        self._entries: list[_MatchEntry] = []

    def add(
        self,
        key: str,
        fingerprint: _FingerprintKey,
        score: float = 0.0,
    ) -> None:
        self._entries.append(_MatchEntry(
            key=key, fingerprint=fingerprint, historical_score=score
        ))

    def find_similar(
        self,
        current: _FingerprintKey,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        scored: list[tuple[str, float]] = []
        for entry in self._entries:
            sim = sum(
                w for dim, w in self.WEIGHTS.items()
                if getattr(current, dim) == getattr(entry.fingerprint, dim)
            )
            combined = sim * 0.85 + entry.historical_score * 0.15
            scored.append((entry.key, combined))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


@dataclass
class ExplanationSection:
    """解释的一个章节（含证据溯源）。"""

    heading: str
    body: str
    evidence_refs: list[str]  # 引用了哪些 evidence 字段

    def to_dict(self) -> dict[str, Any]:
        return {
            "heading": self.heading,
            "body": self.body,
            "evidence_refs": self.evidence_refs,
        }


@dataclass
class RCAExplanation:
    """完整解释。"""

    summary: str                       # 1 句话总结
    sections: list[ExplanationSection]  # 详细分析
    evidence_count: int                # 引用了多少条证据
    confidence_level: str              # "high" / "medium" / "low"
    llm_used: bool                     # 是否调用 LLM
    fallback_reason: str = ""          # LLM 失败的原因

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "evidence_count": self.evidence_count,
            "confidence_level": self.confidence_level,
            "llm_used": self.llm_used,
            "fallback_reason": self.fallback_reason,
        }


class RCAExplainer:
    """把 RCAEvent 翻译成人话。"""

    def __init__(self, llm_provider=None) -> None:
        from app.nlu.intent_llm import LLMProvider, RuleOnlyProvider
        self.llm_provider = llm_provider or RuleOnlyProvider()
        # v15 M4：5 维指纹匹配器 - 检索相似历史 case
        self._fingerprint_matcher = _FingerprintMatcher()

    def explain(self, rca_event) -> RCAExplanation:
        """主入口：先结构化模板，再尝试 LLM 润色。"""
        # 阶段 1：结构化模板（100% 基于 evidence，无幻觉）
        template = self._build_template(rca_event)

        # 阶段 2：尝试用 LLM 润色
        if rca_event.confidence >= 0.7 and rca_event.evidence:
            llm_version = self._try_llm_polish(rca_event, template)
            if llm_version:
                return llm_version

        return template

    # ==================== 模板方式（默认） ====================

    def _build_template(self, rca_event) -> RCAExplanation:
        """结构化模板：用 evidence 字段填空。"""
        evidence = rca_event.evidence or {}
        refs: list[str] = []
        sections: list[ExplanationSection] = []

        # ===== 1. 根因结论 =====
        cause = rca_event.root_cause or "unknown"
        confidence = rca_event.confidence

        confidence_words = {0.9: "高", 0.7: "中高", 0.5: "中等", 0.0: "低"}
        confidence_level = "low"
        for threshold, word in sorted(confidence_words.items(), reverse=True):
            if confidence >= threshold:
                confidence_level = word
                break

        # 1.1 根因 + 置信度
        section1_body = (
            f"判定根因为 **{cause}**，置信度 **{confidence:.1%}（{confidence_level}）**。"
        )

        # 贝叶斯后验
        bayesian_top = evidence.get("bayesian_top_causes", [])
        if bayesian_top:
            top_cause = bayesian_top[0]
            section1_body += (
                f"\n\n贝叶斯推理后验 Top-1：{top_cause.get('cause', '?')} "
                f"（posterior={top_cause.get('posterior', 0):.2f}）"
            )
            refs.append("evidence.bayesian_top_causes")

        sections.append(ExplanationSection(
            heading="根因结论",
            body=section1_body,
            evidence_refs=refs.copy(),
        ))

        # 1.2 证据链贡献
        contribution_lines = []
        rag_matches = evidence.get("rag_matches", [])
        if rag_matches:
            top_match = rag_matches[0]
            contribution_lines.append(
                f"- 知识库 RAG 命中：{top_match.get('id', '?')}（类别 {top_match.get('category', '?')}）"
            )
            refs.append("evidence.rag_matches")

        log_evidence = evidence.get("log_evidence", {})
        if log_evidence.get("available"):
            entries = log_evidence.get("entries_count", 0)
            boosted = log_evidence.get("boosted_cause")
            if entries > 0:
                line = f"- Loki 日志：共 {entries} 条 ERROR 日志"
                if boosted:
                    if log_evidence.get("corroborates_bayesian"):
                        line += f"，签名投票指向 **{boosted}**（与贝叶斯一致 → 佐证）"
                    elif log_evidence.get("diverges_from_bayesian"):
                        line += f"，签名投票指向 **{boosted}**（与贝叶斯不一致 → 分歧）"
                contribution_lines.append(line)
                refs.append("evidence.log_evidence")

        if contribution_lines:
            sections.append(ExplanationSection(
                heading="证据链",
                body="\n".join(contribution_lines),
                evidence_refs=refs.copy(),
            ))

        # 1.3 影响链路
        impact_chain = rca_event.impact_chain or []
        if impact_chain:
            chain_text = (
                f"影响范围（共 {len(impact_chain)} 个服务）：\n"
                + "\n".join(f"- `{s}`" for s in impact_chain[:5])
            )
            if len(impact_chain) > 5:
                chain_text += f"\n- ...等 {len(impact_chain) - 5} 个"
            sections.append(ExplanationSection(
                heading="影响链路",
                body=chain_text,
                evidence_refs=["rca_event.impact_chain"],
            ))

        # 1.4 推荐操作
        actions = rca_event.recommended_actions or []
        if actions:
            actions_text = "建议执行：\n" + "\n".join(
                f"{i+1}. `{a}`" for i, a in enumerate(actions[:5])
            )
            sections.append(ExplanationSection(
                heading="建议操作",
                body=actions_text,
                evidence_refs=["rca_event.recommended_actions"],
            ))

        # 1.5 不确定性提示
        if confidence < 0.5:
            sections.append(ExplanationSection(
                heading="⚠️ 证据不足",
                body=(
                    f"当前置信度 {confidence:.1%} 偏低，建议：\n"
                    "- 等待更多指标数据或日志样本\n"
                    "- 检查相关服务是否有近期变更\n"
                    "- 考虑人工介入排查"
                ),
                evidence_refs=["rca_event.confidence"],
            ))

        # ===== Summary =====
        summary = (
            f"根因={cause}（置信度 {confidence:.1%}），"
            f"基于 {len(refs)} 类证据，"
            f"影响 {len(impact_chain)} 个服务。"
        )

        return RCAExplanation(
            summary=summary,
            sections=sections,
            evidence_count=len(refs),
            confidence_level=confidence_level,
            llm_used=False,
        )

    # ==================== LLM 润色 ====================

    def _try_llm_polish(self, rca_event, template: RCAExplanation) -> RCAExplanation | None:
        """用 LLM 润色模板（不改变事实）。"""
        import asyncio

        prompt = self._build_polish_prompt(rca_event, template)
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llm_resp = loop.run_until_complete(
                self.llm_provider.classify(
                    query=prompt,
                    rule_intent="rca_explanation",
                    rule_confidence=rca_event.confidence,
                    rule_clarification_questions=[],
                )
            )
            loop.close()
        except Exception as exc:
            logger.warning("RCA LLM polish failed", error=str(exc))
            template.fallback_reason = f"llm_call_failed: {exc}"
            return None

        if not llm_resp or not llm_resp.reasoning:
            return None

        # LLM 润色：替换 summary + 第一个 section 的 body，其它保持
        template.summary = llm_resp.reasoning[:300]
        if template.sections:
            template.sections[0].body = llm_resp.reasoning
        template.llm_used = True
        return template

    def _build_polish_prompt(self, rca_event, template: RCAExplanation) -> str:
        """构造 LLM prompt（真实 LLM 才用；MockLLMProvider 直接透传 reasoning）。"""
        return (
            f"基于以下事实写一段简洁的故障分析（≤200字，事实必须 100% 来自 evidence）：\n"
            f"根因: {rca_event.root_cause}\n"
            f"置信度: {rca_event.confidence:.1%}\n"
            f"影响: {rca_event.impact_chain}\n"
            f"建议: {rca_event.recommended_actions}"
        )
