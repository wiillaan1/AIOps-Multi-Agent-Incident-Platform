"""PPT 思路主链路 RCA 合成优化（feature flag 控制）。

本模块把阿里云 ECS《Agentic Skill 在运维中的实践》与 RCA 策略分析文档
（docs/RCA策略合理性与业界方案综合分析-2026-07-28.md）指出的主链路 RCA 缺口，
在主链路 RCA 的「合成层」做算法级修正。

通过 feature flag `AGENT_RCA_V2_ENABLED`（兼容 `AIOPS_USE_RCA_V2`）控制：
- 关闭（默认）：rca_agent._synthesize_analysis 字节不变，走旧逻辑。
- 开启：本模块接管合成，修正以下 4 项缺口：

  1. 相似度语义修正：记忆加成改用 `semantic_similarity`（真实向量相似度），
     而非 `retrieval_score`（重要性/访问频率/时效性，与相似度无关）。
  2. 结构化历史特征（PPT 第 15 页）：相似度不足时叠加结构化特征匹配
     （错误码/异常栈/拓扑路径/服务），降低文本伪相似。
  3. 证据冲突降确定性：日志与贝叶斯分歧时不再 *0.3 加分，而是施加确定性惩罚；
     并真正消费 `bayesian_penalty`（旧版是写了不读的死参数）。
  4. 消费 rejected_root_causes + provisional 记忆门禁：
     被证伪的根因从候选排除；未验证（provisional）记忆不参与置信度加成。

注意：本模块只优化「合成层」，不重写「证据收集」（5 路收集仍顺序执行）。
完整的「按假设动态下钻 + 反事实验证」由 W9 InvestigationLoopEngine（旁路闭环）承担，
两者互补、不冲突。
"""

from __future__ import annotations

import os
from typing import Any

from app.models.events import AlertEvent


class RCASynthesisV2:
    """主链路 RCA 合成优化器（flag 控制，关闭=旧行为）。"""

    # 记忆加成阈值（PPT：文本相似 40% 误报 → 向量化+结构化 92%）
    MIN_SEMANTIC_SIMILARITY = 0.6   # 语义相似度低于此值视为伪相似
    MIN_STRUCTURAL_MATCH = 0.3      # 结构化特征匹配低于此值不加成
    MAX_MEMORY_BOOST = 0.10         # 即使验证过，记忆最多弱加成（它不是因果证据）
    LOG_DIVERGENCE_PENALTY = 0.15   # 日志与贝叶斯分歧时降低的总确定性

    # 结构化特征权重（service 匹配最重，因为是上下文最强的信号）
    STRUCT_WEIGHTS = {
        "service": 0.4,
        "metric": 0.2,
        "error_signature": 0.2,
        "topology": 0.2,
    }

    def synthesize(
        self,
        bayesian_results: list[Any],
        rag_results: list[dict[str, Any]],
        impact_chain: list[Any],
        alert: AlertEvent,
        memory_results: list[dict[str, Any]] | None = None,
        log_evidence: dict[str, Any] | None = None,
        reflection_context: dict[str, Any] | None = None,
    ) -> tuple[str, float, dict[str, Any]]:
        """执行优化后的合成。返回 (root_cause, confidence, evidence)。

        与旧版 _synthesize_analysis 的契约一致，可无缝替换。
        """
        top_bayesian = bayesian_results[0] if bayesian_results else None
        top_rag = rag_results[0] if rag_results else None
        top_memory = memory_results[0] if memory_results else None

        # ===== 缺口 4 之一：排除被证伪的根因（rejected_root_causes）=====
        rejected = self._rejected_causes(reflection_context)
        candidates = (
            [b for b in bayesian_results if b.name not in rejected]
            if rejected else list(bayesian_results)
        )
        # 全部被拒时不强行清空，退回完整候选（避免无候选可用）
        if not candidates:
            candidates = list(bayesian_results)
        top_b = candidates[0] if candidates else None

        # ===== 缺口 1 + 2：记忆加成用真实相似度 + 结构化特征 =====
        memory_boost, structural_match = self._compute_memory_boost(top_memory, alert)

        # ===== 缺口 3：日志证据与冲突处理 =====
        log_evidence = log_evidence or {"available": False, "source": "not_provided"}
        log_boost = (
            log_evidence.get("confidence_boost", 0.0)
            if log_evidence.get("available") else 0.0
        )
        log_cause = log_evidence.get("boosted_cause")
        log_corroboration = False
        log_divergence = False
        if log_cause and top_b and log_boost > 0.04:
            if log_cause == top_b.name:
                log_corroboration = True
            else:
                log_divergence = True

        # ===== 综合根因选择（与旧版同优先级，但基于排除后的候选）=====
        root_cause = self._choose_root_cause(
            top_b, top_rag, top_memory, candidates
        )

        # ===== 综合置信度（修正冲突 + 消费 bayesian_penalty）=====
        confidence, divergence_penalty, bayes_penalty = self._compute_confidence(
            top_b, top_rag, memory_boost, log_boost,
            log_corroboration, log_divergence, reflection_context,
        )

        # ===== 构建 evidence（含 Top-K 候选）=====
        evidence = self._build_evidence(
            top_b, top_rag, top_memory, log_evidence, log_boost,
            log_corroboration, log_divergence, memory_boost, structural_match,
            divergence_penalty, bayes_penalty, candidates, impact_chain, alert,
        )
        return root_cause, round(confidence, 4), evidence

    # ------------------------------------------------------------------
    # 缺口 1 + 2：记忆加成
    # ------------------------------------------------------------------
    def _compute_memory_boost(
        self,
        top_memory: dict[str, Any] | None,
        alert: AlertEvent,
    ) -> tuple[float, float]:
        """返回 (memory_boost, structural_match)。

        只有「语义相似 > 阈值 AND 结构化匹配 AND 已验证」才给弱加成。
        provisional（未验证）记忆一律不加成——切断自我强化回路。
        """
        if not top_memory:
            return 0.0, 0.0

        # 缺口 1：读 semantic_similarity，不再读 retrieval_score
        semantic_similarity = float(top_memory.get("semantic_similarity", 0.0))

        # 缺口 2：结构化特征匹配
        structural_match = self._structural_similarity(top_memory, alert)

        # 缺口 4 之二：provisional 记忆不加成
        verification_status = top_memory.get("verification_status", "provisional")
        if verification_status != "verified":
            return 0.0, structural_match

        # 双门槛：语义相似 + 结构化都达标才加成（PPT 向量化+结构化 92%）
        if (semantic_similarity >= self.MIN_SEMANTIC_SIMILARITY
                and structural_match >= self.MIN_STRUCTURAL_MATCH):
            boost = min(self.MAX_MEMORY_BOOST, semantic_similarity * 0.12)
            return boost, structural_match
        return 0.0, structural_match

    def _structural_similarity(
        self, memory: dict[str, Any], alert: AlertEvent
    ) -> float:
        """基于结构化特征计算相似度（service/metric/错误签名/拓扑）。

        文本相似度（40% 误报）的根因是「表象相似但机制不同」；
        结构化特征（错误码、栈、拓扑路径）更能区分真实机制。
        """
        if not memory:
            return 0.0
        score = 0.0
        weights = self.STRUCT_WEIGHTS

        # service 匹配（上下文最强信号）
        mem_content = str(memory.get("content", "")) + " " + str(memory.get("summary", ""))
        mem_tags = memory.get("tags", [])
        if alert.service and (
            alert.service in mem_content or alert.service in mem_tags
        ):
            score += weights["service"]

        # metric 匹配
        if alert.metric and alert.metric in mem_content:
            score += weights["metric"]

        # 错误签名匹配（日志证据里的错误模式）
        log_sig = memory.get("log_signature") or self._extract_log_signature(mem_content)
        if log_sig and alert.labels:
            alert_sig = alert.labels.get("error_signature", "")
            if alert_sig and alert_sig == log_sig:
                score += weights["error_signature"]

        # 拓扑路径匹配（历史记忆涉及的上下游与当前影响链重合）
        if memory.get("impact_chain"):
            mem_chain = set(memory.get("impact_chain", []))
            # alert 自身服务在历史影响链里即视为拓扑相关
            if alert.service in mem_chain:
                score += weights["topology"]

        return min(score, 1.0)

    @staticmethod
    def _extract_log_signature(content: str) -> str:
        """从记忆文本里粗提取错误签名（用于结构化匹配）。"""
        signatures = [
            "timeout", "connection refused", "oom", "outofmemory",
            "deadlock", "lock wait", "nullpointer", "stacktrace",
        ]
        low = content.lower()
        for sig in signatures:
            if sig in low:
                return sig
        return ""

    # ------------------------------------------------------------------
    # 缺口 4 之一：rejected 排除
    # ------------------------------------------------------------------
    @staticmethod
    def _rejected_causes(
        reflection_context: dict[str, Any] | None
    ) -> set[str]:
        """从 reflection_context 提取被证伪的根因集合。"""
        if not reflection_context:
            return set()
        rejected = reflection_context.get("rejected_root_causes", [])
        if isinstance(rejected, list):
            return {str(r) for r in rejected}
        return set()

    # ------------------------------------------------------------------
    # 根因选择
    # ------------------------------------------------------------------
    @staticmethod
    def _choose_root_cause(
        top_b: Any | None,
        top_rag: dict[str, Any] | None,
        top_memory: dict[str, Any] | None,
        candidates: list[Any],
    ) -> str:
        """综合根因选择。基于排除 rejected 后的候选，不让 RAG 重新引入被拒根因。"""
        if top_b is None:
            return "unknown"
        # 历史记忆只在「已验证」时作为弱佐证（这里不再用历史覆盖贝叶斯，
        # 避免把「历史曾如此」当成「当前也如此」——修正旧版历史优先级最高的问题）。
        if top_rag and top_rag.get("match_score", 0) > 0.7:
            rag_causes = top_rag.get("root_causes", [])
            candidate_names = {c.name for c in candidates}
            # 只采纳「RAG 支持 且 在排除后候选集内」的根因，
            # 避免 RAG 高匹配把被证伪的根因重新引入。
            if rag_causes:
                for rc in rag_causes:
                    if rc in candidate_names:
                        return rc
        return top_b.name

    # ------------------------------------------------------------------
    # 缺口 3：置信度计算（冲突降确定性 + bayesian_penalty）
    # ------------------------------------------------------------------
    def _compute_confidence(
        self,
        top_b: Any | None,
        top_rag: dict[str, Any] | None,
        memory_boost: float,
        log_boost: float,
        log_corroboration: bool,
        log_divergence: bool,
        reflection_context: dict[str, Any] | None,
    ) -> tuple[float, float, float]:
        """返回 (confidence, divergence_penalty, bayes_penalty)。

        核心修正：分歧时不再 *0.3 加分，而是施加惩罚降低总确定性。
        同时真正消费 bayesian_penalty（旧版死参数）。
        """
        if top_b is None:
            return 0.3, 0.0, 0.0

        # 基础合成（权重与旧版一致，保证可比性）
        if top_rag:
            confidence = min(
                0.95,
                0.55 * top_b.posterior
                + 0.25 * top_rag.get("match_score", 0)
                + 0.10 * top_b.evidence_strength
                + 0.10 * memory_boost * 10,
            )
        else:
            confidence = min(0.90, top_b.posterior * 0.8 + memory_boost)

        # 缺口 3 修复：日志一致才加分；分歧不加分
        effective_log_boost = log_boost if log_corroboration else 0.0
        confidence = min(0.95, confidence + effective_log_boost)

        # 分歧惩罚：降低总确定性（而非加分）
        divergence_penalty = self.LOG_DIVERGENCE_PENALTY if log_divergence else 0.0

        # 消费 bayesian_penalty（旧版 suggested_context 里写了但从不读）
        bayes_penalty = float((reflection_context or {}).get("bayesian_penalty", 0.0))

        confidence = max(0.0, confidence - divergence_penalty - bayes_penalty)
        return confidence, divergence_penalty, bayes_penalty

    # ------------------------------------------------------------------
    # evidence 构建（含 Top-K 候选）
    # ------------------------------------------------------------------
    def _build_evidence(
        self,
        top_b: Any | None,
        top_rag: dict[str, Any] | None,
        top_memory: dict[str, Any] | None,
        log_evidence: dict[str, Any],
        log_boost: float,
        log_corroboration: bool,
        log_divergence: bool,
        memory_boost: float,
        structural_match: float,
        divergence_penalty: float,
        bayes_penalty: float,
        candidates: list[Any],
        impact_chain: list[Any],
        alert: AlertEvent,
    ) -> dict[str, Any]:
        """构建结构化证据 dict，含 Top-K 候选（P1）。"""
        return {
            "bayesian_top": top_b.name if top_b else "unknown",
            "bayesian_posterior": round(top_b.posterior, 4) if top_b else 0,
            "rag_top_match": top_rag["id"] if top_rag else "none",
            "rag_match_score": round(top_rag.get("match_score", 0), 4) if top_rag else 0,
            "historical_memory_matches": 1 if top_memory else 0,
            # 缺口 1：同时暴露两个分数，审计时可看出差异
            "memory_retrieval_score": round(top_memory.get("retrieval_score", 0), 4) if top_memory else 0,
            "memory_semantic_similarity": round(top_memory.get("semantic_similarity", 0), 4) if top_memory else 0,
            "memory_structural_match": round(structural_match, 4),
            "memory_confidence_boost": round(memory_boost, 4),
            "memory_verification_status": top_memory.get("verification_status", "none") if top_memory else "none",
            "log_evidence": {
                "available": log_evidence.get("available", False),
                "source": log_evidence.get("source", "unavailable"),
                "entries_count": log_evidence.get("entries_count", 0),
                "boosted_cause": log_evidence.get("boosted_cause"),
                "cause_hits": log_evidence.get("cause_hits", {}),
                "confidence_boost": log_evidence.get("confidence_boost", 0.0),
                "effective_boost": round(log_boost if log_corroboration else 0.0, 4),
                "corroborates_bayesian": log_corroboration,
                "diverges_from_bayesian": log_divergence,
                "samples": log_evidence.get("samples", {}),
            },
            "log_corroboration": log_corroboration,
            "log_divergence": log_divergence,
            # 缺口 3：显式记录惩罚量，审计可追溯
            "divergence_penalty": round(divergence_penalty, 4),
            "bayesian_penalty_applied": round(bayes_penalty, 4),
            # P1：Top-K 候选输出（不只单一根因）
            "top_k_candidates": [
                {
                    "cause": c.name,
                    "posterior": round(c.posterior, 4),
                    "evidence_strength": round(getattr(c, "evidence_strength", 0), 4),
                    "supporting_evidence": [],
                    "counter_evidence": [],
                    "verification_status": "unverified",
                }
                for c in candidates[:3]
            ],
            "affected_services": [s.service for s in impact_chain],
            "critical_services_affected": [
                s.service for s in impact_chain if getattr(s, "tier", "") == "critical"
            ],
            "total_impact_hops": max(
                (getattr(s, "hop_distance", 0) for s in impact_chain), default=0
            ),
            "alert_metric": alert.metric,
            "alert_value": alert.value,
            "alert_threshold": alert.threshold,
            "synthesis_version": "v2",
        }


# ----------------------------------------------------------------------
# feature flag + 单例（仿 W8 main_path_gate.py 模式）
# ----------------------------------------------------------------------
_singleton: RCASynthesisV2 | None = None


def _flag_enabled() -> bool:
    """读取 RCA V2 feature flag（兼容 env 与 config 两种来源）。"""
    # 优先 env（与 W8 AIOPS_USE_W8_X 一致）
    val = os.getenv("AIOPS_USE_RCA_V2", "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    # 回退 config
    try:
        from app.config import get_config
        return bool(get_config().agent.rca_v2_enabled)
    except Exception:
        return False


def get_rca_synthesis_v2() -> RCASynthesisV2 | None:
    """返回 V2 优化器单例；flag 关闭时返回 None（调用方走旧逻辑）。"""
    global _singleton
    if not _flag_enabled():
        return None
    if _singleton is None:
        _singleton = RCASynthesisV2()
    return _singleton


def reset_rca_synthesis_v2() -> None:
    """重置单例（测试用）。"""
    global _singleton
    _singleton = None
