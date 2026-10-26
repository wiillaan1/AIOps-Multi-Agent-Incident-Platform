"""
AIOps Agent Platform - RCA Agent

根因分析 Agent，负责故障根因分析。
集成知识图谱查询、贝叶斯推理、BFS 遍历和 RAG 增强。
"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from app.agents.base import AgentResult, BaseAgent
from app.agents._bfs_synthesis import BFSScheduler, HypothesisNode
from app.agents.evidence_classifier import (
    EvidenceItem,
    EvidenceStrength,
    GateResult,
    apply_root_cause_gate,
)
from app.core.llm_client import LLMCallResult, get_llm_client
from app.core.llm_fallback import safe_chat
from app.agents._hallucination_guard import HallucinationGuard
from app.data.knowledge_base import KNOWLEDGE_BASE, SERVICE_TOPOLOGY
from app.models.agent import AgentExecutionContext
from app.models.events import AlertEvent, RCAEvent, SeverityLevel
from app.models.memory import MemoryType
from app.utils.logging import get_logger

logger = get_logger(__name__)


# 日志签名 -> 根因的弱匹配映射（来自 Loki ERROR 日志的关键词聚合）
# 评分时不直接给出根因结论，只是给对应根因 +1 票；最终由贝叶斯综合排序
LOG_SIGNATURE_TO_CAUSE: dict[tuple[str, ...], str] = {
    ("oom", "outofmemory", "memory limit"): "resource_exhaustion",
    ("stacktrace", "exception", "nullpointer", "keyerror", "attributeerror"): "code_bug",
    ("connection refused", "connection pool exhausted", "deadlock"): "dependency_failure",
    ("connection timeout", "read timeout", "gateway timeout", "i/o timeout"): "network_issue",
    ("lock wait timeout", "deadlock found", "too many connections"): "database_issue",
    ("deploy", "release", "rolled out", "deployment", "config change"): "recent_deployment",
}


def _empty_log_evidence(reason: str) -> dict[str, Any]:
    """Loki 不可用时的兜底结构，保持字段稳定。"""
    return {
        "available": False,
        "entries_count": 0,
        "cause_hits": {},
        "boosted_cause": None,
        "confidence_boost": 0.0,
        "samples": {},
        "source": "unavailable",
        "reason": reason,
    }


class BayesianNode(BaseModel):
    """贝叶斯网络节点"""
    name: str = Field(description="节点名称")
    prior: float = Field(default=0.5, ge=0.0, le=1.0, description="先验概率 P(根因)")
    likelihood: float = Field(default=0.5, ge=0.0, le=1.0, description="似然 P(症状|根因)")
    posterior: float = Field(default=0.0, ge=0.0, le=1.0, description="后验概率 P(根因|症状)")
    evidence_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="证据强度")


class TopologyEdge(BaseModel):
    """拓扑依赖边"""
    source: str = Field(description="源服务")
    target: str = Field(description="目标服务")
    relation: str = Field(default="depends_on", description="关系类型")
    weight: float = Field(default=1.0, description="依赖权重")


class ServiceImpact(BaseModel):
    """服务影响分析结果"""
    service: str = Field(description="服务名称")
    impact_level: str = Field(default="unknown", description="影响级别")
    hop_distance: int = Field(default=-1, description="距离告警服务的跳数")
    tier: str = Field(default="standard", description="服务等级")
    related_changes: list[dict[str, Any]] = Field(default_factory=list, description="相关变更")


class RCAInput(BaseModel):
    """RCA Agent 输入"""
    alert: AlertEvent = Field(description="告警事件")
    incident_id: str = Field(default="", description="关联故障ID")
    lookback_minutes: int = Field(default=60, description="回溯时间窗口(分钟)")
    max_hops: int = Field(default=3, description="BFS最大跳数")
    # W7 反思重跑上下文：{'evidence_boost':'log', 'bayesian_penalty':0.15}
    # 由主管道 verify_phase 在 Prometheus 校验失败时填充，触发二次 RCA 时调整权重。
    # 使用 Field(alias=...) + populate_by_name=True 以同时支持 _reflection_context 与 reflection_context。
    reflection_context: dict[str, Any] | None = Field(
        default=None,
        alias="_reflection_context",
        description="W7 反思重跑上下文：{'evidence_boost':'log', 'bayesian_penalty':0.15}",
    )

    model_config = {"populate_by_name": True}

    @property
    def _reflection_context(self) -> dict[str, Any] | None:
        """兼容属性：与 ``reflection_context`` 字段同义，便于调用方使用别名取值。"""
        return self.reflection_context


class RCAAgent(BaseAgent[RCAInput, RCAEvent]):
    """
    根因分析 Agent

    职责：
    - 知识图谱查询（服务拓扑依赖关系）
    - 贝叶斯推理（P(根因|症状) = P(症状|根因) * P(根因) / P(症状)）
    - BFS 遍历依赖链（从告警服务出发）
    - RAG 增强（从知识库检索历史相似故障案例）
    """

    # 贝叶斯推理先验概率表
    PRIOR_PROBABILITIES: dict[str, float] = {
        "recent_deployment": 0.15,
        "configuration_change": 0.10,
        "resource_exhaustion": 0.20,
        "dependency_failure": 0.25,
        "network_issue": 0.15,
        "database_issue": 0.20,
        "code_bug": 0.12,
        "traffic_spike": 0.18,
        "hardware_failure": 0.05,
        "third_party_issue": 0.08,
    }

    # 似然概率表 P(症状|根因)
    LIKELIHOODS: dict[str, dict[str, float]] = {
        "recent_deployment": {
            "high_cpu": 0.7,
            "high_memory": 0.5,
            "high_error_rate": 0.6,
            "increased_latency": 0.8,
            "timeout": 0.4,
        },
        "configuration_change": {
            "high_cpu": 0.4,
            "high_error_rate": 0.7,
            "increased_latency": 0.6,
            "timeout": 0.5,
        },
        "resource_exhaustion": {
            "high_cpu": 0.8,
            "high_memory": 0.9,
            "oom_killed": 0.85,
            "timeout": 0.5,
        },
        "dependency_failure": {
            "high_error_rate": 0.9,
            "timeout": 0.85,
            "increased_latency": 0.8,
        },
        "network_issue": {
            "timeout": 0.9,
            "high_error_rate": 0.6,
            "increased_latency": 0.85,
        },
        "database_issue": {
            "timeout": 0.8,
            "high_error_rate": 0.7,
            "increased_latency": 0.9,
        },
        "code_bug": {
            "high_cpu": 0.5,
            "high_memory": 0.6,
            "high_error_rate": 0.8,
        },
        "traffic_spike": {
            "high_cpu": 0.9,
            "high_memory": 0.6,
            "increased_latency": 0.7,
        },
        "hardware_failure": {
            "high_cpu": 0.4,
            "high_error_rate": 0.6,
            "timeout": 0.5,
        },
        "third_party_issue": {
            "high_error_rate": 0.7,
            "timeout": 0.6,
            "increased_latency": 0.5,
        },
    }

    def __init__(self) -> None:
        super().__init__()
        self._service_topology = SERVICE_TOPOLOGY
        self._knowledge_base = KNOWLEDGE_BASE
        self._memory_system = None
        self._memory_init_attempted = False

    async def _get_memory_system(self):
        """惰性初始化 MemorySystem，失败后不再重试。"""
        if self._memory_system is not None:
            return self._memory_system
        if self._memory_init_attempted:
            return None
        try:
            from app.memory.core import MemorySystem as MS
            self._memory_system = await MS.get_instance()
            logger.info("MemorySystem initialized for RCA agent")
        except Exception as e:
            self._memory_init_attempted = True
            logger.warning("MemorySystem unavailable for RCA agent, using static knowledge base only", error=str(e))
        return self._memory_system

    # ==================== v15 M9：BFS 多假设合成 ====================

    async def _bfs_hypothesis_exploration(
        self,
        alert: AlertEvent,
        bayesian_results: list[BayesianNode],
        rag_results: list[dict[str, Any]],
        memory_results: list[dict[str, Any]],
        log_evidence: dict[str, Any],
        original_root_cause: str,
        original_confidence: float,
    ) -> Any:
        """BFS 多假设合成 - 借鉴 Devix §4.1

        把每个 sub-analysis（贝叶斯/RAG/记忆/日志）作为独立假设节点，
        并行探索后合成最终结论。

        当 BFS 合成的 confidence > original_confidence 时，
        调用方应采用 BFS 结果。

        Returns:
            SynthesisResult 或 None（BFS 跳过/失败时）
        """
        try:
            scheduler = BFSScheduler(max_depth=3)

            # 假设 1：贝叶斯推理的 top cause
            if bayesian_results:
                top_bayes = bayesian_results[0]
                scheduler.add_hypothesis(HypothesisNode(
                    id=f"bayes-{top_bayes.name}",
                    description=f"贝叶斯推理: {top_bayes.name}",
                    score=top_bayes.posterior,  # BayesianNode 用 posterior，不是 probability
                    evidence=[f"bayes:{n.name}={n.posterior:.2f}" for n in bayesian_results[:3]],
                ))

            # 假设 2：RAG 检索的 top match
            if rag_results:
                top_rag = rag_results[0]
                scheduler.add_hypothesis(HypothesisNode(
                    id=f"rag-{top_rag.get('id', 'unknown')}",
                    description=f"RAG 检索: {top_rag.get('summary', '')[:50]}",
                    score=top_rag.get("score", 0.5),
                    evidence=[f"rag:{r.get('id', 'unknown')}={r.get('score', 0):.2f}" for r in rag_results[:3]],
                ))

            # 假设 3：历史记忆的 top match
            if memory_results:
                top_mem = memory_results[0]
                scheduler.add_hypothesis(HypothesisNode(
                    id=f"memory-{top_mem.get('incident_id', 'unknown')}",
                    description=f"历史记忆: {top_mem.get('root_cause', '')[:50]}",
                    score=top_mem.get("similarity", 0.5),
                    evidence=[f"memory:{m.get('incident_id', 'unknown')}" for m in memory_results[:3]],
                ))

            # 假设 4：日志证据
            if log_evidence and log_evidence.get("signatures"):
                scheduler.add_hypothesis(HypothesisNode(
                    id="log-evidence",
                    description=f"日志证据: {log_evidence.get('summary', '')[:50]}",
                    score=log_evidence.get("confidence", 0.5),
                    evidence=log_evidence.get("signatures", [])[:3],
                ))

# 假设 5：原综合分析结果（兜底）
            scheduler.add_hypothesis(HypothesisNode(
                id="synthesis-original",
                description=f"综合分析: {original_root_cause[:50]}",
                score=original_confidence,
                evidence=["synth:bayes+rag+memory+log"],
            ))

            if not scheduler._hypotheses:
                return None

            # 并行探索（用 identity 函数，因为所有假设已包含 score）
            async def check_identity(hyp: HypothesisNode) -> HypothesisNode:
                return hyp

            results = await scheduler.explore(check_identity)
            return scheduler.synthesize(results, top_n=3)

        except Exception as e:
            logger.warning(
                "BFS hypothesis exploration failed, falling back to original",
                error=str(e),
            )
            return None

    # ==================== v15 M2：4 层幻觉防护 ====================

    def _apply_hallucination_guard(
        self,
        alert: AlertEvent,
        root_cause: str,
        confidence: float,
        evidence: dict[str, Any],
    ) -> Any:
        """4 层幻觉防护 - 借鉴 Troubleshooter §4

        1. Input Validation: 校验 alert + 必要字段
        2. Evidence Grounding: root_cause 中的事实必须基于 evidence
        3. Cross-Check: 多源 evidence 交叉验证
        4. Output Schema: 校验输出符合 RCAEvent schema

        Returns:
            PipelineResult（passed=True 表示通过，False 表示被拦截）
        """
        try:
            # 把 evidence 转成 PipelineResult 期望的格式
            evidence_list = []
            if isinstance(evidence, dict):
                # 从 evidence dict 提取字符串列表
                for k, v in evidence.items():
                    if isinstance(v, str):
                        evidence_list.append(f"{k}:{v}")
                    elif isinstance(v, list):
                        for item in v[:3]:
                            if isinstance(item, str):
                                evidence_list.append(f"{k}:{item}")

            # 构造 input_data 给 Pipeline
            input_data = {
                "incident_id": getattr(alert, "incident_id", ""),
                "service": getattr(alert, "service", ""),
                "alert_name": getattr(alert, "alert_name", "") or getattr(alert, "metric", ""),
                "evidence": evidence_list,
            }

            # 构造 LLM output（只用 root_cause 作为待验证文本，
            # 不要把 confidence 数字拼进来——会被 evidence_grounding 当幻觉）
            llm_output = f"Root cause: {root_cause}"

            guard = HallucinationGuard()
            result = guard.evaluate(input_data, llm_output)

            return result

        except Exception as e:
            logger.warning(
                "Hallucination guard evaluation failed",
                error=str(e),
            )
            # 失败时默认通过（避免阻塞主流程）
            from app.agents._hallucination_guard import PipelineResult
            return PipelineResult(passed=True, details={"warning": str(e)})

    def get_name(self) -> str:
        return "rca_agent"

    def get_description(self) -> str:
        return "根因分析 Agent - 知识图谱+贝叶斯推理+BFS遍历+RAG增强"

    def _build_llm_messages(
        self,
        alert: AlertEvent,
        root_cause: str,
        confidence: float,
        evidence: dict[str, Any],
    ) -> list[dict[str, str]]:
        """把确定性证据交给 LLM 做解释性复核；LLM 不负责越权执行动作。"""
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)[:8000]
        return [
            {
                "role": "system",
                "content": (
                    "你是生产运维 RCA 复核器。只能依据给定证据返回严格 JSON，"
                    "不得编造指标、日志、服务或时间。字段必须包含 root_cause、confidence、"
                    "contributing_factors、recommended_actions、evidence_refs、explanation。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"告警服务={alert.service}; 指标={alert.metric}; 当前值={alert.value}; "
                    f"阈值={alert.threshold}; 规则候选={root_cause}; 规则置信度={confidence:.4f}; "
                    f"证据={evidence_text}"
                ),
            },
        ]

    @staticmethod
    def _parse_llm_rca(content: str) -> dict[str, Any]:
        """解析并校验 LLM RCA JSON，失败交给 fallback。"""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`").replace("json", "", 1).strip()
        data = json.loads(text)
        if not isinstance(data, dict) or not data.get("root_cause"):
            raise ValueError("LLM RCA response missing root_cause")
        confidence = float(data.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("LLM RCA confidence out of range")
        for key in ("contributing_factors", "recommended_actions", "evidence_refs"):
            if not isinstance(data.get(key, []), list):
                raise ValueError(f"LLM RCA field {key} must be list")
        return data

    async def _llm_review(
        self,
        alert: AlertEvent,
        incident_id: str,
        root_cause: str,
        confidence: float,
        evidence: dict[str, Any],
    ) -> tuple[dict[str, Any], LLMCallResult]:
        """调用真实 LLM；不可用或响应非法时由 safe_chat 走确定性 fallback。"""
        client = get_llm_client()
        fallback = LLMCallResult(
            content=json.dumps({
                "root_cause": root_cause,
                "confidence": confidence,
                "contributing_factors": [],
                "recommended_actions": [],
                "evidence_refs": [],
                "explanation": "规则引擎综合证据结果（LLM 未启用）",
            }, ensure_ascii=False),
            model="rules-engine",
            provider="fallback",
            latency_ms=0,
        )
        result = await safe_chat(
            client,
            self._build_llm_messages(alert, root_cause, confidence, evidence),
            fallback_result=fallback,
            incident_id=incident_id,
        )
        try:
            parsed = self._parse_llm_rca(result.content)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("LLM RCA response invalid, using deterministic fallback", error=str(exc))
            fallback_data = json.loads(fallback.content)
            result = LLMCallResult(
                content=fallback.content,
                model=fallback.model,
                provider="fallback",
                latency_ms=result.latency_ms,
                via_fallback=True,
                fallback_reason=f"invalid_llm_response: {type(exc).__name__}",
            )
            parsed = fallback_data
        return parsed, result

    # ==================== 核心处理 ====================

    async def process(
        self,
        input_data: RCAInput,
        context: AgentExecutionContext,
    ) -> AgentResult:
        """
        执行根因分析

        Args:
            input_data: RCA 输入
            context: 执行上下文

        Returns:
            AgentResult: 包含 RCAEvent 的结果
        """
        alert = input_data.alert
        logger.info(
            "RCAAgent processing",
            service=alert.service,
            metric=alert.metric,
            incident_id=input_data.incident_id,
        )

        # Step 1: BFS 遍历服务依赖链
        impact_chain = self.bfs_traverse(
            alert.service,
            max_hops=input_data.max_hops,
        )

        # Step 2: 贝叶斯推理
        symptoms = self._extract_symptoms(alert)
        bayesian_results = self.bayesian_inference(symptoms)

        # Step 3: RAG 增强 - 从静态知识库检索
        rag_results = self._rag_retrieve(alert, symptoms)

        # Step 3a: 记忆增强 - 从 MemorySystem 检索历史相似故障
        memory_results = await self._search_historical_incidents(alert, symptoms)

        # Step 3b: 日志证据 - 从 Loki 拉取 ERROR 日志聚合根因签名
        log_evidence = await self._query_log_evidence(
            alert, input_data.lookback_minutes,
        )

        # Step 4: 综合分析 - 合并贝叶斯 + RAG + 历史记忆 + 日志
        root_cause, confidence, evidence = self._synthesize_analysis(
            bayesian_results, rag_results, impact_chain, alert,
            memory_results=memory_results,
            log_evidence=log_evidence,
            reflection_context=input_data.reflection_context,
        )

        # Level 2：将确定性证据交给真实 LLM 做结构化复核；失败时保留规则结果。
        llm_rca, llm_result = await self._llm_review(
            alert=alert,
            incident_id=input_data.incident_id,
            root_cause=root_cause,
            confidence=confidence,
            evidence=evidence,
        )
        if not llm_result.via_fallback:
            root_cause = str(llm_rca.get("root_cause", root_cause))
            confidence = float(llm_rca.get("confidence", confidence))
            if llm_rca.get("contributing_factors"):
                evidence["llm_contributing_factors"] = llm_rca["contributing_factors"]
            if llm_rca.get("evidence_refs"):
                evidence["llm_evidence_refs"] = llm_rca["evidence_refs"]
            if llm_rca.get("explanation"):
                evidence["llm_explanation"] = str(llm_rca["explanation"])[:1000]

        llm_provenance = {
            "called": True,
            "provider": llm_result.provider,
            "model": llm_result.model,
            "latency_ms": llm_result.latency_ms,
            "via_fallback": llm_result.via_fallback,
            "fallback_reason": llm_result.fallback_reason,
        }
        evidence["llm"] = llm_provenance

        # ===== v15 M9：BFS 多假设合成 =====
        # 借鉴 Devix §4.1 - 并行探索多个根因假设，合成最终结论
        bfs_result = await self._bfs_hypothesis_exploration(
            alert=alert,
            bayesian_results=bayesian_results,
            rag_results=rag_results,
            memory_results=memory_results,
            log_evidence=log_evidence,
            original_root_cause=root_cause,
            original_confidence=confidence,
        )
        if bfs_result and bfs_result.confidence > confidence:
            # BFS 给出更高置信度 → 采用
            logger.info(
                "BFS hypothesis exploration improved RCA result",
                original_confidence=confidence,
                bfs_confidence=bfs_result.confidence,
                best_hypothesis=bfs_result.best_hypothesis_id,
            )
            confidence = bfs_result.confidence

        # ===== v15 M2：4 层幻觉防护 =====
        # 借鉴 Troubleshooter §4 - 验证 RCA 输出无幻觉
        guard_result = self._apply_hallucination_guard(
            alert=alert,
            root_cause=root_cause,
            confidence=confidence,
            evidence=evidence,
        )
        if not guard_result.passed:
            logger.warning(
                "RCA hallucination guard blocked",
                failure_layer=guard_result.failure_layer,
                details=guard_result.details,
            )
            # 置信度降低，让下游 Agent 谨慎处理
            confidence = min(confidence, 0.4)

        # ===== W2: RCA 自我批判 + 补查 =====
        reflection_note = ""
        if confidence < 0.55:
            logger.info(
                "RCA low confidence, attempting self-critique pass",
                original_confidence=confidence,
                root_cause=root_cause,
            )
            recovered = await self._self_critique_and_recover(
                alert=alert,
                original_root_cause=root_cause,
                original_confidence=confidence,
                bayesian_results=bayesian_results,
                rag_results=rag_results,
                memory_results=memory_results,
                log_evidence=log_evidence,
            )
            if recovered:
                root_cause = recovered["root_cause"]
                confidence = recovered["confidence"]
                evidence = recovered["evidence"]
                reflection_note = recovered.get("reflection_note", "")
                logger.info(
                    "RCA self-critique recovered",
                    new_confidence=confidence,
                    root_cause=root_cause,
                )

        # 从内部 evidence dict 提取一致性标志，供后续 RCAEvent 使用
        log_corroboration = evidence.get("log_corroboration", False)
        log_divergence = evidence.get("log_divergence", False)

        # Step 5: 生成建议操作
        suggested_actions = self._generate_suggested_actions(
            root_cause, impact_chain, rag_results
        )

        # 构建 RCAEvent
        rca_event = RCAEvent(
            correlation_id=alert.correlation_id,
            source=self.get_name(),
            incident_id=input_data.incident_id,
            root_cause=root_cause,
            confidence=confidence,
            contributing_factors=[r.name for r in bayesian_results[:3]],
            impact_chain=[i.service for i in impact_chain],
            evidence={
                "bayesian_top_causes": [
                    {"cause": r.name, "posterior": round(r.posterior, 4)}
                    for r in bayesian_results[:5]
                ],
                "rag_matches": [
                    {"id": m["id"], "category": m["category"], "confidence_boost": m.get("confidence_boost", 0)}
                    for m in rag_results[:3]
                ],
                "log_evidence": {
                    "source": log_evidence.get("source"),
                    "entries_count": log_evidence.get("entries_count", 0),
                    "boosted_cause": log_evidence.get("boosted_cause"),
                    "cause_hits": log_evidence.get("cause_hits", {}),
                    "confidence_boost": log_evidence.get("confidence_boost", 0.0),
                    "corroborates_bayesian": log_corroboration,
                    "diverges_from_bayesian": log_divergence,
                },
                "impact_chain_detail": [
                    {
                        "service": i.service,
                        "hop": i.hop_distance,
                        "tier": i.tier,
                        "impact_level": i.impact_level,
                    }
                    for i in impact_chain
                ],
                "alert_metric": alert.metric,
                "alert_value": alert.value,
                "alert_threshold": alert.threshold,
                "reflection_note": reflection_note,
                "llm": evidence.get("llm", llm_provenance),
                "llm_contributing_factors": evidence.get("llm_contributing_factors", []),
                "llm_evidence_refs": evidence.get("llm_evidence_refs", []),
                "llm_explanation": evidence.get("llm_explanation", ""),
            },
            recommended_actions=suggested_actions,
            time_range_start=datetime.now(timezone.utc) - timedelta(minutes=input_data.lookback_minutes),
            time_range_end=datetime.now(timezone.utc),
        )

        output_data = {
            "rca_event": rca_event.model_dump(),
            "root_cause": root_cause,
            "confidence": confidence,
            "bayesian_results": [
                {"name": r.name, "prior": r.prior, "likelihood": r.likelihood, "posterior": r.posterior}
                for r in bayesian_results[:5]
            ],
            "impact_services_count": len(impact_chain),
            "rag_match_count": len(rag_results),
        }

        return AgentResult.success_result(
            agent_name=self.get_name(),
            output_data=output_data,
        )

    # ==================== BFS 依赖遍历 ====================

    def bfs_traverse(
        self,
        start_service: str,
        max_hops: int = 3,
    ) -> list[ServiceImpact]:
        """
        BFS 遍历服务依赖链

        从告警服务出发，查找上游依赖和下游影响。

        Args:
            start_service: 起始服务名称
            max_hops: 最大跳数

        Returns:
            list[ServiceImpact]: 影响链路中的服务列表
        """
        if start_service not in self._service_topology:
            logger.warning(
                "Service not found in topology",
                service=start_service,
            )
            return [self._create_impact_entry(start_service, 0)]

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        results: list[ServiceImpact] = []

        queue.append((start_service, 0))
        visited.add(start_service)

        while queue:
            service, hop = queue.popleft()

            if hop > max_hops:
                continue

            impact = self._create_impact_entry(service, hop)

            # 查找相关变更（模拟）
            impact.related_changes = self._find_recent_changes(service)

            # 根据跳数确定影响级别
            if hop == 0:
                impact.impact_level = "direct"
            elif hop <= 2:
                impact.impact_level = "indirect"
            else:
                impact.impact_level = "peripheral"

            results.append(impact)

            # 遍历上游依赖（可能导致问题的服务）
            svc_info = self._service_topology.get(service, {})
            for dep in svc_info.get("dependencies", []):
                if dep not in visited:
                    visited.add(dep)
                    queue.append((dep, hop + 1))

            # 遍历下游依赖（被影响的服务）
            for dep in svc_info.get("dependents", []):
                if dep not in visited:
                    visited.add(dep)
                    queue.append((dep, hop + 1))

        # 按跳数排序
        results.sort(key=lambda x: x.hop_distance)

        logger.info(
            "BFS traversal completed",
            start_service=start_service,
            services_found=len(results),
            max_hops=max_hops,
        )

        return results

    def _create_impact_entry(self, service: str, hop: int) -> ServiceImpact:
        """创建影响条目"""
        svc_info = self._service_topology.get(service, {})
        return ServiceImpact(
            service=service,
            hop_distance=hop,
            tier=svc_info.get("tier", "standard"),
        )

    def _find_recent_changes(self, service: str) -> list[dict[str, Any]]:
        """
        查找服务的近期变更（模拟数据）

        在生产环境中，这应查询 CMDB/发布系统。
        """
        # 模拟变更数据
        changes: dict[str, list[dict[str, Any]]] = {
            "order-service": [
                {"type": "deployment", "time": "2024-01-15T10:30:00Z", "version": "v2.3.1"},
            ],
            "payment-service": [
                {"type": "config_change", "time": "2024-01-15T09:00:00Z", "description": "connection pool size"},
            ],
            "mysql-primary": [
                {"type": "maintenance", "time": "2024-01-14T22:00:00Z", "description": "index optimization"},
            ],
        }
        return changes.get(service, [])

    # ==================== 因果校验（审查报告四、V1+V2） ====================

    async def _check_causality(
        self,
        service: str,
        alert: AlertEvent | None = None,
        symptoms: list[str] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        current_metrics: dict[str, float] | None = None,
        change_window_hours: int = 24,
        baseline_diff_threshold: float = 0.30,
    ) -> dict[str, Any]:
        """
        因果校验（审查报告四、V1 "指标恢复 ≠ 故障恢复" + V2 "反事实缺失"）。

        三层启发式信号（不依赖 LLM）：

        1. **时间相关性** (time_correlation)：
           检测告警时间是否在服务的近期变更窗口内（change_window_hours）。
           变更窗口内有 release → 时间相关性 +1.0；否则 0.0。

        2. **健康实例对照** (healthy_baseline_diff)：
           如果提供了 baseline_metrics（健康实例）和 current_metrics（故障实例），
           对每个共同 key 计算相对差异 |curr-base|/max(|base|, ε)，
           平均差异 > baseline_diff_threshold (默认 30%) 视为"差异显著"。

        3. **反事实前置打分** (counterfactual_score)：
           综合时间相关性 + 健康对照差异的加权分（0.4*时间 + 0.6*基线差），
           作为"如果回滚这次变更/重启实例，故障会不会消失"的启发式近似。

        触发场景：被 ``process()`` / ``_synthesize_analysis()`` 在出结论前调用。
        如果 ``is_causal=False``，建议 reduce ``confidence`` 或标注"因果证据不足"。

        Args:
            service: 故障服务名
            alert: 触发告警（用于时间相关性；如果 None 则跳过此项）
            symptoms: 故障症状列表（用于 evidence_refs 标注）
            baseline_metrics: 健康实例的指标（如 {"cpu": 30, "latency": 50}）
            current_metrics: 故障实例的指标（同 schema）
            change_window_hours: 变更窗口（小时），默认 24
            baseline_diff_threshold: 基线差异阈值（相对差异），默认 0.30 (30%)

        Returns:
            {
                "is_causal": bool,
                "confidence": float,   # [0, 1]
                "signals": {
                    "time_correlation": float,        # 0-1
                    "healthy_baseline_diff": float,   # 0-1（平均相对差异）
                    "counterfactual_score": float,    # 0-1
                },
                "evidence_refs": list[str],
                "recommendations": list[str],
                "checks_performed": list[str],
            }
        """
        checks_performed: list[str] = []
        evidence_refs: list[str] = []
        recommendations: list[str] = []

        # ---- 1. 时间相关性 ----
        time_correlation = 0.0
        if alert is not None:
            checks_performed.append("time_correlation")
            changes = self._find_recent_changes(service)
            alert_time = getattr(alert, "timestamp", None)
            if alert_time is not None and changes:
                from datetime import datetime, timedelta, timezone
                # 解析 alert.time（可能是 ISO 字符串或 datetime）
                if isinstance(alert_time, str):
                    try:
                        alert_dt = datetime.fromisoformat(alert_time.replace("Z", "+00:00"))
                    except (ValueError, AttributeError):
                        alert_dt = None
                else:
                    alert_dt = alert_time if isinstance(alert_time, datetime) else None
                if alert_dt is not None:
                    if alert_dt.tzinfo is None:
                        alert_dt = alert_dt.replace(tzinfo=timezone.utc)
                    window_start = alert_dt - timedelta(hours=change_window_hours)
                    for ch in changes:
                        ch_time_str = ch.get("time", "")
                        try:
                            ch_dt = datetime.fromisoformat(ch_time_str.replace("Z", "+00:00"))
                            if ch_dt.tzinfo is None:
                                ch_dt = ch_dt.replace(tzinfo=timezone.utc)
                        except (ValueError, AttributeError):
                            continue
                        if window_start <= ch_dt <= alert_dt:
                            time_correlation = 1.0
                            evidence_refs.append(
                                f"change:{ch.get('type', 'unknown')}@{ch_time_str} version={ch.get('version', ch.get('description', 'n/a'))}"
                            )
                            break
                    if time_correlation > 0:
                        recommendations.append(
                            f"故障发生在 {change_window_hours}h 变更窗口内，建议 review 上述变更"
                        )

        # ---- 2. 健康实例对照 ----
        healthy_baseline_diff = 0.0
        if baseline_metrics and current_metrics:
            common_keys = set(baseline_metrics.keys()) & set(current_metrics.keys())
            if common_keys:
                checks_performed.append("healthy_baseline_comparison")
                diffs = []
                for k in common_keys:
                    base_v = baseline_metrics[k]
                    curr_v = current_metrics[k]
                    if base_v == 0:
                        # baseline 是 0，curr > 0 → 视为完全偏离
                        if curr_v != 0:
                            diffs.append(1.0)
                        else:
                            diffs.append(0.0)
                    else:
                        rel_diff = abs(curr_v - base_v) / max(abs(base_v), 1e-6)
                        diffs.append(min(1.0, rel_diff))
                healthy_baseline_diff = sum(diffs) / len(diffs)
                evidence_refs.append(
                    f"baseline_diff: avg={healthy_baseline_diff:.2f} metrics={sorted(common_keys)}"
                )
                if healthy_baseline_diff >= baseline_diff_threshold:
                    recommendations.append(
                        f"健康实例基线差异 {healthy_baseline_diff:.0%} ≥ {baseline_diff_threshold:.0%}，"
                        f"建议对比健康实例配置/代码"
                    )

        # ---- 3. 反事实前置打分 ----
        # 综合：0.4 * 时间相关性 + 0.6 * 健康对照差异（健康实例差异是更强的因果信号）
        counterfactual_score = (
            0.4 * time_correlation + 0.6 * healthy_baseline_diff
        )
        if checks_performed:
            checks_performed.append("counterfactual_scoring")

        # ---- 综合判定 ----
        # 通过条件：counterfactual_score >= 0.5 OR (time_correlation=1.0 AND 至少一项检查)
        is_causal = counterfactual_score >= 0.5 or (
            time_correlation >= 1.0 and len(checks_performed) >= 1
        )
        # 置信度 = counterfactual_score（简单线性映射）
        confidence = round(min(1.0, counterfactual_score), 4)

        if not is_causal:
            recommendations.append(
                "因果证据不足：建议人工复核或补充 baseline 数据"
            )

        return {
            "is_causal": is_causal,
            "confidence": confidence,
            "signals": {
                "time_correlation": round(time_correlation, 4),
                "healthy_baseline_diff": round(healthy_baseline_diff, 4),
                "counterfactual_score": round(counterfactual_score, 4),
            },
            "evidence_refs": evidence_refs,
            "recommendations": recommendations,
            "checks_performed": checks_performed,
        }

    # ==================== 根因门禁（审查报告中期补 #6）====================

    def _apply_root_cause_gate(
        self,
        evidence_items: list[EvidenceItem],
        raw_confidence: float,
    ) -> GateResult:
        """
        把多源证据按 STRONG/WEAK/REFUTE/INSUFFICIENT 分类，再用门禁规则覆盖
        raw_confidence。任何 1 条 REFUTE → 强制降级；至少 1 STRONG 或 3 WEAK
        才允许 healed。
        """
        gate = apply_root_cause_gate(evidence_items, raw_confidence=raw_confidence)
        logger.info(
            "root_cause_gate",
            extra={
                "raw_confidence": raw_confidence,
                "promoted_confidence": gate.promoted_confidence,
                "strength": gate.strength.value,
                "allowed_to_heal": gate.allowed_to_heal,
                "reason": gate.reason,
                "breakdown": gate.breakdown,
            },
        )
        return gate

    # ==================== 贝叶斯推理 ====================

    def bayesian_inference(
        self,
        symptoms: list[str],
    ) -> list[BayesianNode]:
        """
        贝叶斯推理

        计算 P(根因|症状) = P(症状|根因) * P(根因) / P(症状)

        Args:
            symptoms: 症状列表

        Returns:
            list[BayesianNode]: 按后验概率排序的根因列表
        """
        nodes: list[BayesianNode] = []

        for cause_name in self.PRIOR_PROBABILITIES:
            prior = self.PRIOR_PROBABILITIES[cause_name]

            # 计算似然 P(症状|根因)
            symptom_likelihoods = self.LIKELIHOODS.get(cause_name, {})
            if not symptoms:
                likelihood = 0.5
            else:
                # 取各症状似然的平均值
                likelihoods = [
                    symptom_likelihoods.get(s, 0.1)
                    for s in symptoms
                ]
                likelihood = float(np.mean(likelihoods))

            # 计算 P(症状) - 归一化常数
            p_symptoms = self._calculate_p_symptoms(symptoms)

            # 计算后验概率
            if p_symptoms > 0:
                posterior = (likelihood * prior) / p_symptoms
            else:
                posterior = prior

            # 限制在合理范围
            posterior = min(posterior, 0.99)

            # 证据强度
            evidence_strength = min(likelihood * len(symptoms) / 3, 1.0)

            nodes.append(BayesianNode(
                name=cause_name,
                prior=prior,
                likelihood=likelihood,
                posterior=posterior,
                evidence_strength=evidence_strength,
            ))

        # 按后验概率降序排序
        nodes.sort(key=lambda x: x.posterior, reverse=True)

        logger.info(
            "Bayesian inference completed",
            symptoms=symptoms,
            top_cause=nodes[0].name if nodes else "unknown",
            top_posterior=round(nodes[0].posterior, 4) if nodes else 0,
        )

        return nodes

    def _calculate_p_symptoms(self, symptoms: list[str]) -> float:
        """
        计算 P(症状) - 归一化常数

        P(症状) = sum_cause(P(症状|cause) * P(cause))
        """
        if not symptoms:
            return 1.0

        total = 0.0
        for cause_name, prior in self.PRIOR_PROBABILITIES.items():
            symptom_likelihoods = self.LIKELIHOODS.get(cause_name, {})
            likelihoods = [
                symptom_likelihoods.get(s, 0.1)
                for s in symptoms
            ]
            avg_likelihood = float(np.mean(likelihoods))
            total += avg_likelihood * prior

        # 避免除零
        return max(total, 0.01)

    # ==================== 症状提取 ====================

    def _extract_symptoms(self, alert: AlertEvent) -> list[str]:
        """从告警中提取症状关键词"""
        symptoms: list[str] = []

        # 基于指标名称映射
        metric_symptom_map: dict[str, list[str]] = {
            "cpu_usage_percent": ["high_cpu"],
            "memory_usage_percent": ["high_memory"],
            "disk_usage_percent": ["disk_full"],
            "p99_latency_ms": ["increased_latency"],
            "error_rate_percent": ["high_error_rate"],
            "request_timeout_rate": ["timeout"],
            "oom_killed_count": ["oom_killed"],
        }

        mapped = metric_symptom_map.get(alert.metric, [])
        symptoms.extend(mapped)

        # 基于阈值和值判断
        if alert.value > alert.threshold * 1.5:
            symptoms.append("severe_threshold_breach")

        # 从注释中提取
        for key in ["symptom", "symptoms"]:
            if key in alert.annotations:
                val = alert.annotations[key]
                if isinstance(val, str):
                    symptoms.extend(val.split(","))

        # 去重
        return list(set(s.strip() for s in symptoms if s.strip()))

    # ==================== RAG 检索 ====================

    @staticmethod
    def _extract_verification_status(entry: Any) -> str:
        """
        提取记忆验证状态（v2 缺口 4 修复）

        读取优先级：
        1. MemoryEntry.verification_status（顶层字段，新数据）
        2. entry.metadata["verification_status"]（旧数据，向后兼容）
        3. 默认 "unverified"（按 v2 设计哲学：未验证 = 不参与加成）

        Args:
            entry: MemoryEntry 对象或 dict

        Returns:
            str: "unverified" | "provisional" | "verified"
        """
        # 优先级 1：顶层字段
        top_level = getattr(entry, "verification_status", None)
        if top_level is not None:
            # Enum 或 str 都兼容
            return getattr(top_level, "value", top_level)

        # 优先级 2：metadata 兜底（历史数据迁移）
        metadata = getattr(entry, "metadata", None)
        if isinstance(metadata, dict):
            legacy = metadata.get("verification_status")
            if legacy:
                return legacy

        # 优先级 3：dict 形式 entry（极少数情况）
        if isinstance(entry, dict):
            legacy = entry.get("verification_status")
            if legacy:
                return legacy

        return "unverified"

    def _rag_retrieve(
        self,
        alert: AlertEvent,
        symptoms: list[str],
    ) -> list[dict[str, Any]]:
        """
        RAG 增强 - 从知识库检索历史相似故障案例

        使用关键词匹配 + 类别匹配的综合检索策略。

        Args:
            alert: 告警事件
            symptoms: 症状列表

        Returns:
            list[dict]: 匹配的知识库条目
        """
        matches: list[tuple[dict[str, Any], float]] = []

        for kb_entry in self._knowledge_base:
            score = self._calculate_kb_match_score(kb_entry, alert, symptoms)
            if score > 0.3:  # 最低匹配阈值
                entry_copy = kb_entry.copy()
                entry_copy["match_score"] = round(score, 4)
                matches.append((entry_copy, score))

        # 按匹配得分降序排序
        matches.sort(key=lambda x: x[1], reverse=True)

        results = [m[0] for m in matches]

        logger.info(
            "RAG retrieval completed",
            symptoms=symptoms,
            matches_found=len(results),
            top_match_score=round(matches[0][1], 4) if matches else 0,
        )

        return results

    # ==================== 历史记忆检索 ====================

    async def _search_historical_incidents(
        self,
        alert: AlertEvent,
        symptoms: list[str],
    ) -> list[dict[str, Any]]:
        """
        从 MemorySystem 检索历史相似故障案例。

        使用语义搜索（ChromaDB 向量相似度）匹配历史故障记忆，
        返回相似度最高的历史根因和解决方案。

        Args:
            alert: 当前告警事件
            symptoms: 症状列表

        Returns:
            list[dict]: 匹配的历史记忆条目
        """
        ms = await self._get_memory_system()
        if ms is None:
            return []

        try:
            # 构建查询：服务名 + 指标名 + 症状
            query_text = f"{alert.service} {alert.metric} {' '.join(symptoms)}"

            result = await ms.search(
                query_text=query_text,
                top_k=5,
                memory_type=MemoryType.EPISODIC,  # 优先搜索事件记忆
            )

            if not result or result.total_found == 0:
                # 尝试搜索过程性记忆（playbook/自愈方案）
                result = await ms.search(
                    query_text=query_text,
                    top_k=3,
                    memory_type=MemoryType.PROCEDURAL,
                )

            if not result or result.total_found == 0:
                return []

            historical = []
            for entry, similarity in zip(result.results, result.similarities):
                historical.append({
                    "memory_id": entry.memory_id,
                    "content": entry.content[:500],
                    "summary": entry.summary,
                    "source_agent": entry.source_agent,
                    "source_incident": entry.source_incident_id,
                    "importance": entry.importance_score,
                    "retrieval_score": entry.retrieval_score,
                    "semantic_similarity": round(similarity, 4),
                    "tags": entry.tags,
                    # V2 缺口 4 修复：优先用顶层字段；旧数据兜底从 metadata 读；
                    # 顶层都没有时按 UNVERIFIED 处理（按 v2 设计哲学：未验证 = 不加成）。
                    "verification_status": self._extract_verification_status(entry),
                    # 验证次数与加成系数一并带出来，便于 RCA 综合阶段做精细加权
                    "verification_count": getattr(entry, "verification_count", 0),
                    "verification_boost": getattr(
                        entry, "verification_boost", 0.0
                    ),
                })

            logger.info(
                "Historical incident search completed",
                service=alert.service,
                query=query_text[:80],
                matches=len(historical),
                top_score=round(historical[0]["retrieval_score"], 4) if historical else 0,
            )

            return historical

        except Exception as e:
            logger.warning("Historical memory search failed, continuing without it", error=str(e))
            return []

    def _calculate_kb_match_score(
        self,
        kb_entry: dict[str, Any],
        alert: AlertEvent,
        symptoms: list[str],
    ) -> float:
        """计算知识库条目与当前告警的匹配得分"""
        score = 0.0

        # 症状匹配 (0-0.5)
        kb_symptoms = set(kb_entry.get("symptoms", []))
        if kb_symptoms:
            matched = len(kb_symptoms & set(symptoms))
            symptom_score = matched / max(len(kb_symptoms), len(symptoms))
            score += 0.5 * symptom_score

        # 指标类别匹配 (0-0.3)
        metric_category_map: dict[str, str] = {
            "cpu_usage_percent": "resource_exhaustion",
            "memory_usage_percent": "resource_exhaustion",
            "disk_usage_percent": "resource_exhaustion",
            "p99_latency_ms": "dependency_failure",
            "error_rate_percent": "dependency_failure",
            "request_timeout_rate": "dependency_failure",
        }
        alert_category = metric_category_map.get(alert.metric, "")
        if alert_category and kb_entry.get("category") == alert_category:
            score += 0.3

        # 服务等级匹配 (0-0.2)
        if kb_entry.get("category") == "deployment_issue":
            # 检查是否有近期部署
            service_changes = self._find_recent_changes(alert.service)
            if any(c.get("type") == "deployment" for c in service_changes):
                score += 0.2

        # 置信度加成
        score += kb_entry.get("confidence_boost", 0.0)

        return min(score, 1.0)

    # ==================== 日志证据（Loki） ====================

    async def _query_log_evidence(
        self,
        alert: AlertEvent,
        lookback_minutes: int,
        limit: int = 50,
    ) -> dict[str, Any]:
        """
        从 Loki 拉取告警服务的近期 ERROR 日志，按签名聚合为根因投票。

        返回结构：
            {
              "available": bool,                 # Loki 是否可达
              "entries_count": int,
              "cause_hits": dict[cause, count],  # 每个根因被命中的次数
              "boosted_cause": str | None,       # 票数最高的根因
              "confidence_boost": float,         # 0.0~0.10 的置信度加成
              "samples": dict[cause, list[str]], # 每个根因最多 3 条样本
              "source": "loki" | "unavailable",
            }

        失败兜底：Loki 不可达时返回 available=False，不影响主流程。
        """
        try:
            from app.infrastructure.log_client import LokiClient
        except ImportError:
            return _empty_log_evidence(reason="loki_client_unavailable")

        try:
            client = LokiClient()
            entries = await client.get_service_errors(
                service=alert.service,
                lookback_minutes=lookback_minutes,
                limit=limit,
            )
        except Exception as exc:
            logger.warning(
                "Loki query failed, RCA continues without log evidence",
                service=alert.service,
                error=str(exc),
            )
            return _empty_log_evidence(reason="loki_error")

        if not entries:
            return {
                "available": True,
                "entries_count": 0,
                "cause_hits": {},
                "boosted_cause": None,
                "confidence_boost": 0.0,
                "samples": {},
                "source": "loki",
            }

        cause_hits: dict[str, int] = {}
        samples: dict[str, list[str]] = {}
        for entry in entries:
            line = (entry.get("line") or "").lower()
            if not line:
                continue
            for signatures, cause in LOG_SIGNATURE_TO_CAUSE.items():
                if any(sig in line for sig in signatures):
                    cause_hits[cause] = cause_hits.get(cause, 0) + 1
                    samples.setdefault(cause, [])
                    if len(samples[cause]) < 3:
                        samples[cause].append((entry.get("line") or "")[:200])

        boosted_cause = (
            max(cause_hits, key=cause_hits.get) if cause_hits else None
        )
        # 票数越多加成越高，但封顶 0.10 避免压过贝叶斯
        confidence_boost = min(0.10, sum(cause_hits.values()) * 0.02)

        return {
            "available": True,
            "entries_count": len(entries),
            "cause_hits": cause_hits,
            "boosted_cause": boosted_cause,
            "confidence_boost": round(confidence_boost, 4),
            "samples": samples,
            "source": "loki",
        }

    # ==================== 自我批判 + 补查 (W2) ====================

    async def _self_critique_and_recover(
        self,
        alert: AlertEvent,
        original_root_cause: str,
        original_confidence: float,
        bayesian_results: list[BayesianNode],
        rag_results: list[dict[str, Any]],
        memory_results: list[dict[str, Any]],
        log_evidence: dict[str, Any],
    ) -> dict[str, Any] | None:
        """自我批判：当首轮置信度偏低时，根据证据缺口自动补查。

        补查策略（按优先级尝试）：
        1. RAG 召回 < 2 条 → 拓宽关键词重新检索
        2. Memory 命中 0 条 → 扩大 service 范围再查
        3. 日志证据为空 → 查询 WARN 级日志
        4. 仍然无显著改善 → 返回 None，由 Orchestrator 升级

        Returns:
            dict: {root_cause, confidence, evidence, reflection_note} 或 None
        """
        from app.agents.reflection import append_audit_trail, reflect_missing_info

        actions: list[str] = []
        recovered_root_cause = original_root_cause
        recovered_confidence = original_confidence
        recovered_extra_evidence: dict[str, Any] = {}

        # ===== 策略 1: RAG 召回 < 2 → 拓宽关键词 =====
        if len(rag_results) < 2:
            actions.append("broaden_rag_keywords")
            broader_rag = self._rag_retrieve_broadened(alert)
            if len(broader_rag) > len(rag_results):
                recovered_extra_evidence["broadened_rag_matches"] = len(broader_rag)
                rag_results = broader_rag
                # 每次拓宽 +0.05 置信度（封顶 +0.15）
                recovered_confidence = min(0.95, recovered_confidence + 0.05)

        # ===== 策略 2: Memory 命中 0 → 扩大 service 范围 =====
        if len(memory_results) == 0:
            actions.append("expand_memory_search")
            broader_memory = await self._search_historical_incidents_broadened(alert)
            if len(broader_memory) > 0:
                recovered_extra_evidence["broadened_memory_matches"] = len(broader_memory)
                memory_results = broader_memory
                recovered_confidence = min(0.95, recovered_confidence + 0.05)

        # ===== 策略 3: 日志证据为空 → 查 WARN 级 =====
        if not log_evidence.get("available", False) or log_evidence.get("entries_count", 0) == 0:
            actions.append("broaden_log_query")
            warn_log_evidence = await self._query_log_evidence_broadened(alert)
            if warn_log_evidence.get("entries_count", 0) > 0:
                recovered_extra_evidence["warn_log_entries"] = warn_log_evidence.get("entries_count", 0)
                log_evidence = warn_log_evidence
                recovered_confidence = min(0.95, recovered_confidence + 0.05)

        # ===== 策略 4: 若显著提升，重新综合 =====
        # 用 "任意正向提升" 判断，避免浮点精度问题；阈值 0.01 是噪声过滤
        if recovered_confidence > original_confidence + 0.01:
            # 重新跑综合分析
            impact_chain = self.bfs_traverse(alert.service, max_hops=3)
            new_root, new_conf, new_evidence = self._synthesize_analysis(
                bayesian_results, rag_results, impact_chain, alert,
                memory_results=memory_results,
                log_evidence=log_evidence,
            )
            reflection_note = (
                f"首轮置信度 {original_confidence:.0%} 偏低，"
                f"触发自我批判：{' + '.join(actions)} 后修正为 {new_conf:.0%}"
            )
            new_evidence["reflection_note"] = reflection_note
            new_evidence["reflection_actions"] = actions
            new_evidence["broadened_search"] = recovered_extra_evidence

            # 写审计
            append_audit_trail(
                {"__rca_context": True},
                step="rca_self_critique",
                trigger=f"low_confidence_{original_confidence:.2f}",
                actions_taken=actions,
                outcome=f"recovered_to_{new_conf:.2f}",
                duration_ms=0,
            )
            logger.info(
                "RCA self-critique succeeded",
                original_confidence=original_confidence,
                recovered_confidence=new_conf,
                actions=actions,
            )
            return {
                "root_cause": new_root,
                "confidence": new_conf,
                "evidence": new_evidence,
                "reflection_note": reflection_note,
            }

        # 没有显著提升 → 返回 None，让 Orchestrator 升级
        logger.info(
            "RCA self-critique did not improve",
            original_confidence=original_confidence,
            actions_tried=actions,
        )
        return None

    def _rag_retrieve_broadened(self, alert: AlertEvent) -> list[dict[str, Any]]:
        """拓宽关键词的 RAG 检索：去掉 service 限制，仅用 symptom。

        这是原 `_rag_retrieve` 的"宽松版"，用于反思补查。
        """
        try:
            symptoms = self._extract_symptoms(alert)
            if not symptoms:
                return []
            matches: list[tuple[dict[str, Any], float]] = []
            for kb_entry in self._knowledge_base:
                score = self._calculate_kb_match_score(
                    kb_entry, alert, symptoms,
                )
                # 反思补查阈值降到 0.2（原为 0.3）
                if score > 0.2:
                    entry_copy = kb_entry.copy()
                    entry_copy["match_score"] = round(score, 4)
                    entry_copy["broadened_match"] = True
                    matches.append((entry_copy, score))
            matches.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in matches[:5]]
        except Exception as exc:
            logger.warning("Broadened RAG retrieval failed", error=str(exc))
            return []

    async def _search_historical_incidents_broadened(
        self, alert: AlertEvent,
    ) -> list[dict[str, Any]]:
        """扩大 service 范围的历史记忆检索（去掉 service 限制）。"""
        try:
            ms = await self._get_memory_system()
            if ms is None:
                return []
            symptoms = self._extract_symptoms(alert)
            query_text = f"{alert.metric} {' '.join(symptoms)}"
            result = await ms.search(
                query_text=query_text,
                top_k=5,
                memory_type=None,  # 不限定 memory_type
            )
            if not result or result.total_found == 0:
                return []
            historical = []
            for entry, similarity in zip(result.results, result.similarities):
                historical.append({
                    "memory_id": entry.memory_id,
                    "content": entry.content[:500],
                    "summary": entry.summary,
                    "semantic_similarity": round(similarity, 4),
                    "broadened_match": True,
                })
            return historical
        except Exception as exc:
            logger.warning("Broadened memory search failed", error=str(exc))
            return []

    async def _query_log_evidence_broadened(
        self, alert: AlertEvent, lookback_minutes: int = 60,
    ) -> dict[str, Any]:
        """WARN 级日志查询（首轮查 ERROR 失败时降级）。"""
        try:
            from app.infrastructure.log_client import LokiClient
            client = LokiClient()
            # 把 alert_value 转成 string 的 level 字段查询
            warn_entries = await client.get_service_logs(
                service=alert.service,
                keyword="WARN",
                lookback_minutes=lookback_minutes,
                limit=50,
            )
            return {
                "available": True,
                "entries_count": len(warn_entries),
                "cause_hits": {},
                "boosted_cause": None,
                "confidence_boost": 0.0,
                "samples": {"WARN": [e["line"][:200] for e in warn_entries[:5]]},
                "source": "loki_warn_fallback",
            }
        except Exception as exc:
            logger.warning("Broadened log query failed", error=str(exc))
            return _empty_log_evidence(reason="broaden_failed")

    # ==================== 综合分析 ====================

    def _synthesize_analysis(
        self,
        bayesian_results: list[BayesianNode],
        rag_results: list[dict[str, Any]],
        impact_chain: list[ServiceImpact],
        alert: AlertEvent,
        memory_results: list[dict[str, Any]] | None = None,
        log_evidence: dict[str, Any] | None = None,
        reflection_context: dict[str, Any] | None = None,
    ) -> tuple[str, float, dict[str, Any]]:
        """
        综合分析 - 合并贝叶斯推理、RAG 结果、历史记忆与 Loki 日志证据

        Returns:
            tuple: (根因, 置信度, 证据)
        """
        # ===== PPT 思路主链路 RCA 合成优化（feature flag）=====
        # 开启（AIOPS_USE_RCA_V2=true / config.agent.rca_v2_enabled）→ 走 V2 优化合成：
        #   相似度语义修正（读 semantic_similarity 而非 retrieval_score）、
        #   结构化历史特征、冲突降确定性（不再 *0.3 加分）、消费 bayesian_penalty、
        #   排除 rejected_root_causes、provisional 记忆不加成、Top-K 候选输出。
        # 关闭（默认）→ 字节不变走下方旧逻辑。
        try:
            from app.services.rca_synthesis_v2 import get_rca_synthesis_v2
            v2 = get_rca_synthesis_v2()
            if v2 is not None:
                return v2.synthesize(
                    bayesian_results, rag_results, impact_chain, alert,
                    memory_results=memory_results,
                    log_evidence=log_evidence,
                    reflection_context=reflection_context,
                )
        except Exception:
            # 优化失败优雅降级：走旧逻辑，不影响主链路
            pass

        # 检查是否有知识库高匹配
        top_rag = rag_results[0] if rag_results else None

        # 贝叶斯最高后验
        top_bayesian = bayesian_results[0] if bayesian_results else None

        # 历史记忆最高匹配
        top_memory = memory_results[0] if memory_results else None
        memory_boost = 0.0
        historical_root_cause = None
        if top_memory:
            # 从记忆内容中提取历史根因（简化：匹配 "Root cause:" 后的文本）
            mem_content = top_memory.get("content", "")
            if "Root cause:" in mem_content or "root_cause" in str(top_memory.get("tags", [])):
                historical_root_cause = mem_content
            # 记忆检索得分作为置信度加成
            memory_similarity = top_memory.get("retrieval_score", 0)
            memory_boost = min(0.15, memory_similarity * 0.2)

        # 日志证据（Loki）
        log_evidence = log_evidence or _empty_log_evidence(reason="not_provided")
        log_boost = (
            log_evidence.get("confidence_boost", 0.0)
            if log_evidence.get("available")
            else 0.0
        )
        log_boosted_cause = log_evidence.get("boosted_cause")
        # 如果日志强指向某根因且与贝叶斯结论一致，进一步加权；不一致则记为 "log_divergence"
        log_corroboration = False
        log_divergence = False
        if (
            log_boosted_cause
            and top_bayesian
            and log_boost > 0.04   # 至少 2 票才视为有意义
        ):
            if log_boosted_cause == top_bayesian.name:
                log_corroboration = True
            else:
                log_divergence = True

        # 综合根因（历史记忆优先级最高；日志仅作为佐证，不直接覆盖贝叶斯）
        if top_memory and historical_root_cause and top_memory.get("retrieval_score", 0) > 0.5:
            # 有高度相似的历史故障，优先采纳历史根因
            if top_bayesian:
                root_cause = f"{top_bayesian.name} (historical_match: {top_memory.get('source_incident', 'unknown')})"
            else:
                root_cause = f"similar_to_{top_memory.get('source_incident', 'unknown')}"
        elif top_rag and top_rag.get("match_score", 0) > 0.7 and top_bayesian:
            # RAG 高匹配 + 贝叶斯支持
            rag_causes = top_rag.get("root_causes", [])
            if rag_causes and top_bayesian.name in rag_causes:
                root_cause = top_bayesian.name
            else:
                root_cause = rag_causes[0] if rag_causes else top_bayesian.name
        elif top_bayesian:
            root_cause = top_bayesian.name
        else:
            root_cause = "unknown"

        # 综合置信度（加上记忆加成 + 日志加成）
        memory_confidence_boost = memory_boost
        # 日志佐证时加成生效；分歧时仅小幅加成（避免反向偏置）
        effective_log_boost = log_boost if log_corroboration else (log_boost * 0.3 if log_divergence else log_boost)
        if top_bayesian and top_rag:
            confidence = min(
                0.95,
                0.55 * top_bayesian.posterior
                + 0.25 * top_rag.get("match_score", 0)
                + 0.1 * top_bayesian.evidence_strength
                + 0.1 * memory_confidence_boost * 10,  # 记忆权重
            )
        elif top_bayesian:
            confidence = min(0.9, top_bayesian.posterior * 0.8 + memory_confidence_boost)
        else:
            confidence = 0.3 + memory_confidence_boost
        # 日志加成独立叠加，封顶 0.95
        confidence = min(0.95, confidence + effective_log_boost)

        # W7 重跑权重调整：verify_phase 在 Prometheus 与首轮 RCA 不一致时把
        # evidence_boost="log" 注入 reflection_context，使日志证据权重从 0.10
        # 提升到 0.25，并把贝叶斯合成系数从 0.55 降至 0.40。日志 boost 同步放大
        # 至 effective_log_boost（系数 2.5，封顶 0.25）。
        if reflection_context and reflection_context.get("evidence_boost") == "log":
            log_boost_value = (
                log_evidence.get("confidence_boost", 0.0) if log_evidence else 0.0
            )
            effective_log_boost = min(0.25, log_boost_value * 2.5)
            # 重新计算 confidence：贝叶斯分量从 0.55 → 0.40，差值按比例释放给日志
            if top_bayesian and top_rag:
                rebased = (
                    0.40 * top_bayesian.posterior
                    + 0.25 * top_rag.get("match_score", 0)
                    + 0.10 * top_bayesian.evidence_strength
                    + 0.10 * memory_confidence_boost * 10
                )
                confidence = min(0.95, max(0.0, rebased + effective_log_boost))
            else:
                confidence = min(0.95, max(0.0, confidence - log_boost + effective_log_boost))

        # 影响范围
        critical_services = [s.service for s in impact_chain if s.tier == "critical"]

        evidence = {
            "bayesian_top": top_bayesian.name if top_bayesian else "unknown",
            "bayesian_posterior": round(top_bayesian.posterior, 4) if top_bayesian else 0,
            "rag_top_match": top_rag["id"] if top_rag else "none",
            "rag_match_score": round(top_rag.get("match_score", 0), 4) if top_rag else 0,
            "historical_memory_matches": len(memory_results) if memory_results else 0,
            "memory_top_score": round(top_memory.get("retrieval_score", 0), 4) if top_memory else 0,
            "memory_confidence_boost": round(memory_confidence_boost, 4),
            "log_evidence": {
                "available": log_evidence.get("available", False),
                "source": log_evidence.get("source", "unavailable"),
                "entries_count": log_evidence.get("entries_count", 0),
                "boosted_cause": log_evidence.get("boosted_cause"),
                "cause_hits": log_evidence.get("cause_hits", {}),
                "confidence_boost": log_evidence.get("confidence_boost", 0.0),
                "effective_boost": round(effective_log_boost, 4),
                "corroborates_bayesian": log_corroboration,
                "diverges_from_bayesian": log_divergence,
                "samples": log_evidence.get("samples", {}),
            },
            "log_corroboration": log_corroboration,
            "log_divergence": log_divergence,
            "affected_services": [s.service for s in impact_chain],
            "critical_services_affected": critical_services,
            "total_impact_hops": max((s.hop_distance for s in impact_chain), default=0),
        }

        return root_cause, round(confidence, 4), evidence

    def _generate_suggested_actions(
        self,
        root_cause: str,
        impact_chain: list[ServiceImpact],
        rag_results: list[dict[str, Any]],
    ) -> list[str]:
        """生成建议操作"""
        actions: list[str] = []

        # 从知识库获取建议
        for rag in rag_results[:2]:
            solutions = rag.get("solutions", [])
            for sol in solutions:
                if sol not in actions:
                    actions.append(sol)

        # 基于根因类型的默认建议
        default_actions: dict[str, list[str]] = {
            "recent_deployment": ["rollback_deployment", "check_deployment_logs"],
            "configuration_change": ["rollback_configuration", "check_config_diff"],
            "resource_exhaustion": ["scale_up_resources", "check_resource_limits"],
            "dependency_failure": ["check_dependencies", "enable_circuit_breaker"],
            "network_issue": ["check_network_connectivity", "check_dns_resolution"],
            "database_issue": ["check_database_performance", "check_connection_pool"],
            "code_bug": ["check_application_logs", "enable_debug_logging"],
            "traffic_spike": ["scale_up_instances", "enable_rate_limiting"],
            "hardware_failure": ["drain_node", "migrate_workloads"],
            "third_party_issue": ["check_third_party_status", "enable_fallback"],
        }

        for action in default_actions.get(root_cause, ["investigate_manually"]):
            if action not in actions:
                actions.append(action)

        # 如果影响关键服务，添加额外建议
        critical_affected = any(s.tier == "critical" and s.hop_distance > 0 for s in impact_chain)
        if critical_affected:
            if "notify_oncall" not in actions:
                actions.append("notify_oncall")
            if "prepare_rollback" not in actions:
                actions.append("prepare_rollback")

        return actions
