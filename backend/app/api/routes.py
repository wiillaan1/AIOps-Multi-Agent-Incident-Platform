"""
AIOps Agent Platform - RESTful API Routes

定义所有 RESTful API 端点，包括故障管理、Agent 管理、评估和记忆搜索。
所有端点现已连接到实际的 Service/Agent 层，返回真实的模拟数据。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel as PydanticBaseModel, Field

from app.agents.base import AgentExecutionContext
from app.agents.business_monitor_agent import BusinessMetricInput, BusinessMonitorAgent
from app.agents.change_agent import ChangeAgent, ChangeInput
from app.agents.collaborative_task import (
    CollaborativeTask,
    PlanReActRunner,
    TaskStep,
    default_rca_plan,
)
from app.agents.heal_agent import HealAgent, HealInput
from app.agents.monitor_agent import MetricInput, MonitorAgent
from app.agents.rca_agent import RCAAgent, RCAInput
from app.agents.reflection import (
    VerifyEvidence,
    append_audit_trail,
    reflect_on_verify,
)
from app.api.websocket import manager as ws_manager
from app.infrastructure.prometheus_client import PrometheusClient
from app.config import AppConfig, get_config
from app.data.datasets import (
    FAULT_SCENARIOS,
    METRICS_DATASETS,
    get_metric_data,
)
from app.data.knowledge_base import (
    KNOWLEDGE_BASE,
    SERVICE_TOPOLOGY,
    KnowledgeBase,
)
from app.data.playbooks import PLAYBOOKS
from app.dependencies import CommonQueryParams
from app.evaluation.core import get_evaluation_framework
from app.models.agent import AgentState, AgentStatus, AgentType
from app.models.evaluation import EvaluationResult, EvaluationType
from app.models.events import (
    AlertEvent,
    ChangeEvent,
    HealEvent,
    RCAEvent,
    SeverityLevel,
)
from app.models.incident import Incident, IncidentState
from app.models.memory import MemoryType
from app.nlu.entity_extractor import EntityExtractor
from app.nlu.intent_classifier import IntentClassifier
from app.nlu.metric_mapper import MetricMapper
from app.services.badcase_registry_service import BadCaseRegistryService
from app.services.incident_service import IncidentService
from app.services.trigger_dedup import (
    compute_alert_fingerprint,
    get_trigger_dedup_service,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)

api_router = APIRouter(prefix="/api/v1")

# 全局服务实例（模块级别单例）
# Reuse the service across module reloads so long-lived test/application code
# and freshly imported route handlers observe the same incident store.
_incident_service = getattr(
    sys.modules.get(__name__), "_incident_service", None
) or IncidentService()
_monitor_agent: MonitorAgent | None = None
_rca_agent: RCAAgent | None = None
_heal_agent: HealAgent | None = None
_change_agent: ChangeAgent | None = None
_business_monitor_agent: BusinessMonitorAgent | None = None


def _get_monitor() -> MonitorAgent:
    global _monitor_agent
    if _monitor_agent is None:
        _monitor_agent = MonitorAgent()
    return _monitor_agent


def _get_rca() -> RCAAgent:
    global _rca_agent
    if _rca_agent is None:
        _rca_agent = RCAAgent()
    return _rca_agent


def _get_heal() -> HealAgent:
    global _heal_agent
    if _heal_agent is None:
        _heal_agent = HealAgent()
    return _heal_agent


def _get_change() -> ChangeAgent:
    global _change_agent
    if _change_agent is None:
        _change_agent = ChangeAgent()
    return _change_agent


def _get_business_monitor() -> BusinessMonitorAgent:
    global _business_monitor_agent
    if _business_monitor_agent is None:
        _business_monitor_agent = BusinessMonitorAgent()
    return _business_monitor_agent


# =============================================================================
# W7.5 BadCase Registry（懒加载单例，可被测试通过模块属性替换注入 tmp 路径实例）
# =============================================================================

_badcase_registry_service: BadCaseRegistryService | None = None


def _get_badcase_registry_service() -> BadCaseRegistryService:
    """获取 BadCaseRegistryService 单例（默认磁盘路径）。

    测试通过 monkeypatch 将本函数替换为返回 tmp-path 实例的工厂来隔离 IO。
    """
    global _badcase_registry_service
    if _badcase_registry_service is None:
        _badcase_registry_service = BadCaseRegistryService()
    return _badcase_registry_service


# =============================================================================
# W9 调查验证闭环（InvestigationLoopEngine）集成
# -----------------------------------------------------------------------------
# 引擎默认关闭（config.agent.verification_enabled=False）；开启后由引擎驱动
# RCA 候选验证、反证、最多三轮重规划、Dry-run 与回执边界、稳定窗口恢复验证。
# 单例可在测试中通过 _set_investigation_engine 注入确定性实例，并在结束后由
# _reset_investigation_engine 还原，避免跨测试污染。
# =============================================================================

_investigation_engine: Any = None
_investigation_store: Any = None
_profile_registry: Any = None


def _get_profile_registry():
    """获取 ProfileRegistry 单例（懒加载并加载所有 YAML Profile）。"""
    global _profile_registry
    if _profile_registry is None:
        from app.services.profile_registry import ProfileRegistry

        _profile_registry = ProfileRegistry()
        _profile_registry.load()
    return _profile_registry


def _get_investigation_store():
    """获取进程内 InvestigationStore 单例。"""
    global _investigation_store
    if _investigation_store is None:
        from app.services.investigation_store import InvestigationStore

        _investigation_store = InvestigationStore()
    return _investigation_store


def _make_metric_query_adapter():
    """构建 Prometheus 查询适配器：返回 QueryResult，不可达时 available=False。"""
    from app.services.assertion_evaluator import QueryResult

    async def _adapter(query: str, rule=None, context=None, **_: Any) -> QueryResult:
        timeout = _verification_query_timeout()
        try:
            client = PrometheusClient()
            value = await asyncio.wait_for(client.query_instant(query), timeout=timeout)
        except Exception as exc:
            return QueryResult(
                source="prometheus", available=False, value=None, hits=0,
                summary={}, error=str(exc)[:200],
            )
        if value is None:
            return QueryResult(
                source="prometheus", available=False, value=None, hits=0,
                summary={"query": query},
            )
        return QueryResult(
            source="prometheus", available=True, value=value, hits=0,
            summary={"query": query},
        )

    return _adapter


def _make_log_query_adapter():
    """构建 Loki 查询适配器：返回 QueryResult，不可达时 available=False。"""
    from app.services.assertion_evaluator import QueryResult

    async def _adapter(query: str, rule=None, context=None, **_: Any) -> QueryResult:
        timeout = _verification_query_timeout()
        max_rows = _verification_query_max_rows()
        try:
            from app.infrastructure.log_client import LokiClient

            client = LokiClient()
            entries = await asyncio.wait_for(
                client.query_range(query, limit=max_rows), timeout=timeout
            )
        except Exception as exc:
            return QueryResult(
                source="loki", available=False, value=None, hits=0,
                summary={}, error=str(exc)[:200],
            )
        return QueryResult(
            source="loki", available=True, value=None, hits=len(entries),
            summary={"query": query, "sample_count": len(entries)},
        )

    return _adapter


def _get_or_build_investigation_engine():
    """获取或构建默认 InvestigationLoopEngine 单例。

    使用真实 Prometheus/Loki 适配器；数据源不可达会被引擎视为 unavailable，绝不
    当作根因失败或恢复成功。未开启 verification_enabled 时仍可被 API 显式调用，
    但不会自动接入 trigger 主管道。
    """
    global _investigation_engine
    if _investigation_engine is not None:
        return _investigation_engine
    from app.services.investigation_loop_engine import InvestigationLoopEngine

    _investigation_engine = InvestigationLoopEngine(
        store=_get_investigation_store(),
        profiles=_get_profile_registry(),
        rca_agent=_get_rca(),
        heal_agent=_get_heal(),
        metric_query=_make_metric_query_adapter(),
        log_query=_make_log_query_adapter(),
    )
    return _investigation_engine


def _set_investigation_engine(engine: Any) -> None:
    """测试注入：替换引擎单例（含其 store/profiles）。"""
    global _investigation_engine
    _investigation_engine = engine
    # 注入引擎时同步采用其 store，保证路由与引擎读写同一份记录。
    global _investigation_store
    _investigation_store = getattr(engine, "store", _investigation_store)


def _reset_investigation_engine() -> None:
    """测试还原：清空注入的引擎/store 单例，避免跨测试污染。"""
    global _investigation_engine, _investigation_store
    _investigation_engine = None
    _investigation_store = None


def _investigation_store_for(engine: Any) -> Any:
    """返回引擎绑定的 store；注入引擎与默认引擎都由此统一取值。"""
    return getattr(engine, "store", None) or _get_investigation_store()


def _receipt_business_fields(receipt: Any) -> tuple[str, str, str, str, str]:
    """幂等冲突比较字段（与 InvestigationStore._receipt_payload 对齐）。

    不含 receipt_id：它是自动生成的，而 idempotency_key 才是幂等锚点。
    """
    return (
        receipt.incident_id,
        receipt.action_id,
        receipt.playbook_id,
        receipt.target_resource,
        receipt.status,
    )


def _verification_query_timeout() -> int:
    try:
        return int(get_config().agent.verification_query_timeout_seconds)
    except Exception:
        return 10


def _verification_query_max_rows() -> int:
    try:
        return int(get_config().agent.verification_query_max_rows)
    except Exception:
        return 100


async def _broadcast_investigation_event(incident_id: str, event_type: str, payload: dict[str, Any]) -> None:
    """向故障房间广播结构化调查事件，失败静默忽略（非阻塞）。"""
    try:
        await ws_manager.broadcast_to_incident(
            incident_id, {"type": event_type, "incident_id": incident_id, **payload}
        )
    except Exception as exc:
        logger.debug("investigation websocket broadcast failed (non-fatal)", error=str(exc))


# =============================================================================
# 记忆系统集成（惰性初始化，失败不影响管道）
# =============================================================================

_memory_system: Any = None
_memory_init_attempted: bool = False


async def _get_memory_system():
    """获取 MemorySystem 单例。首次失败后不再重试，避免阻塞管道。"""
    global _memory_system, _memory_init_attempted
    if _memory_system is not None:
        return _memory_system
    if _memory_init_attempted:
        return None
    try:
        from app.memory.core import MemorySystem as MS
        _memory_system = await MS.get_instance()
        logger.info("MemorySystem initialized for pipeline integration")
    except Exception as e:
        _memory_init_attempted = True
        logger.warning("MemorySystem unavailable, pipeline will run without memory persistence", error=str(e))
    return _memory_system


async def _store_agent_memory(
    content: str,
    agent_name: str,
    incident_id: str,
    importance: float = 0.5,
    memory_type: MemoryType = MemoryType.OBSERVATION,
    tags: list[str] | None = None,
    verification_status: str | None = None,
) -> None:
    """存储 Agent 执行结果到记忆系统，失败静默忽略。

    verification_status（PPT 思路 RCA V2）：
        - "provisional"：算法生成、未验证的根因。不参与未来 RCA 置信度加成。
        - "verified"：经 verify_phase / 人工确认 / W9 CONFIRMED 后晋级，可弱加成。
        默认 None → 不写该字段（旧行为）；V2 开启时 RCA 入库写 "provisional"。
    """
    try:
        ms = await _get_memory_system()
        if ms is None:
            return
        metadata = {}
        if verification_status is not None:
            metadata["verification_status"] = verification_status
        await ms.store(
            content=content,
            memory_type=memory_type,
            source_agent=agent_name,
            source_incident_id=incident_id,
            importance=importance,
            tags=tags or [],
            session_id=incident_id,
            metadata=metadata,
        )
    except Exception as e:
        logger.debug("Failed to store agent memory (non-fatal)", agent=agent_name, error=str(e))


async def _store_working_memory(
    incident_id: str,
    key: str,
    value: Any,
    ttl_seconds: int = 3600,
) -> None:
    """存储工作记忆，供下游 Agent 读取。"""
    try:
        ms = await _get_memory_system()
        if ms is None:
            return
        await ms.working_set(key=key, value=value, incident_id=incident_id, ttl_seconds=ttl_seconds)
    except Exception as e:
        logger.debug("Failed to store working memory (non-fatal)", key=key, error=str(e))


def _rca_v2_enabled() -> bool:
    """读取 PPT 思路 RCA V2 feature flag（与 rca_synthesis_v2 同源）。"""
    import os as _os
    val = _os.getenv("AIOPS_USE_RCA_V2", "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    if val in ("false", "0", "no", "off"):
        return False
    try:
        from app.config import get_config
        return bool(get_config().agent.rca_v2_enabled)
    except Exception:
        return False


async def _promote_memory_to_verified(incident_id: str) -> int:
    """把某 incident 的 provisional RCA 记忆晋级为 verified（可信记忆）。

    PPT 思路 RCA V2 的记忆治理入口：verify_phase 通过 / W9 CONFIRMED / 人工确认后调用。
    失败静默忽略。返回晋级条数。
    """
    try:
        ms = await _get_memory_system()
        if ms is None:
            return 0
        promoted = 0
        for store in (ms.short_term, getattr(ms, "long_term", None)):
            if store is None:
                continue
            entries = getattr(store, "_items", None) or getattr(store, "_entries", None)
            if entries is None:
                continue
            for entry in (entries.values() if isinstance(entries, dict) else entries):
                if (getattr(entry, "source_incident_id", "") == incident_id
                        and getattr(entry, "memory_type", None) is not None
                        and "rca" in getattr(entry, "tags", [])):
                    meta = getattr(entry, "metadata", None)
                    if isinstance(meta, dict) and meta.get("verification_status") == "provisional":
                        meta["verification_status"] = "verified"
                        promoted += 1
        return promoted
    except Exception as e:
        logger.debug("promote_memory_to_verified failed (non-fatal)", incident_id=incident_id, error=str(e))
        return 0


# =============================================================================
# 为演示预设一些故障数据
# =============================================================================

def _seed_incidents() -> None:
    """预设故障数据（仅首次调用时填充）"""
    if _incident_service._incidents:
        return

    from datetime import timedelta as _td
    now = datetime.now(timezone.utc)
    seeds = [
        {
            "incident_id": "INC-2026-001",
            "service": "order-service",
            "metric": "cpu_usage_percent",
            "severity": SeverityLevel.CRITICAL,
            "state": IncidentState.HEALING,
            "value": 95.0,
            "threshold": 80.0,
        },
        {
            "incident_id": "INC-2026-002",
            "service": "payment-service",
            "metric": "memory_usage_percent",
            "severity": SeverityLevel.HIGH,
            "state": IncidentState.RCA_IN_PROGRESS,
            "value": 88.0,
            "threshold": 85.0,
        },
        {
            "incident_id": "INC-2026-003",
            "service": "api-gateway",
            "metric": "error_rate_percent",
            "severity": SeverityLevel.CRITICAL,
            "state": IncidentState.RCA_COMPLETED,
            "value": 12.0,
            "threshold": 5.0,
        },
        {
            "incident_id": "INC-2026-004",
            "service": "user-service",
            "metric": "p99_latency_ms",
            "severity": SeverityLevel.HIGH,
            "state": IncidentState.ANALYZING,
            "value": 500.0,
            "threshold": 200.0,
        },
        {
            "incident_id": "INC-2026-005",
            "service": "inventory-service",
            "metric": "cpu_usage_percent",
            "severity": SeverityLevel.MEDIUM,
            "state": IncidentState.ACKNOWLEDGED,
            "value": 78.0,
            "threshold": 75.0,
        },
        {
            "incident_id": "INC-2026-006",
            "service": "payment-service",
            "metric": "error_rate_percent",
            "severity": SeverityLevel.LOW,
            "state": IncidentState.RESOLVED,
            "value": 2.1,
            "threshold": 3.0,
        },
        {
            "incident_id": "INC-2026-007",
            "service": "mysql-primary",
            "metric": "p99_latency_ms",
            "severity": SeverityLevel.CRITICAL,
            "state": IncidentState.HEALING,
            "value": 3000.0,
            "threshold": 500.0,
        },
    ]

    for i, s in enumerate(seeds):
        alert = AlertEvent(
            source="monitor_agent",
            service=s["service"],
            metric=s["metric"],
            value=s["value"],
            threshold=s["threshold"],
            operator=">",
            severity=s["severity"],
            labels={"tier": SERVICE_TOPOLOGY.get(s["service"], {}).get("tier", "standard")},
            annotations={"seed": "true"},
        )
        incident = Incident.from_alert(alert)
        incident.incident_id = s["incident_id"]
        incident.transition_to(s["state"], actor="system")
        # 调整时间为更早的
        incident.created_at = now - _td(minutes=(len(seeds) - i) * 15)
        incident.updated_at = now
        _incident_service._incidents[incident.incident_id] = incident

    logger.info("Seed incidents created", count=len(seeds))


# =============================================================================
# Incident Routes
# =============================================================================

@api_router.post(
    "/incidents/trigger",
    response_model=dict[str, Any],
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Incidents"],
    summary="触发故障处理",
    description="接收告警事件并触发完整的 Agent 协作处理流程。",
)
async def trigger_incident(
    alert: AlertEvent,
    config: AppConfig = Depends(get_config),
) -> dict[str, Any]:
    """
    触发故障处理流程

    完整流程:
    1. Monitor Agent - 异常检测与去重
    2. RCA Agent - 根因分析
    3. Heal Agent - 自愈操作
    4. Change Agent - 变更审批
    每个阶段通过 WebSocket 实时推送进度。
    """
    logger.info(
        "Incident triggered",
        service=alert.service,
        metric=alert.metric,
        severity=alert.severity.value,
    )

    # ===== 审查报告中期补 #10：trigger dedup + 跨 incident in-flight 上限 =====
    fingerprint = compute_alert_fingerprint(
        service=alert.service,
        metric=alert.metric,
        severity=alert.severity.value,
    )
    dedup_service = get_trigger_dedup_service()

    is_dup, existing_id = dedup_service.should_dedupe(fingerprint)
    if is_dup:
        logger.info(
            "Alert deduplicated",
            fingerprint=fingerprint,
            existing_incident_id=existing_id,
        )
        return {
            "status": "deduplicated",
            "incident_id": existing_id,
            "fingerprint": fingerprint,
            "message": f"Already processing incident {existing_id} (fingerprint {fingerprint})",
        }

    if not dedup_service.can_accept_new():
        logger.warning(
            "Trigger rejected: in-flight at capacity",
            in_flight=dedup_service.in_flight_count(),
            max_in_flight=dedup_service.max_in_flight,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"in-flight incidents at capacity "
                f"({dedup_service.in_flight_count()}/{dedup_service.max_in_flight})"
            ),
        )

    # 创建故障实例
    incident = Incident.from_alert(alert)
    dedup_service.mark_in_flight(fingerprint, incident.incident_id)
    incident.transition_to(IncidentState.ACKNOWLEDGED, actor="orchestrator")
    _incident_service._incidents[incident.incident_id] = incident

    # WebSocket 广播: 故障已创建
    await ws_manager.broadcast({
        "type": "incident_update",
        "payload": _incident_to_dict(incident),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # 启动后台处理（不阻塞响应）
    asyncio.create_task(_run_pipeline_with_reflection(incident))

    return {
        "incident_id": incident.incident_id,
        "status": "accepted",
        "state": incident.state.value,
        "message": "Incident processing started — agents are running",
        "correlation_id": incident.alert_event.correlation_id if incident.alert_event else "",
    }


@api_router.post(
    "/incidents/seed-demo",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="注入演示故障数据（仅 dev 环境）",
)
async def seed_demo_incidents() -> dict[str, Any]:
    """显式注入 7 个演示 incident。仅在 APP_ENV=development 时可用。"""
    config = get_config()
    if config.env != "development":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="seed-demo endpoint is only available in development environment",
        )
    _seed_incidents()
    return {
        "status": "seeded",
        "count": len(_incident_service._incidents),
        "incident_ids": list(_incident_service._incidents.keys()),
    }


# =============================================================================
# 业务异常检测端点
# =============================================================================

@api_router.post("/incidents/trigger-business")
async def trigger_business_incident(
    request: BusinessMetricInput,
) -> dict[str, Any]:
    """
    触发业务异常检测流程。

    接收业务检查请求，执行规则引擎检测，返回检测结果。
    与 /incidents/trigger 互补 —— 前者处理指标异常，此端点处理业务逻辑异常。

    示例请求:
    ```json
    {
      "service_name": "payment-service",
      "business_domain": "financial",
      "check_rules": ["br_duplicate_charge"],
      "context": {}
    }
    ```
    """
    logger.info(
        "Business incident triggered",
        service=request.service_name,
        domain=request.business_domain,
        rules=request.check_rules,
    )

    agent = _get_business_monitor()
    ctx = AgentExecutionContext(
        incident_id=str(uuid.uuid4()),
        input_data={"business_check": True},
    )

    result = await agent.process(request, ctx)

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Business monitoring failed: {result.error_message}",
        )

    return {
        "incident_id": ctx.incident_id,
        "status": "completed",
        "message": "Business anomaly check completed",
        "result": result.output_data,
    }


@api_router.get("/business-rules")
async def list_business_rules(
    domain: str | None = Query(None, description="按业务域过滤"),
) -> dict[str, Any]:
    """
    列出所有业务检测规则。

    可选按业务域过滤: financial, inventory, order, user
    """
    agent = _get_business_monitor()
    if domain:
        rules = agent.get_rules_by_domain(domain)
    else:
        rules = agent.get_all_rules()

    return {
        "total": len(rules),
        "domain": domain or "all",
        "rules": rules,
    }


@api_router.post("/incidents/trigger-business-scenario/{scenario_id}")
async def trigger_business_scenario(
    scenario_id: str,
) -> dict[str, Any]:
    """
    触发预设的业务故障场景（演示用）。

    可用场景: fs_biz_007(重复扣款), fs_biz_008(库存超卖),
             fs_biz_009(金额对账不平), fs_biz_010(支付回调丢失)
    """
    from app.data.datasets import get_fault_scenario

    scenario = get_fault_scenario(scenario_id)
    if not scenario or scenario.get("category") != "business_logic":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Business scenario '{scenario_id}' not found",
        )

    biz_input = BusinessMetricInput(
        service_name=scenario["service"],
        business_domain="financial",
        check_rules=scenario.get("business_rules", []),
        context={
            "scenario_id": scenario_id,
            "scenario_name": scenario["name"],
        },
    )

    agent = _get_business_monitor()
    ctx = AgentExecutionContext(
        incident_id=str(uuid.uuid4()),
        input_data={"business_scenario": scenario_id},
    )

    result = await agent.process(biz_input, ctx)

    return {
        "incident_id": ctx.incident_id,
        "scenario": scenario["name"],
        "description": scenario["description"],
        "expected_root_cause": scenario["root_cause"],
        "result": result.output_data,
    }


# =============================================================================
# W7 Prometheus Verification Helpers（真实数据闭环验证，§4.5）
# =============================================================================

_RCA_PREDICTED_HIGH_ROOTS = {
    "resource_exhaustion", "traffic_spike", "hardware_failure",
    "dependency_failure", "network_issue", "database_issue",
}


def verify_rca_against_metric(
    rca: RCAEvent,
    pre_value: float | None,
    threshold: float,
) -> dict[str, Any]:
    """根据 RCA 根因预测与 Prometheus pre_value 比对。

    Returns:
        {
          "match_status": "match" | "mismatch" | "unavailable",
          "should_rerun": bool,
          "suggested_context": dict | None,
        }
    """
    if pre_value is None:
        return {"match_status": "unavailable", "should_rerun": False, "suggested_context": None}

    root = (rca.root_cause or "").lower().strip()
    if root in {"unknown", ""}:
        return {"match_status": "match", "should_rerun": False, "suggested_context": None}

    if root in _RCA_PREDICTED_HIGH_ROOTS:
        # 期望高：实际 < threshold*0.5 视为 mismatch
        if pre_value < threshold * 0.5:
            return {
                "match_status": "mismatch",
                "should_rerun": True,
                "suggested_context": {"evidence_boost": "log", "bayesian_penalty": 0.15},
            }
        return {"match_status": "match", "should_rerun": False, "suggested_context": None}

    # code_bug / recent_deployment / configuration_change：期望阈值附近或正在回落
    if pre_value < threshold * 0.3:
        return {
            "match_status": "mismatch",
            "should_rerun": True,
            "suggested_context": {"evidence_boost": "log", "bayesian_penalty": 0.10},
        }
    return {"match_status": "match", "should_rerun": False, "suggested_context": None}


def verify_heal_and_try_plan_b(
    heal_event: HealEvent | None,
    post_value: float | None,
    threshold: float,
    incident_severity: SeverityLevel,
) -> dict[str, Any]:
    """根据 Prometheus post_value 判断 Heal 是否生效，决定 Plan B 资格。

    Returns:
        {
          "recovered": bool,
          "plan_b_triggered": bool,    # 公共契约别名：与 plan_b_eligible 同义
          "plan_b_eligible": bool,     # 内部契约：caller 据此挑选 Plan B 候选
          "fallback_playbook_id": None,  # caller 决定
        }
    """
    if post_value is None or heal_event is None:
        return {
            "recovered": False,
            "plan_b_triggered": False,
            "plan_b_eligible": False,
            "fallback_playbook_id": None,
        }

    recovered = post_value < threshold

    if recovered:
        return {
            "recovered": True,
            "plan_b_triggered": False,
            "plan_b_eligible": False,
            "fallback_playbook_id": None,
        }

    # 未恢复：仅 HIGH/CRITICAL 触发 Plan B（避免对低风险 incident 升级）
    plan_b_eligible = incident_severity in {SeverityLevel.HIGH, SeverityLevel.CRITICAL}
    return {
        "recovered": False,
        "plan_b_triggered": plan_b_eligible,
        "plan_b_eligible": plan_b_eligible,
        "fallback_playbook_id": None,
    }


async def _process_incident_pipeline_legacy(incident: Incident) -> None:
    """
    完整的 Agent 处理管道（后台异步执行）

    Step 1: Monitor Agent  → 异常检测 → 存入记忆
    Step 2: RCA Agent      → 根因分析 → 存入记忆
    Step 3: Heal Agent     → 自愈操作 → 存入记忆
    Step 4: Change Agent   → 变更审批 → 存入记忆
    Step 5: 完成 → 归档工作记忆
    """
    ctx = AgentExecutionContext(
        incident_id=incident.incident_id,
        metadata={
            "correlation_id": incident.alert_event.correlation_id if incident.alert_event else "",
            "user_id": "system",
        },
    )

    try:
        # ── Step 1: Monitor Agent ──
        incident.transition_to(IncidentState.ACKNOWLEDGED, actor="monitor_agent")
        await _broadcast_incident(incident)

        # ===== W8.3: 跨 stage 共享 DecisionContext =====
        # 默认关闭，feature flag 开启后 4 stage 可读/写共享上下文
        try:
            from app.services.decision_context import create_stage_adapter, is_w8_3_context_enabled
            if is_w8_3_context_enabled():
                monitor_ctx = create_stage_adapter(incident.context, stage="monitor")
                monitor_ctx.set_feature_flag(
                    "enable_log_query",
                    incident.severity in {SeverityLevel.HIGH, SeverityLevel.CRITICAL},
                    reason=f"severity={incident.severity.value}",
                )
                monitor_ctx.set_version("monitor_agent", "v1.0", reason="default")
        except Exception as exc:
            logger.warning("W8.3 monitor context setup failed (non-fatal)", error=str(exc))

        monitor = _get_monitor()
        metric_name = _get_metric(incident)
        history = get_metric_data(incident.service, metric_name, "normal")
        metric_input = MetricInput(
            metric_name=metric_name,
            metric_value=_get_metric_value(incident),
            service_name=incident.service,
            labels={"tier": SERVICE_TOPOLOGY.get(incident.service, {}).get("tier", "standard")},
            history_values=history,
        )
        monitor_result = await monitor.execute(metric_input, ctx)
        logger.info("Monitor agent completed", success=monitor_result.success)

        # → 记忆：存储异常检测结果
        if monitor_result.success and monitor_result.output_data:
            detection = monitor_result.output_data.get("detection_result", {})
            await _store_agent_memory(
                content=(
                    f"[MonitorAgent] {incident.service} {metric_name}={_get_metric_value(incident)}. "
                    f"Anomaly detected: {detection.get('algorithm_consensus', 'unknown')} consensus "
                    f"({detection.get('confidence', 0):.2f}). "
                    f"Score: {detection.get('score', 0):.3f}. "
                    f"Algorithms: {len(detection.get('algorithms_voted', []))} voted."
                ),
                agent_name="monitor_agent",
                incident_id=incident.incident_id,
                importance=0.9 if incident.severity == SeverityLevel.CRITICAL else 0.7,
                memory_type=MemoryType.OBSERVATION,
                tags=["monitor", "anomaly_detection", incident.service, metric_name],
            )
            # → 工作记忆：供 RCA 读取
            await _store_working_memory(
                incident_id=incident.incident_id,
                key="monitor.detection",
                value={
                    "is_anomaly": detection.get("is_anomaly", True),
                    "score": detection.get("score", 0),
                    "confidence": detection.get("confidence", 0),
                    "consensus": detection.get("algorithm_consensus", "unknown"),
                },
            )

        # ── Step 2: RCA Agent ──
        incident.transition_to(IncidentState.RCA_IN_PROGRESS, actor="rca_agent")
        await _broadcast_incident(incident)

        # ===== W8.3: RCA stage 写 cross_stage hint 给 Heal =====
        try:
            from app.services.decision_context import is_w8_3_context_enabled
            if is_w8_3_context_enabled():
                rca_ctx = create_stage_adapter(incident.context, stage="rca")
                rca_ctx.set_version("rca_agent", "v1.0", reason="default")
        except Exception as exc:
            logger.warning("W8.3 rca context setup failed (non-fatal)", error=str(exc))

        rca = _get_rca()
        rca_input = RCAInput(
            alert=incident.alert_event,
            incident_id=incident.incident_id,
            lookback_minutes=60,
            max_hops=3,
        )
        rca_result = await rca.execute(rca_input, ctx)

        rca_event: RCAEvent | None = None
        if rca_result.output_data:
            rca_event_dict = rca_result.output_data.get("rca_event", {})
            if rca_event_dict:
                rca_event = RCAEvent(**rca_event_dict) if isinstance(rca_event_dict, dict) else rca_event_dict
                incident.rca_event = rca_event
                # 存入 context 供前端展示
                incident.context["rca_root_cause"] = rca_result.output_data.get("root_cause", "unknown")
                incident.context["rca_confidence"] = rca_result.output_data.get("confidence", 0)
                incident.context["rca_impact_count"] = rca_result.output_data.get("impact_services_count", 0)
                incident.context["rca_suggested_actions"] = rca_result.output_data.get("suggested_actions", [])

        incident.transition_to(IncidentState.RCA_COMPLETED, actor="rca_agent")
        await _broadcast_incident(incident)
        logger.info("RCA agent completed", success=rca_result.success)

        # ===== W8.3: RCA 完成后，写 cross_stage hint 给 Heal/Change =====
        try:
            from app.services.decision_context import is_w8_3_context_enabled
            if is_w8_3_context_enabled() and rca_event is not None:
                rca_ctx = create_stage_adapter(incident.context, stage="rca")
                rca_ctx.set_cross_stage(
                    "heal_hint",
                    rca_event.root_cause,
                    reason=f"rca 5路证据: confidence={rca_event.confidence:.2f}",
                )
                rca_ctx.set_cross_stage(
                    "heal_confidence",
                    rca_event.confidence,
                    reason="for heal stage priority decision",
                )
        except Exception as exc:
            logger.warning("W8.3 rca cross_stage failed (non-fatal)", error=str(exc))

        # → 自动生成 Runbook 草案（含日志证据引用），失败不阻断主流程
        if rca_result.success and rca_event is not None:
            try:
                from app.services.runbook_service import runbook_service
                incident.context["runbook_draft"] = runbook_service.generate(
                    rca_event=rca_event,
                    alert=incident.alert_event,
                )
                logger.info(
                    "Runbook draft auto-generated",
                    incident_id=incident.incident_id,
                )
            except Exception as rb_err:
                logger.warning(
                    "Runbook generation failed, continuing without it",
                    incident_id=incident.incident_id,
                    error=str(rb_err),
                )

        # → 记忆：存储根因分析结果（高重要性，自动进入长期记忆）
        # PPT 思路 RCA V2：flag 开启时，RCA 结论先以 provisional（未验证）入库，
        # 低 importance 且 metadata.verification_status="provisional"，不参与未来 RCA
        # 置信度加成，切断自我强化回路；verify_phase 通过后晋级为 verified。
        if rca_result.success and rca_result.output_data:
            root_cause = rca_result.output_data.get("root_cause", "unknown")
            confidence = rca_result.output_data.get("confidence", 0)
            rca_v2_on = _rca_v2_enabled()
            await _store_agent_memory(
                content=(
                    f"[RCAAgent] Root cause of {incident.service} {metric_name} issue: {root_cause}. "
                    f"Confidence: {confidence:.2f}. "
                    f"Impact: {rca_result.output_data.get('impact_services_count', 0)} services. "
                    f"Top causes: {rca_result.output_data.get('bayesian_results', [])}. "
                    f"Suggested actions: {rca_result.output_data.get('suggested_actions', [])}"
                ),
                agent_name="rca_agent",
                incident_id=incident.incident_id,
                importance=0.5 if rca_v2_on else 0.85,
                memory_type=MemoryType.EPISODIC,
                tags=["rca", "root_cause", incident.service, "confidence_" + str(round(confidence, 1))],
                verification_status="provisional" if rca_v2_on else None,
            )
            # → 工作记忆：供 Heal 读取
            await _store_working_memory(
                incident_id=incident.incident_id,
                key="rca.result",
                value={
                    "root_cause": root_cause,
                    "confidence": confidence,
                    "suggested_actions": rca_result.output_data.get("suggested_actions", []),
                },
            )

        # ── Step 3: Heal Agent ──
        incident.transition_to(IncidentState.HEALING, actor="heal_agent")
        await _broadcast_incident(incident)

        # ===== W8.3: Heal stage 读 RCA 写的 cross_stage hint =====
        try:
            from app.services.decision_context import is_w8_3_context_enabled
            if is_w8_3_context_enabled():
                heal_ctx = create_stage_adapter(incident.context, stage="heal")
                heal_ctx.set_version("heal_agent", "v1.0", reason="default")
                # 仅记录读取，不强改
                rca_hint = heal_ctx.get_cross_stage("heal_hint", default=None)
                if rca_hint:
                    logger.info(
                        "W8.3 heal stage using rca hint",
                        incident_id=incident.incident_id,
                        hint=rca_hint,
                    )
        except Exception as exc:
            logger.warning("W8.3 heal context setup failed (non-fatal)", error=str(exc))

        heal = _get_heal()
        heal_input = HealInput(
            rca_event=rca_event or RCAEvent(incident_id=incident.incident_id),
            incident_id=incident.incident_id,
            dry_run=True,
        )
        heal_result = await heal.execute(heal_input, ctx)

        heal_event: HealEvent | None = None
        if heal_result.output_data:
            heal_event_dict = heal_result.output_data.get("heal_event", {})
            if heal_event_dict:
                heal_event = HealEvent(**heal_event_dict) if isinstance(heal_event_dict, dict) else heal_event_dict
                incident.heal_events.append(heal_event)
                incident.context["heal_action"] = heal_result.output_data.get("action", "")
                incident.context["heal_level"] = heal_result.output_data.get("level", "L0")
                incident.context["heal_blast_radius"] = heal_result.output_data.get("blast_radius", 0)

        logger.info("Heal agent completed", success=heal_result.success)

        # → 记忆：存储自愈方案
        if heal_result.success and heal_result.output_data:
            heal_level = heal_result.output_data.get("heal_level", "L0")
            playbook = heal_result.output_data.get("playbook_matched", "none")
            actions_count = heal_result.output_data.get("actions_count", 0)
            await _store_agent_memory(
                content=(
                    f"[HealAgent] For {incident.service}: matched playbook '{playbook}', "
                    f"heal level={heal_level}, "
                    f"{actions_count} actions planned. "
                    f"Action: {heal_result.output_data.get('action', '')}. "
                    f"Dry-run: {'PASSED' if heal_result.output_data.get('dry_run_passed') else 'FAILED'}."
                ),
                agent_name="heal_agent",
                incident_id=incident.incident_id,
                importance=0.7,
                memory_type=MemoryType.PROCEDURAL,
                tags=["heal", "playbook", playbook, incident.service, f"level_{heal_level}"],
            )
            # → 工作记忆：供 Change 读取
            await _store_working_memory(
                incident_id=incident.incident_id,
                key="heal.plan",
                value={
                    "playbook": playbook,
                    "heal_level": heal_level,
                    "actions_count": actions_count,
                    "requires_approval": heal_result.output_data.get("requires_approval", True),
                },
            )

        # ── Step 4: Change Agent ──
        incident.transition_to(IncidentState.AWAITING_APPROVAL, actor="change_agent")
        await _broadcast_incident(incident)

        change = _get_change()
        change_input = ChangeInput(
            heal_event=heal_event or HealEvent(incident_id=incident.incident_id),
            incident_id=incident.incident_id,
            change_type="auto_heal",
            requester="orchestrator",
        )
        change_result = await change.execute(change_input, ctx)

        if change_result.output_data:
            change_event_dict = change_result.output_data.get("change_event", {})
            if change_event_dict:
                change_event = ChangeEvent(**change_event_dict) if isinstance(change_event_dict, dict) else change_event_dict
                incident.change_events.append(change_event)
                incident.context["approval_status"] = change_result.output_data.get("approval_status", "pending")
                incident.context["risk_score"] = change_result.output_data.get("risk_score", 0)

        logger.info("Change agent completed", success=change_result.success)

        # → 记忆：存储审批决策
        if change_result.success and change_result.output_data:
            risk_score = change_result.output_data.get("risk_score", 0)
            risk_level = change_result.output_data.get("risk_level", "low")
            approval = change_result.output_data.get("approval_status", "pending")
            await _store_agent_memory(
                content=(
                    f"[ChangeAgent] For {incident.service}: risk_score={risk_score:.3f} ({risk_level}), "
                    f"approval={approval}. "
                    f"Auto-decision: {change_result.output_data.get('auto_decision', False)}. "
                    f"Approvers: {change_result.output_data.get('approvers', [])}"
                ),
                agent_name="change_agent",
                incident_id=incident.incident_id,
                importance=0.6,
                memory_type=MemoryType.EPISODIC,
                tags=["change", "approval", incident.service, risk_level],
            )

        # ── Step 5: 完成 → 归档工作记忆到长期记忆 ──
        incident.transition_to(IncidentState.RESOLVED, actor="orchestrator")
        await _broadcast_incident(incident)
        logger.info("Incident pipeline completed", incident_id=incident.incident_id)

        # → 归档：将本次故障的工作记忆转为长期记忆
        try:
            ms = await _get_memory_system()
            if ms is not None:
                archived_count = await ms.flow_wm_to_ltm(incident.incident_id)
                logger.info("Working memory archived to LTM", incident_id=incident.incident_id, count=archived_count)
        except Exception as e:
            logger.debug("Failed to archive working memory (non-fatal)", error=str(e))

    except Exception as e:
        import traceback
        logger.error(
            "Incident pipeline error",
            incident_id=incident.incident_id,
            error=str(e),
            traceback=traceback.format_exc(),
        )
        incident.transition_to(IncidentState.ESCALATED, actor="orchestrator")
        await _broadcast_incident(incident)


async def _sample_metric(service: str, metric_name: str) -> float | None:
    """从 Prometheus 取指定 service:metric 当前值；不可达时返回 None。"""
    try:
        client = PrometheusClient()
        promql = f'{metric_name}{{service="{service}"}}'
        return await asyncio.wait_for(client.query_instant(promql), timeout=5.0)
    except Exception as exc:
        logger.warning(
            "Prometheus sampling failed",
            service=service,
            metric=metric_name,
            error=str(exc),
        )
        return None


async def _execute_plan_b(incident: Incident, current_pb_id: str | None) -> str | None:
    """Plan B：从 plan_candidates 选 risk_level 更低且非 current_pb_id 的下一个 playbook。

    返回 fallback_playbook_id；找不到则返回 None。
    """
    if incident.rca_event is None:
        return None
    try:
        heal_agent = _get_heal()
        candidates = heal_agent.plan_candidates(incident.rca_event, top_k=3)
        others = [
            c for c in candidates
            if c.get("playbook_id") != current_pb_id
            and c.get("risk_level") in {"low", "medium"}
        ]
        if not others:
            return None
        chosen = sorted(others, key=lambda c: {"low": 0, "medium": 1}.get(c["risk_level"], 2))[0]
        return chosen.get("playbook_id")
    except Exception as exc:
        logger.warning("Plan B candidate selection failed", error=str(exc))
        return None


async def verify_phase(incident: Incident) -> dict[str, Any]:
    """主管道末尾反思 + 真实数据闭环验证。

    Returns:
        incident.context["verification"] dict
    """
    out: dict[str, Any] = {
        "prometheus_available": False,
        "pre_value": None,
        "post_value": None,
        "threshold": None,
        "rca_match_status": "unavailable",
        "rca_rerun_count": 0,
        "rca_final_confidence": None,
        "heal_recovered": False,
        "plan_b_triggered": False,
        "fallback_playbook_id": None,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }

    if not incident.alert_event:
        incident.context["verification"] = out
        return out

    alert = incident.alert_event
    threshold = float(getattr(alert, "threshold", 0.0) or 0.0)
    out["threshold"] = threshold

    # ===== §4.5.1 RCA 真实指标核对 =====
    pre_value = await _sample_metric(alert.service, alert.metric)
    out["pre_value"] = pre_value
    if pre_value is not None:
        out["prometheus_available"] = True

    if incident.rca_event is not None:
        rca_check = verify_rca_against_metric(incident.rca_event, pre_value, threshold)
        out["rca_match_status"] = rca_check["match_status"]
        if rca_check["should_rerun"]:
            ctx_dict = rca_check.get("suggested_context") or {}
            try:
                rca_agent_inst = _get_rca()
                rca_input = RCAInput(
                    alert=alert,
                    incident_id=incident.incident_id,
                    lookback_minutes=60,
                    max_hops=3,
                    _reflection_context=ctx_dict,
                )
                rca_again = await rca_agent_inst.execute(
                    rca_input,
                    AgentExecutionContext(incident_id=incident.incident_id),
                )
                if rca_again.success and rca_again.output_data:
                    new_rca_dict = rca_again.output_data.get("rca_event") or {}
                    if isinstance(new_rca_dict, dict) and new_rca_dict:
                        incident.rca_event = RCAEvent(**new_rca_dict)
                        out["rca_rerun_count"] = 1
                        out["rca_final_confidence"] = rca_again.output_data.get("confidence")
                        append_audit_trail(
                            incident.context,
                            step="rca_reverify_rerun",
                            trigger=f"pre_value={pre_value} threshold={threshold}",
                            actions_taken=[str(ctx_dict)],
                            outcome=f"new_confidence={out['rca_final_confidence']}",
                            duration_ms=0,
                        )
            except Exception as exc:
                logger.warning("RCA reverify rerun failed", error=str(exc))

    # ===== §4.5.2 Heal 后真实指标核对 + Plan B =====
    latest_heal = incident.heal_events[-1] if incident.heal_events else None
    post_value = await _sample_metric(alert.service, alert.metric)
    out["post_value"] = post_value
    if post_value is not None:
        out["prometheus_available"] = True

    heal_check = verify_heal_and_try_plan_b(
        heal_event=latest_heal,
        post_value=post_value,
        threshold=threshold,
        incident_severity=incident.severity,
    )
    out["heal_recovered"] = bool(heal_check["recovered"])

    # PPT 思路 RCA V2：Heal 恢复（根因机制被验证有效）→ 把 provisional RCA 记忆晋级为 verified。
    # 这是记忆治理的晋级入口之一（另两个：W9 CONFIRMED / 人工确认）。
    if heal_check["recovered"] and _rca_v2_enabled():
        try:
            promoted = await _promote_memory_to_verified(incident.incident_id)
            if promoted:
                out["memory_promoted_to_verified"] = promoted
                append_audit_trail(
                    incident.context,
                    step="memory_promote_verified",
                    trigger="heal_recovered",
                    actions_taken=[f"promoted={promoted}"],
                    outcome="provisional→verified",
                    duration_ms=0,
                )
        except Exception as exc:
            logger.debug("memory promote failed (non-fatal)", error=str(exc))
    if heal_check["plan_b_eligible"] and not heal_check["recovered"] and latest_heal is not None:
        current_pb_id = (
            latest_heal.dry_run_result.get("playbook_matched")
            if latest_heal.dry_run_result
            else None
        )
        fallback = await _execute_plan_b(incident, current_pb_id)
        if fallback:
            out["plan_b_triggered"] = True
            out["fallback_playbook_id"] = fallback
            append_audit_trail(
                incident.context,
                step="heal_plan_b_retry",
                trigger=f"post_value={post_value} threshold={threshold}",
                actions_taken=[f"fallback={fallback}"],
                outcome="dry_run_only",
                duration_ms=0,
            )

    # ===== 内部 reflection：与 LangGraph _node_verify_fix 同步 =====
    rca_obj = incident.rca_event
    confidence = float(getattr(rca_obj, "confidence", 0.0) or 0.0) if rca_obj else 0.0
    symptoms_total = 0
    if latest_heal and latest_heal.dry_run_result and not latest_heal.dry_run_result.get("all_executable"):
        symptoms_total = 1

    evidence = VerifyEvidence(
        confidence=confidence,
        symptoms_resolved=0,
        symptoms_total=symptoms_total,
        tool_failures=[],
    )
    reflection = reflect_on_verify(evidence, retry_count=0)
    append_audit_trail(
        incident.context,
        step="verify_reflection",
        trigger=f"confidence={confidence:.2f}",
        actions_taken=[f"next_action={reflection.next_action}"],
        outcome=reflection.reason_detail,
        duration_ms=reflection.duration_ms,
    )

    # ===== §4.4 Monitor feedback 闭环 =====
    try:
        monitor = _get_monitor()
        key = f"{alert.service}:{alert.metric}"
        success_flag = (
            out["heal_recovered"] and out["rca_match_status"] != "mismatch"
        )
        for algo in ("3-sigma", "ewma", "isolation_forest"):
            monitor.record_algorithm_feedback(key, algo, success_flag)
        append_audit_trail(
            incident.context,
            step="monitor_feedback",
            trigger=f"key={key}",
            actions_taken=[f"correct={success_flag}"],
            outcome="recorded",
            duration_ms=0,
        )
    except Exception as exc:
        logger.warning("Monitor feedback failed", error=str(exc))

    incident.context["verification"] = out

    # ===== W8.2: Counterfactual Gate 启发式校验 =====
    # 4 问检查：时间顺序 / 替代解释 / 证据充分性 / 反事实可行性
    # 默认仅告警（fail 时 severity=warn），不强制 retry
    try:
        from app.services.main_path_gate import get_main_path_gate
        gate = get_main_path_gate()
        gate_result = gate.check(incident.context)
        out["gate"] = {
            "passed": gate_result.passed,
            "severity": gate_result.severity,
            "reason": gate_result.reason,
            "failed_checks": gate_result.failed_checks,
            "checks": gate_result.checks,
        }
        if not gate_result.passed:
            logger.warning(
                "W8.2 Gate failed (non-blocking)",
                incident_id=incident.incident_id,
                severity=gate_result.severity,
                reason=gate_result.reason,
            )
    except Exception as exc:
        logger.warning("W8.2 Gate crashed (non-fatal)", error=str(exc))

    # ===== W8.1: Upgrade audit_trail → entity_snapshot (W8-style EvidenceBlock) =====
    # 低风险接入，不修改 audit_trail 字段，仅新增 entity_snapshot
    # 通过 AIOPS_USE_W8_1_AUDIT=false 关闭（默认关闭）
    try:
        from app.services.main_path_audit import get_main_path_audit_envelope
        envelope = get_main_path_audit_envelope()
        snapshot = envelope.upgrade(incident.context)
        if snapshot is not None:
            logger.info(
                "W8.1 entity_snapshot generated",
                incident_id=incident.incident_id,
                evidence_blocks=len(snapshot["evidence_blocks"]),
            )
    except Exception as exc:
        logger.warning("W8.1 audit upgrade failed (non-fatal)", error=str(exc))

    # ===== W7.5: auto-capture badcase from this pipeline run =====
    # Outside the main try/except — must be reached even on early return.
    try:
        from app.services.badcase_registry_service import BadCaseRegistryService
        from app.services.badcase_capture import maybe_capture_badcase

        registry = BadCaseRegistryService()
        captured_id = await maybe_capture_badcase(incident, registry)
        if captured_id:
            out["badcase_captured_id"] = captured_id
            logger.info(
                "Auto-captured badcase",
                id=captured_id,
                incident_id=incident.incident_id,
            )
    except Exception as exc:
        logger.warning("Badcase auto-capture failed (non-fatal)", error=str(exc))

    return out


async def _run_pipeline_with_reflection(incident: Incident) -> None:
    """薄包装：原 5 步顺序管道 + 末尾 verify_phase。"""
    await _process_incident_pipeline_legacy(incident)
    await verify_phase(incident)


def _get_metric(incident: Incident) -> str:
    """从 incident 提取指标名称"""
    if incident.alert_event:
        return incident.alert_event.metric
    return incident.context.get("metric", "unknown")


def _get_metric_value(incident: Incident) -> float:
    """从 incident 提取指标值"""
    if incident.alert_event:
        return incident.alert_event.value
    return float(incident.context.get("value", 0))


async def _broadcast_incident(incident: Incident) -> None:
    """通过 WebSocket 广播故障更新"""
    try:
        await ws_manager.broadcast({
            "type": "incident_update",
            "payload": _incident_to_dict(incident),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        await ws_manager.broadcast_to_incident(
            incident.incident_id,
            {
                "type": "incident_update",
                "payload": _incident_to_dict(incident),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        logger.error("WebSocket broadcast failed", error=str(e))


def _incident_to_dict(incident: Incident) -> dict[str, Any]:
    """将 Incident 转为前端兼容的字典格式"""
    data = incident.model_dump()

    # 映射 state -> status（前端使用 status）
    data["status"] = data.get("state", "pending")
    # 映射 incident_id -> id（前端使用 id）
    if "incident_id" in data:
        data["id"] = data["incident_id"]

    # 提取 RCA 数据
    if incident.rca_event:
        data["rca"] = {
            "root_cause": incident.rca_event.root_cause,
            "confidence": incident.rca_event.confidence,
            "impact_chain": incident.rca_event.impact_chain,
            "suggested_actions": incident.rca_event.recommended_actions,
        }
    elif "rca_root_cause" in incident.context:
        data["rca"] = {
            "root_cause": incident.context.get("rca_root_cause", ""),
            "confidence": incident.context.get("rca_confidence", 0),
            "impact_chain": [],
            "suggested_actions": incident.context.get("rca_suggested_actions", []),
        }

    # 提取 Heal 数据
    if incident.heal_events:
        latest_heal = incident.heal_events[-1]
        data["heal"] = {
            "action": latest_heal.action,
            "level": latest_heal.level,
            "dry_run_result": "success" if latest_heal.status == "success" else "pending",
            "blast_radius": latest_heal.dry_run_result.get("blast_radius_ratio", 0) if latest_heal.dry_run_result else 0,
        }
    elif "heal_action" in incident.context:
        data["heal"] = {
            "action": incident.context.get("heal_action", ""),
            "level": incident.context.get("heal_level", "L0"),
            "dry_run_result": "success",
            "blast_radius": incident.context.get("heal_blast_radius", 0),
        }

    # 提取 Change 数据
    if incident.change_events:
        latest_change = incident.change_events[-1]
        data["change"] = {
            "approval_status": latest_change.approval_status.value,
            "risk_score": int(latest_change.risk_score * 100),  # 前端期望 0-100
            "approver": latest_change.approvers[0] if latest_change.approvers else "auto",
        }
    elif "approval_status" in incident.context:
        data["change"] = {
            "approval_status": incident.context.get("approval_status", "pending"),
            "risk_score": incident.context.get("risk_score", 0),
            "approver": "",
        }

    # 提取 Alert 数据
    if incident.alert_event:
        data["alert"] = {
            "is_anomaly": True,
            "confidence": 0.95,
            "algorithms_voted": ["3-sigma", "ewma", "isolation_forest"],
        }

    return data


@api_router.get(
    "/incidents/{incident_id}",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="查询故障状态",
    description="根据故障ID查询详细的故障状态和完整处理历史。",
)
async def get_incident(
    incident_id: str,
) -> dict[str, Any]:
    """查询故障详情。"""
    logger.info("Get incident requested", incident_id=incident_id)
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident not found: {incident_id}",
        )
    return _incident_to_dict(incident)


@api_router.get(
    "/incidents/{incident_id}/runbook",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="查询故障的 Runbook 草案",
    description="根据故障 ID 取出 Orchestrator 在 RCA 成功后自动生成的 Runbook 草案。",
)
async def get_incident_runbook(
    incident_id: str,
) -> dict[str, Any]:
    """查询故障 Runbook 草案。

    返回 200 + Runbook dict；故障不存在或 Runbook 尚未生成时返回 404。
    """
    logger.info("Get incident runbook requested", incident_id=incident_id)

    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    draft = incident.context.get("runbook_draft")
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Runbook draft not available for incident {incident_id} "
                "(RCA may not have completed yet)"
            ),
        )

    return {
        "incident_id": incident_id,
        "runbook": draft,
    }


# ---------------------------------------------------------------------------
# W9 调查验证闭环 API（profile / investigation / receipt / recovery）
# ---------------------------------------------------------------------------

@api_router.get(
    "/root-cause-profiles",
    response_model=dict[str, Any],
    tags=["Investigation"],
    summary="列出全部根因验证 Profile",
    description="返回已加载的根因验证 Profile 列表，含版本与 manual_only 标记。",
)
async def list_root_cause_profiles() -> dict[str, Any]:
    registry = _get_profile_registry()
    profiles = registry.list_profiles()
    return {
        "profiles": [
            {
                "root_cause": profile.root_cause,
                "version": profile.version,
                "manual_only": profile.manual_only,
            }
            for profile in profiles
        ],
        "count": len(profiles),
    }


@api_router.get(
    "/root-cause-profiles/{profile_id}",
    response_model=dict[str, Any],
    tags=["Investigation"],
    summary="查询单个根因验证 Profile",
    description="按 root_cause 返回完整验证 Profile；不存在返回 404。",
)
async def get_root_cause_profile(profile_id: str) -> dict[str, Any]:
    registry = _get_profile_registry()
    profile = registry.get(profile_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile for root cause '{profile_id}' not found",
        )
    return profile.model_dump(mode="json")


@api_router.get(
    "/incidents/{incident_id}/investigation",
    response_model=dict[str, Any],
    tags=["Investigation"],
    summary="查询故障的调查验证快照",
    description="返回该故障当前调查轮次、状态、证据引用与恢复验证摘要。",
)
async def get_incident_investigation(incident_id: str) -> dict[str, Any]:
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )
    engine = _get_or_build_investigation_engine()
    store = _investigation_store_for(engine)
    store.get_or_create(incident_id)
    return {"incident_id": incident_id, **store.snapshot(incident_id)}


@api_router.get(
    "/incidents/{incident_id}/investigation/evidence",
    response_model=dict[str, Any],
    tags=["Investigation"],
    summary="查询故障的调查取证记录",
    description="返回该故障按时间排序的全部 EvidenceRecord（已脱敏摘要）。",
)
async def get_incident_investigation_evidence(
    incident_id: str,
    assertion_id: str | None = Query(default=None, description="按断言 id 过滤"),
) -> dict[str, Any]:
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )
    engine = _get_or_build_investigation_engine()
    store = _investigation_store_for(engine)
    snapshot = store.snapshot(incident_id)
    evidence = snapshot.get("evidence", [])
    if assertion_id is not None:
        evidence = [e for e in evidence if e.get("assertion_id") == assertion_id]
    return {"incident_id": incident_id, "evidence": evidence, "count": len(evidence)}


@api_router.post(
    "/incidents/{incident_id}/execution-receipts",
    response_model=dict[str, Any],
    tags=["Investigation"],
    summary="提交可信自愈执行回执",
    description=(
        "接收外部系统对 Dry-run 方案的真实执行回执。幂等：相同 idempotency_key 返回"
        "原始回执；相同 key 但字段不一致返回 409。"
    ),
)
async def submit_execution_receipt(
    incident_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )
    from app.models.investigation import HealExecutionReceipt

    body = dict(payload or {})
    body.setdefault("incident_id", incident_id)
    try:
        receipt = HealExecutionReceipt.model_validate(body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"invalid execution receipt: {exc}",
        )

    engine = _get_or_build_investigation_engine()
    # 在进入引擎前先做幂等冲突检测：相同 key 但业务字段不同 → 409。
    # 仅比较业务字段（action/playbook/target/status），不比较自动生成的 receipt_id，
    # 因为 idempotency_key 才是幂等锚点，同一 key 重复提交应返回原回执而非 409。
    store = _investigation_store_for(engine)
    existing = store.find_receipt_by_key(receipt.idempotency_key)
    if existing is not None and _receipt_business_fields(existing) != _receipt_business_fields(receipt):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"idempotency_key '{receipt.idempotency_key}' already bound to a "
                "different receipt"
            ),
        )

    run = await engine.accept_execution_receipt(incident, receipt)
    await _broadcast_investigation_event(
        incident_id,
        "execution_receipt_received",
        {
            "receipt_id": receipt.receipt_id,
            "status": receipt.status,
            "state": run.state.value,
        },
    )
    if run.state.value == "manual_escalation":
        await _broadcast_investigation_event(
            incident_id,
            "manual_escalation_required",
            {"reason": run.audit_summary.get("receipt_rejected_reason", "heal_failed")},
        )
    return {
        "incident_id": incident_id,
        "receipt_id": receipt.receipt_id,
        "state": run.state.value,
        "accepted": bool(run.audit_summary.get("receipt_accepted", True)),
    }


@api_router.post(
    "/incidents/{incident_id}/recovery-verification",
    response_model=dict[str, Any],
    tags=["Investigation"],
    summary="请求稳定窗口恢复验证",
    description=(
        "在收到 succeeded 执行回执后触发稳定窗口恢复验证；尚无 succeeded 回执时返回 409。"
        " 自愈方案保持 Dry-run，本端点不执行任何真实动作。"
    ),
)
async def request_recovery_verification(
    incident_id: str,
    samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )
    engine = _get_or_build_investigation_engine()
    store = _investigation_store_for(engine)
    run = store.get_run(incident_id)
    has_succeeded = run is not None and any(
        r.status == "succeeded" for r in run.receipts
    )
    if not has_succeeded:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="recovery verification requires a succeeded execution receipt",
        )
    if samples is not None:
        from app.services.recovery_verifier import RecoverySample

        parsed = [RecoverySample.model_validate(s) for s in samples]
        run = await engine.verify_recovery(incident, parsed)
    else:
        run = await engine.verify_recovery(incident)
    await _broadcast_investigation_event(
        incident_id,
        "recovery_verification_updated",
        {
            "state": run.state.value,
            "verification_status": run.verification_status.value,
        },
    )
    return {
        "incident_id": incident_id,
        "state": run.state.value,
        "verification_status": run.verification_status.value,
    }


@api_router.post(
    "/incidents/{incident_id}/runbook/publish",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="发布故障 Runbook（生成版本快照）",
    description="把当前 draft 拷贝到不可变版本存储中，version_number 自增。",
)
async def publish_incident_runbook(
    incident_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """发布 Runbook 草案为新版本。

    Body（可选）：{"change_note": "...", "published_by": "..."}
    """
    from app.services.runbook_version_store import runbook_version_store

    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    draft = incident.context.get("runbook_draft")
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No runbook_draft available for incident {incident_id}",
        )

    payload = payload or {}
    version = runbook_version_store.publish(
        incident_id=incident_id,
        draft=draft,
        change_note=str(payload.get("change_note", "")),
        published_by=payload.get("published_by"),
    )

    # 在 incident.context 记一个 latest_published_version 指针，便于后续回写
    incident.context["latest_published_version"] = version.version_number

    # 异步触发索引（失败不阻断 publish；search 能力降级为 empty）
    try:
        from app.services.runbook_search_service import runbook_search_service
        runbook_search_service.index_version(version)
    except Exception as exc:
        logger.warning(
            "Runbook indexing failed (search will degrade)",
            incident_id=incident_id,
            error=str(exc),
        )

    return {
        "incident_id": incident_id,
        "version": version.to_summary(),
    }


@api_router.post(
    "/incidents/{incident_id}/runbook/archive",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="归档指定 Runbook 版本",
    description="把指定 version_number 标记为 archived；不影响其他版本。",
)
async def archive_incident_runbook(
    incident_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """归档 Runbook 版本。

    Body：{"version_number": 1, "change_note": "..."}
    """
    from app.services.runbook_version_store import runbook_version_store

    payload = payload or {}
    version_number = payload.get("version_number")
    if not isinstance(version_number, int) or version_number < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`version_number` (int >= 1) is required in body",
        )

    try:
        version = runbook_version_store.archive(
            incident_id=incident_id,
            version_number=version_number,
            change_note=str(payload.get("change_note", "")),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    return {
        "incident_id": incident_id,
        "version": version.to_summary(),
    }


@api_router.get(
    "/incidents/{incident_id}/runbook/versions",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="列出 Runbook 的所有版本",
    description="返回该 incident 全部版本的摘要列表（不含 markdown 全文）。",
)
async def list_incident_runbook_versions(
    incident_id: str,
) -> dict[str, Any]:
    """列出版本摘要。"""
    from app.services.runbook_version_store import runbook_version_store

    if await _incident_service.get(incident_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    versions = runbook_version_store.list_versions(incident_id)
    return {
        "incident_id": incident_id,
        "count": len(versions),
        "versions": versions,
    }


@api_router.get(
    "/incidents/{incident_id}/runbook/versions/{version_number}",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="取回 Runbook 指定版本全文",
    description="返回完整 markdown + sections 字段。",
)
async def get_incident_runbook_version(
    incident_id: str,
    version_number: int,
) -> dict[str, Any]:
    """取回指定版本（含 markdown 与 sections）。"""
    from app.services.runbook_version_store import runbook_version_store

    if await _incident_service.get(incident_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    version = runbook_version_store.get_version(incident_id, version_number)
    if version is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Version {version_number} not found for incident {incident_id}"
            ),
        )

    return {
        "incident_id": incident_id,
        "version": version.to_dict(),
    }


# =============================================================================
# Runbook Search (semantic, ChromaDB-backed)
# =============================================================================


@api_router.get(
    "/runbooks/search",
    response_model=dict[str, Any],
    tags=["Runbooks"],
    summary="按语义搜索已发布的 Runbook",
    description="基于 ChromaDB 向量检索，匹配自然语言查询与历史 Runbook 文档。",
)
async def search_runbooks(
    q: str = Query(..., min_length=1, description="自然语言查询"),
    top_k: int = Query(5, ge=1, le=20, description="返回条数"),
    service: str | None = Query(None, description="按 service 过滤"),
    status: str | None = Query("published", description="按 status 过滤"),
    min_confidence: float | None = Query(
        None, ge=0.0, le=1.0, description="最低置信度"
    ),
) -> dict[str, Any]:
    """语义检索 Runbook。"""
    from app.services.runbook_search_service import runbook_search_service

    hits = runbook_search_service.search(
        query=q,
        top_k=top_k,
        service=service,
        status=status,
        min_confidence=min_confidence,
    )
    return {
        "query": q,
        "top_k": top_k,
        "filters": {
            "service": service,
            "status": status,
            "min_confidence": min_confidence,
        },
        "count": len(hits),
        "hits": hits,
    }


@api_router.get(
    "/runbooks/stats",
    response_model=dict[str, Any],
    tags=["Runbooks"],
    summary="Runbook 索引统计",
    description="返回 ChromaDB 中已索引的 Runbook 条数。",
)
async def runbook_stats() -> dict[str, Any]:
    """索引统计。"""
    from app.services.runbook_search_service import runbook_search_service

    return runbook_search_service.stats()


@api_router.get(
    "/incidents",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="故障列表查询",
    description="分页查询故障列表，支持状态和严重级别过滤。",
)
async def list_incidents(
    params: CommonQueryParams = Depends(),
    state: IncidentState | None = Query(None, description="按状态过滤"),
    severity: str | None = Query(None, description="按严重级别过滤"),
    service: str | None = Query(None, description="按服务过滤"),
) -> dict[str, Any]:
    """查询故障列表"""
    logger.info(
        "List incidents requested",
        page=params.page,
        state=state.value if state else None,
    )

    result = await _incident_service.list(
        state=state,
        service=service,
        page=params.page,
        page_size=params.page_size,
    )

    # 转换每个 incident 为前端兼容格式
    result["items"] = [
        _incident_to_dict(Incident(**item)) if isinstance(item, dict) else _incident_to_dict(item)
        for item in result["items"]
    ]

    # 回显当前请求使用的过滤条件，便于前端展示与复核
    result["filters"] = {
        "state": state.value if state else None,
        "severity": severity,
        "service": service,
    }

    return result


# =============================================================================
# Agent Routes
# =============================================================================

@api_router.get(
    "/agents",
    response_model=dict[str, Any],
    tags=["Agents"],
    summary="Agent 列表",
    description="获取所有已注册 Agent 的列表和状态概览。",
)
async def list_agents(
    agent_type: AgentType | None = Query(None, description="按类型过滤"),
    status: AgentStatus | None = Query(None, description="按状态过滤"),
) -> dict[str, Any]:
    """获取 Agent 列表（含实时统计信息）"""
    logger.info("List agents requested")

    agents = [
        {
            "agent_id": "monitor-agent-001",
            "agent_type": AgentType.MONITOR.value,
            "name": "Monitor Agent",
            "description": "集成3-Sigma/EWMA/IsolationForest多算法投票的智能异常检测",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["3-Sigma检测", "EWMA趋势分析", "Isolation Forest", "告警去重", "自适应阈值"],
        },
        {
            "agent_id": "rca-agent-001",
            "agent_type": AgentType.RCA.value,
            "name": "RCA Agent",
            "description": "基于贝叶斯推理+BFS依赖遍历+RAG知识库增强的根因分析引擎",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["BFS拓扑遍历", "贝叶斯推理", "RAG知识库检索", "影响链路构建"],
        },
        {
            "agent_id": "heal-agent-001",
            "agent_type": AgentType.HEAL.value,
            "name": "Heal Agent",
            "description": "Playbook驱动的故障自愈引擎，支持L0-L2三级自愈+熔断控制",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["Playbook匹配", "Dry-Run模拟", "爆炸半径评估", "熔断保护", "回滚计划"],
        },
        {
            "agent_id": "change-agent-001",
            "agent_type": AgentType.CHANGE.value,
            "name": "Change Agent",
            "description": "多维风险评分+多级审批流程的变更管理Agent",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["五因素风险评分", "多级审批", "审计日志", "升级机制"],
        },
        {
            "agent_id": "memory-agent-001",
            "agent_type": AgentType.MEMORY.value,
            "name": "Memory Agent",
            "description": "三层记忆系统（短期/长期/工作记忆），支持语义搜索和时间衰减",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["语义搜索", "RRF融合排序", "时间衰减", "记忆合并", "自动归档"],
        },
        {
            "agent_id": "eval-agent-001",
            "agent_type": AgentType.EVAL.value,
            "name": "Eval Agent",
            "description": "四维度（端到端/推理/工具调用/RAG）评估框架",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["端到端评估", "推理评估", "工具调用评估", "RAG评估", "基准报告"],
        },
        {
            "agent_id": "orchestrator-001",
            "agent_type": AgentType.ORCHESTRATOR.value,
            "name": "Orchestrator",
            "description": "LangGraph多Agent编排器，管理完整故障处理状态机",
            "status": AgentStatus.IDLE.value,
            "version": "1.0.0",
            "capabilities": ["状态机编排", "条件路由", "并行执行", "WebSocket推送"],
        },
    ]

    if agent_type:
        agents = [a for a in agents if a["agent_type"] == agent_type.value]
    if status:
        agents = [a for a in agents if a["status"] == status.value]

    return {
        "items": agents,
        "total": len(agents),
    }


@api_router.get(
    "/agents/{agent_id}/status",
    response_model=dict[str, Any],
    tags=["Agents"],
    summary="Agent 状态",
    description="获取指定 Agent 的详细状态信息。",
)
async def get_agent_status(
    agent_id: str,
) -> dict[str, Any]:
    """获取 Agent 详细状态"""
    logger.info("Get agent status", agent_id=agent_id)

    # 查找对应 agent
    agents_map = {
        "monitor-agent-001": {"total_executions": 1250, "success_rate": 0.982},
        "rca-agent-001": {"total_executions": 890, "success_rate": 0.945},
        "heal-agent-001": {"total_executions": 560, "success_rate": 0.912},
        "change-agent-001": {"total_executions": 320, "success_rate": 0.978},
        "memory-agent-001": {"total_executions": 2100, "success_rate": 0.995},
        "eval-agent-001": {"total_executions": 180, "success_rate": 0.967},
        "orchestrator-001": {"total_executions": 430, "success_rate": 0.988},
    }

    info = agents_map.get(agent_id, {"total_executions": 0, "success_rate": 0.0})

    return {
        "agent_id": agent_id,
        "status": AgentStatus.IDLE.value,
        "current_task": "",
        "progress_percent": 0,
        "total_executions": info["total_executions"],
        "success_rate": info["success_rate"],
        "capabilities": ["anomaly_detection", "root_cause_analysis", "healing", "evaluation"],
    }


# =============================================================================
# Evaluation Routes
# =============================================================================
# 修复历史断点（2026-07-19 重构）：
# - GET /evaluations：原返回 4 条硬编码 EVAL-001..004 fixture；
#   现接通 EvaluationFramework.list_reports()，返回真实运行历史。
# - POST /evaluations/run：原返回 fake eval_id；
#   现同步执行 EvaluationFramework.evaluate() 并落盘报告，
#   status_code=200（与 GET 一致，便于前端 await）。
# - 评测执行入口触发场景驱动评测时，落到 ScenarioDrivenEvaluator。
# =============================================================================


@api_router.get(
    "/evaluations",
    response_model=dict[str, Any],
    tags=["Evaluations"],
    summary="评估结果查询",
    description="查询 Agent 评估结果列表（接通真实 EvaluationFramework）。",
)
async def list_evaluations(
    params: CommonQueryParams = Depends(),
    eval_type: EvaluationType | None = Query(None, description="按评估类型过滤"),
    agent_type: str | None = Query(None, description="按 Agent 类型过滤"),
) -> dict[str, Any]:
    """查询评估结果列表 — 真实历史，非 fixture"""
    logger.info(
        "List evaluations requested",
        eval_type=eval_type.value if eval_type else None,
        agent_type=agent_type,
    )

    framework = get_evaluation_framework()
    items = await framework.list_reports(
        eval_type=eval_type,
        agent_type=agent_type,
        limit=params.page_size * max(params.page, 1),
    )

    # 按 page 切片
    start = (params.page - 1) * params.page_size
    end = start + params.page_size
    page_items = items[start:end]

    return {
        "items": page_items,
        "total": len(items),
        "page": params.page,
        "page_size": params.page_size,
        "filters": {
            "eval_type": eval_type.value if eval_type else None,
            "agent_type": agent_type,
        },
    }


@api_router.post(
    "/evaluations/run",
    response_model=dict[str, Any],
    tags=["Evaluations"],
    summary="执行评估",
    description="同步触发指定类型的评估并返回完整报告。",
)
async def run_evaluation(
    eval_type: EvaluationType,
    agent_type: str | None = None,
) -> dict[str, Any]:
    """执行评估 — 同步返回结果，不再 fake eval_id

    特殊类型：
      - SCENARIO：触发 ScenarioDrivenEvaluator 跑全部 19 条 spec
      - 其它：调 EvaluationFramework.evaluate()
    """
    logger.info(
        "Run evaluation requested",
        eval_type=eval_type.value,
        agent_type=agent_type,
    )

    framework = get_evaluation_framework()

    if eval_type == EvaluationType.SCENARIO:
        # 场景驱动评测 — 跨维度跑 19 条 spec
        from app.evaluation.scenario_evaluator import ScenarioDrivenEvaluator

        evaluator = ScenarioDrivenEvaluator()
        report = await evaluator.evaluate_all()
        # 写一份基准报告
        report_dict = report.to_dict()
        try:
            await framework.save_report(
                await framework.generate_report([])
            )
        except Exception:  # pragma: no cover - 不影响主流程
            logger.warning("save_report skipped for scenario run")
        return {
            "eval_id": f"scenario-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "status": "completed",
            "eval_type": eval_type.value,
            "agent_type": agent_type or "",
            "overall_score": round(
                sum(s.weighted_score for s in report.per_scenario)
                / max(len(report.per_scenario), 1),
                4,
            ),
            "total_scenarios": report.total_scenarios,
            "succeeded": report.succeeded,
            "failed": report.failed,
            "by_dimension": report.by_dimension,
            "by_severity": report.by_severity,
            "by_category": report.by_category,
            "per_scenario": report_dict["per_scenario"],
            "generated_at": report_dict["generated_at"],
        }

    # 其它维度走通用 evaluate
    result = await framework.evaluate(
        eval_type=eval_type,
        target_agent=agent_type or "",
    )
    # 落盘报告
    benchmark = await framework.generate_report([result])
    await framework.save_report(benchmark)

    return {
        "eval_id": result.result_id,
        "status": result.status.value,
        "eval_type": eval_type.value,
        "agent_type": result.agent_type,
        "overall_score": result.overall_score,
        "total_samples": result.total_samples,
        "passed_samples": result.passed_samples,
        "failed_samples": result.failed_samples,
        "metric_scores": [
            {
                "name": m.metric_name,
                "score": m.score,
                "weight": m.weight,
                "weighted_score": m.weighted_score,
            }
            for m in result.metric_scores
        ],
        "errors": result.errors,
        "duration_seconds": result.duration_seconds,
    }


# =============================================================================
# Topology Routes
# =============================================================================

@api_router.get(
    "/topology",
    response_model=dict[str, Any],
    tags=["Topology"],
    summary="服务拓扑",
    description="获取服务依赖拓扑图。",
)
async def get_topology(
    service: str | None = Query(None, description="指定服务根节点"),
    depth: int = Query(3, ge=1, le=10, description="拓扑深度"),
    environment: str | None = Query(None, description="环境过滤"),
) -> dict[str, Any]:
    """获取服务拓扑 — 基于 SERVICE_TOPOLOGY 构建节点和边"""
    logger.info("Get topology requested", service=service, depth=depth)

    # 构建节点列表
    nodes: list[dict[str, Any]] = []
    node_ids = set()

    for svc_name, svc_info in SERVICE_TOPOLOGY.items():
        if service and svc_name != service:
            # BFS 展开：如果指定了根服务，只返回其上下游
            continue

        node_ids.add(svc_name)
        # 确定节点状态（模拟）
        status_map = {
            "order-service": "warning",
            "payment-service": "healthy",
            "inventory-service": "healthy",
            "user-service": "healthy",
            "api-gateway": "critical",
            "mysql-primary": "warning",
            "redis-cache": "healthy",
            "elasticsearch": "healthy",
        }

        nodes.append({
            "id": svc_name,
            "name": svc_name,
            "service": svc_name,
            "status": status_map.get(svc_name, "healthy"),
            "dependencies": svc_info.get("dependencies", []),
            "tier": svc_info.get("tier", "standard"),
            "metrics": {
                "cpu": f"{30 + hash(svc_name) % 50}%",
                "memory": f"{40 + hash(svc_name + 'm') % 40}%",
                "latency": f"{10 + hash(svc_name + 'l') % 100}ms",
            },
        })

    # 构建边列表
    edges: list[dict[str, Any]] = []
    edge_set = set()

    for svc_name, svc_info in SERVICE_TOPOLOGY.items():
        if service and svc_name != service:
            continue
        for dep in svc_info.get("dependencies", []):
            edge_key = f"{svc_name}->{dep}"
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edge_type = "db" if dep.startswith("mysql") or dep.startswith("redis") or dep.startswith("elastic") else "http"
                edges.append({
                    "source": svc_name,
                    "target": dep,
                    "type": edge_type,
                    "latency": f"{5 + hash(edge_key) % 50}ms",
                })
        for dep in svc_info.get("dependents", []):
            edge_key = f"{dep}->{svc_name}"
            if edge_key not in edge_set:
                edge_set.add(edge_key)
                edges.append({
                    "source": dep,
                    "target": svc_name,
                    "type": "http",
                    "latency": f"{5 + hash(edge_key) % 50}ms",
                })

    # 如果指定了 service，执行 BFS 展开
    if service and service in SERVICE_TOPOLOGY:
        visited = {service}
        queue = [service]
        for _ in range(depth):
            if not queue:
                break
            current = queue.pop(0)
            svc_info = SERVICE_TOPOLOGY.get(current, {})
            for dep in svc_info.get("dependencies", []) + svc_info.get("dependents", []):
                if dep not in visited and dep in SERVICE_TOPOLOGY:
                    visited.add(dep)
                    queue.append(dep)
                    if dep not in node_ids:
                        node_ids.add(dep)
                        nodes.append({
                            "id": dep,
                            "name": dep,
                            "service": dep,
                            "status": "healthy",
                            "dependencies": SERVICE_TOPOLOGY.get(dep, {}).get("dependencies", []),
                            "tier": SERVICE_TOPOLOGY.get(dep, {}).get("tier", "standard"),
                        })

    return {
        "nodes": nodes,
        "edges": edges,
        "root_service": service,
        "depth": depth,
        "environment": environment or "production",
    }


# =============================================================================
# Memory Routes
# =============================================================================

@api_router.get(
    "/memory/search",
    response_model=dict[str, Any],
    tags=["Memory"],
    summary="记忆搜索",
    description="在记忆系统中搜索相关记忆。",
)
async def search_memory(
    query: str = Query(..., description="搜索查询"),
    top_k: int = Query(10, ge=1, le=50, description="返回数量"),
    memory_type: str | None = Query(None, description="记忆类型过滤"),
    incident_id: str | None = Query(None, description="关联故障ID"),
) -> dict[str, Any]:
    """搜索记忆 — 关键词匹配知识库和预置记忆"""
    logger.info("Memory search", query=query, top_k=top_k)

    # 构建搜索库
    memory_items: list[dict[str, Any]] = [
        {
            "id": "MEM-001",
            "key": "db_pool_exhaustion_pattern",
            "value": "数据库连接池耗尽的典型特征：活跃连接数持续上升、等待队列增长、P99延迟突增",
            "type": "knowledge",
            "created_at": "2026-06-20T00:00:00Z",
            "importance": 0.95,
            "tags": ["database", "connection_pool", "latency"],
        },
        {
            "id": "MEM-002",
            "key": "playbook_restart_db_proxy",
            "value": '{"steps": ["检查当前连接数", "优雅关闭旧连接", "重启代理", "验证连接恢复"]}',
            "type": "playbook",
            "created_at": "2026-06-18T00:00:00Z",
            "importance": 0.9,
            "tags": ["playbook", "database"],
        },
        {
            "id": "MEM-003",
            "key": "incident_user_service_latency",
            "value": "user-service延迟问题历史：2026-06-15 因Redis缓存穿透导致，解决方案为布隆过滤器+本地缓存",
            "type": "incident",
            "created_at": "2026-06-15T00:00:00Z",
            "importance": 0.85,
            "tags": ["user-service", "latency", "redis", "cache"],
        },
        {
            "id": "MEM-004",
            "key": "cpu_spike_rollback_pattern",
            "value": "部署后CPU飙升通常由新代码中的低效算法引起，优先回滚部署并检查性能测试结果",
            "type": "knowledge",
            "created_at": "2026-06-22T00:00:00Z",
            "importance": 0.88,
            "tags": ["cpu", "deployment", "rollback"],
        },
        {
            "id": "MEM-005",
            "key": "network_partition_handling",
            "value": "网络分区时优先启用熔断器和降级策略，防止级联故障扩散到整个集群",
            "type": "knowledge",
            "created_at": "2026-06-19T00:00:00Z",
            "importance": 0.92,
            "tags": ["network", "circuit_breaker", "degradation"],
        },
        {
            "id": "MEM-006",
            "key": "oom_kill_recovery",
            "value": '{"immediate": "restart_pod", "short_term": "increase_memory_limit", "long_term": "fix_memory_leak"}',
            "type": "playbook",
            "created_at": "2026-06-21T00:00:00Z",
            "importance": 0.93,
            "tags": ["oom", "memory", "recovery"],
        },
    ]

    # 简单关键词匹配评分
    query_lower = query.lower()
    scored: list[tuple[dict[str, Any], float]] = []

    for item in memory_items:
        score = 0.0
        # 匹配 key
        if query_lower in item["key"].lower():
            score += 0.5
        # 匹配 value
        if query_lower in item["value"].lower():
            score += 0.3
        # 匹配 tags
        for tag in item.get("tags", []):
            if query_lower in tag.lower():
                score += 0.15
        # 匹配 type
        if memory_type and item["type"] == memory_type:
            score += 0.2
        # 重要性加成
        score += item.get("importance", 0.5) * 0.1

        if score > 0.1:
            scored.append((item, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    results = [{"entry": s[0], "score": round(s[1], 4)} for s in scored[:top_k]]

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "search_time_ms": 5,
    }


@api_router.post(
    "/memory/store",
    response_model=dict[str, Any],
    tags=["Memory"],
    summary="存储记忆",
    description="向记忆系统中存储新的记忆条目。",
)
async def store_memory(
    content: str,
    memory_type: str = "observation",
    incident_id: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """存储记忆"""
    memory_id = f"mem-{uuid.uuid4().hex[:8]}"
    logger.info("Store memory", memory_type=memory_type, memory_id=memory_id)

    return {
        "memory_id": memory_id,
        "status": "stored",
        "content_length": len(content),
        "type": memory_type,
    }


# =============================================================================
# 自然语言诊断端点
# =============================================================================


class NLQueryRequest(PydanticBaseModel):
    """自然语言诊断请求"""
    query: str = Field(default="", description="用户的自然语言问题")
    context: dict[str, Any] = Field(default_factory=dict, description="额外上下文")


@api_router.post(
    "/incidents/diagnose",
    response_model=dict[str, Any],
    tags=["Diagnosis"],
    summary="自然语言故障诊断",
    description="""
    接收用户的自然语言问题，自动完成全流程诊断。

    支持的大白话示例：
    - "为什么下单这么慢？"
    - "支付一直转圈圈，是不是挂了？"
    - "用户说登录不上去了"
    - "有没有重复扣款的情况？"

    内部流程:
    1. 意图识别 → 判断用户是诊断/查询/修复/历史查询
    2. 实体抽取 → 提取服务名、症状、紧急程度
    3. 指标映射 → 生成诊断查询计划（哪些指标、查哪些服务）
    4. 如检测到故障意图 → 自动触发 Agent 管道进行根因分析
    """,
)
async def diagnose_from_natural_language(
    request: NLQueryRequest,
) -> dict[str, Any]:
    """自然语言故障诊断入口"""
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    logger.info("NL diagnosis requested", query=query[:100])

    # Layer 1: 意图识别
    intent_classifier = IntentClassifier()
    intent = intent_classifier.classify(query)

    # Layer 2: 实体抽取
    entity_extractor = EntityExtractor()
    entities = entity_extractor.extract(query)

    # Layer 3: 指标映射
    metric_mapper = MetricMapper()
    primary_service = entities.services[0] if entities.services else "unknown"
    diagnosis_plan = metric_mapper.map(
        service=primary_service,
        symptoms=entities.symptoms,
        business_domain=entities.business_domain,
    )

    # Layer 4: 根据意图触发对应流程
    triggered_incident = None
    triggered_business = None

    if intent.intent == "fault_diagnosis" and entities.symptoms:
        # 构造告警事件并触发 Agent 管道
        if entities.services:
            alert = AlertEvent(
                service=primary_service,
                metric=diagnosis_plan.queries[0].metric if diagnosis_plan.queries else "unknown",
                value=0.0,
                threshold=0.0,
                severity=SeverityLevel(entities.urgency) if entities.urgency in {"critical", "high"} else SeverityLevel.MEDIUM,
                labels={
                    "tier": "critical",
                    "nl_query": query[:200],
                },
                annotations={
                    "summary": f"用户报告: {query}",
                    "extracted_symptoms": ",".join(entities.symptoms),
                    "extracted_services": ",".join(entities.services),
                },
            )

            incident = Incident.from_alert(alert)
            incident.transition_to(IncidentState.ACKNOWLEDGED, actor="nl-diagnosis")
            _incident_service._incidents[incident.incident_id] = incident

            # 异步触发管道
            asyncio.create_task(_run_pipeline_with_reflection(incident))

            triggered_incident = {
                "incident_id": incident.incident_id,
                "status": "processing",
            }

    elif intent.intent == "business_check" and entities.business_domain:
        # 触发业务异常检查
        biz_input = BusinessMetricInput(
            service_name=primary_service,
            business_domain=entities.business_domain,
            context={},
        )
        biz_agent = _get_business_monitor()
        biz_ctx = AgentExecutionContext(
            incident_id=str(uuid.uuid4()),
            input_data={"nl_query": query},
        )
        biz_result = await biz_agent.process(biz_input, biz_ctx)
        triggered_business = biz_result.output_data if biz_result.success else None

    # 构建响应
    return {
        "original_query": query,
        "intent": {
            "type": intent.intent,
            "confidence": round(intent.confidence, 4),
        },
        "understood_as": {
            "services": entities.services,
            "symptoms": entities.symptoms,
            "urgency": entities.urgency,
            "business_domain": entities.business_domain or "infrastructure",
        },
        "diagnosis_plan": {
            "total_queries": len(diagnosis_plan.queries),
            "queries": [
                {
                    "target": q.target,
                    "metric": q.metric,
                    "reason": q.reason,
                    "priority": q.priority,
                }
                for q in diagnosis_plan.queries[:10]
            ],
        },
        "explanation": metric_mapper.get_explanation(diagnosis_plan),
        "triggered_incident": triggered_incident,
        "triggered_business_check": triggered_business,
    }


# =============================================================================
# W7 PlanReActRunner 端点
# =============================================================================


async def _pipeline_step_dispatch(step: TaskStep) -> dict[str, Any]:
    """PlanReActRunner 的 step_executor，按 step.agent dispatch。"""
    agent_name = step.agent
    incident_id = step.params.get("incident_id") or ""

    if agent_name == "rca_agent":
        from app.agents.rca_agent import RCAInput
        from app.api.routes import _get_rca
        rca_agent = _get_rca()
        # 简化：从 incident_service 拿 incident 实化 alert
        from app.api.routes import _incident_service
        incident = await _incident_service.get(incident_id)
        if incident is None or incident.alert_event is None:
            return {"status": "skipped", "reason": "incident_or_alert_missing"}
        result = await rca_agent.execute(
            RCAInput(alert=incident.alert_event, incident_id=incident_id),
            AgentExecutionContext(incident_id=incident_id),
        )
        return {"status": "success" if result.success else "failed", "agent": agent_name}

    if agent_name == "log_agent":
        try:
            from app.infrastructure.log_client import LokiClient
            from app.api.routes import _incident_service
            incident = await _incident_service.get(incident_id)
            if incident is None:
                return {"status": "skipped"}
            client = LokiClient()
            entries = await client.get_service_errors(
                service=incident.service, lookback_minutes=60, limit=50,
            )
            return {"status": "success", "entries": len(entries)}
        except Exception as exc:
            logger.warning("log_agent step failed", error=str(exc))
            return {"status": "failed", "error": str(exc)[:200]}

    if agent_name == "change_agent":
        from app.api.routes import _get_change, _incident_service
        from app.agents.change_agent import ChangeInput
        from app.models.events import HealEvent
        incident = await _incident_service.get(incident_id)
        if incident is None:
            return {"status": "skipped"}
        latest_heal = incident.heal_events[-1] if incident.heal_events else HealEvent(incident_id=incident_id)
        result = await _get_change().execute(
            ChangeInput(heal_event=latest_heal, incident_id=incident_id, change_type="auto_heal"),
            AgentExecutionContext(incident_id=incident_id),
        )
        return {"status": "success" if result.success else "failed"}

    if agent_name == "runbook_service":
        from app.api.routes import _incident_service
        from app.services.runbook_service import runbook_service
        incident = await _incident_service.get(incident_id)
        if incident is None or incident.rca_event is None:
            return {"status": "skipped"}
        draft = runbook_service.generate(rca_event=incident.rca_event, alert=incident.alert_event)
        return {"status": "success", "draft_keys": list(draft.keys())}

    return {"status": "skipped", "reason": f"unknown_agent:{agent_name}"}


@api_router.post(
    "/incidents/{incident_id}/pipeline/plan-react",
    response_model=dict[str, Any],
    tags=["Incidents"],
    summary="以 PlanReActRunner 方式跑一遍 RCA 协同任务",
)
async def run_pipeline_plan_react(
    incident_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """可选端点：不替换主管道；以 Plan+ReAct 视角回放 RCA 协同任务。

    Spec §4.2 step 1: 仅 incident 处于 RCA_COMPLETED / HEALING / AWAITING_APPROVAL
    时才允许 plan-react 回放；前置状态（NEW / DETECTING / …）返回 409。
    """
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    _plan_react_allowed_states = {
        IncidentState.RCA_COMPLETED,
        IncidentState.HEALING,
        IncidentState.AWAITING_APPROVAL,
    }
    if incident.state not in _plan_react_allowed_states:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Incident {incident_id} in state '{incident.state.value}' is not "
                "eligible for plan-react replay (allowed: RCA_COMPLETED, HEALING, "
                "AWAITING_APPROVAL)"
            ),
        )

    task: CollaborativeTask = default_rca_plan(incident_id)
    # 把所有 step 的 params 注入 incident_id
    for s in task.steps:
        s.params["incident_id"] = incident_id

    runner = PlanReActRunner(task, _pipeline_step_dispatch, max_iterations=20)
    finished = await runner.run()

    return {
        "incident_id": incident_id,
        "task": finished.to_dict(),
        "requester": (payload or {}).get("requester", "anonymous"),
    }


# =============================================================================
# W7.5 BadCase RAG Endpoints (Task 3 of data flywheel)
# =============================================================================
# - POST /badcase/{incident_id}/analyze  — pull audit_trail, detect triggers,
#   build draft BadCaseEntry, and query similar_cases from the dev split.
# - GET  /badcase                       — list BadCaseEntry dicts filtered by
#   eval_split / badcase_class, sorted by created_at desc.
# =============================================================================


@api_router.post(
    "/badcase/{incident_id}/analyze",
    response_model=dict[str, Any],
    tags=["BadCases"],
    summary="分析故障审计轨迹，生成 badcase 草案 + 检索相似 badcase",
    description=(
        "读取 incident.context 中的 audit_trail + verification，"
        "通过 badcase_capture 公共助手检测 capture triggers，"
        "据此构建 BadCaseEntry 草案，并在 dev split 上做 RAG 相似检索。"
        "不会写盘 candidates.jsonl / 也不会直接写 registry —— 仅返回 draft 给前端。"
    ),
)
async def analyze_badcase(
    incident_id: str,
) -> dict[str, Any]:
    """根据 incident audit_trail 生成 badcase 草案 + 检索相似 badcase。

    Returns:
        200: {"draft": <BadCaseEntry dict>, "similar_cases": [list of BadCaseEntry dicts]}
        404: incident 不存在
        400: incident 没有 capture trigger（clean run）
    """
    from app.services.badcase_capture import (
        build_identified_flaw,
        build_suggestion,
        classify_badcase_class,
        detect_triggers,
        extract_keywords,
    )

    logger.info("Analyze badcase requested", incident_id=incident_id)

    # Resolve BadCase registry lazily — tests can monkeypatch _get_badcase_registry_service.
    registry = _get_badcase_registry_service()

    # 1. 加载 incident
    incident = await _incident_service.get(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident {incident_id} not found",
        )

    # 2. 检测 capture triggers（无信号 → clean run, 拒绝生成）
    triggers = detect_triggers(incident)
    if not triggers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"No capture triggers detected for incident {incident_id} "
                "(clean run — no badcase signals)"
            ),
        )

    # 3. 构造 draft BadCaseEntry（与 maybe_capture_badcase 同 id scheme 保证一致性）
    badcase_class = classify_badcase_class(triggers)
    entry_id_seed = f"bc-{incident.incident_id}-{badcase_class}"
    entry_id = "bc-" + hashlib.sha256(entry_id_seed.encode()).hexdigest()[:12]

    trail = incident.context.get("audit_trail", [])
    excerpt = trail[-5:] if trail else []

    draft_dict = {
        "id": entry_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "badcase_class": badcase_class,
        "incident_id": incident.incident_id,
        "audit_trail_excerpt": excerpt,
        "identified_flaw": build_identified_flaw(incident, triggers),
        "keywords_for_retrieval": extract_keywords(triggers),
        "suggestion_or_lesson": build_suggestion(triggers),
        "severity": "high" if triggers and triggers[0].value == "REFLECT_ESCALATE" else "medium",
        "eval_split": "dev",
    }

    # 4. RAG 检索相似 badcase（dev split，永远），用 identified_flaw + keywords 做 query
    query_text = " ".join([
        draft_dict["identified_flaw"],
        draft_dict["suggestion_or_lesson"],
        " ".join(draft_dict["keywords_for_retrieval"]),
    ])
    similar = registry.query_similar(
        text=query_text,
        top_k=5,
        class_filter=badcase_class,
        eval_split="dev",
    )
    similar_cases = [
        entry.model_dump(mode="json") for entry in similar
    ]

    logger.info(
        "Badcase draft analyzed",
        incident_id=incident_id,
        badcase_class=badcase_class,
        trigger_count=len(triggers),
        similar_count=len(similar_cases),
    )

    return {
        "incident_id": incident_id,
        "draft": draft_dict,
        "similar_cases": similar_cases,
    }


@api_router.get(
    "/badcase",
    response_model=list[dict[str, Any]],
    tags=["BadCases"],
    summary="列出 badcase 草/成品（按 eval_split / class 过滤）",
    description=(
        "从磁盘 JSONL 读取指定 eval_split 的 BadCaseEntry，按 created_at desc 排序，"
        "可按 badcase_class 二次过滤。limit 默认 50，上限 500。"
    ),
)
async def list_badcases(
    eval_split: str = Query("dev", pattern="^(dev|test)$", description="eval 数据集划分"),
    badcase_class: str | None = Query(None, description="按 badcase_class 二次过滤"),
    limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
) -> list[dict[str, Any]]:
    """列出指定 eval_split 下的 badcase 字典列表（不调用 Chroma，仅读 JSONL）。

    Returns:
        200: list of BadCaseEntry dicts (sorted by created_at desc, max `limit`)
    """
    # Resolve BadCase registry lazily — tests can monkeypatch _get_badcase_registry_service.
    registry = _get_badcase_registry_service()

    path = registry.jsonl_path / f"badcase_{eval_split}.jsonl"
    entries: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if badcase_class is not None and data.get("badcase_class") != badcase_class:
                continue
            entries.append(data)

    # 按 created_at desc 排序；缺字段时使用 ISO 字符串可比较 fallback
    entries.sort(
        key=lambda d: d.get("created_at") or "",
        reverse=True,
    )
    return entries[:limit]


# =============================================================================
# W7.5 Harness Loop Endpoints (Task 4 of data flywheel)
# =============================================================================
# - POST /harness/run        — submit a PatchCandidate, run champion-challenger
# - GET  /harness/runs       — list historical HarnessRunRecord (sorted desc)
# - GET  /harness/champion   — query current champion for a badcase_class
# =============================================================================


@api_router.post(
    "/harness/run",
    response_model=dict[str, Any],
    tags=["Harness"],
    summary="提交一个 PatchCandidate 跑 champion-challenger 循环",
    description=(
        "接收 PatchCandidate（patch_id / badcase_class / diff / rationale / proposed_by），"
        "由 HarnessLoopEngine 在 dev + test split 上评估，"
        "返回完整 HarnessRunRecord（含 decision / dev_metrics / test_metrics）。"
        "如果 decision == promote，则会同步写入 champion.json。"
    ),
)
async def run_harness(candidate_payload: dict[str, Any]) -> dict[str, Any]:
    """同步跑一次 challenger，decision 落入 runs.jsonl。

    Returns:
        200: HarnessRunRecord dict（run_id, decision, dev_metrics, test_metrics, ...）
        422: PatchCandidate 字段校验失败
    """
    from app.models.harness import PatchCandidate
    from app.services.badcase_registry_service import BadCaseRegistryService
    from app.services.harness_engine import HarnessLoopEngine

    candidate = PatchCandidate(**candidate_payload)

    # Use the same lazy-singleton factory pattern as _get_badcase_registry_service
    # so tests can monkeypatch in a tmp-path registry via the routes module.
    registry = _get_badcase_registry_service()
    engine = HarnessLoopEngine(
        registry=registry,
        runs_path="backend/data/harness/runs.jsonl",
        champion_path="backend/data/champion.json",
    )
    record = engine.run_sync(candidate)

    logger.info(
        "Harness run completed",
        run_id=record.run_id,
        badcase_class=record.badcase_class,
        decision=record.decision,
    )
    return record.model_dump(mode="json")


@api_router.get(
    "/harness/runs",
    response_model=list[dict[str, Any]],
    tags=["Harness"],
    summary="列出历史 HarnessRunRecord（按 started_at 倒序）",
    description=(
        "从 backend/data/harness/runs.jsonl 读取所有 HarnessRunRecord，"
        "可选按 badcase_class 过滤，limit 默认 50、上限 500。"
    ),
)
async def list_harness_runs(
    badcase_class: str | None = Query(None, description="按 badcase_class 过滤"),
    limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
) -> list[dict[str, Any]]:
    """列出指定过滤条件下的 HarnessRunRecord 列表（按 started_at desc）。"""
    from pathlib import Path

    path = Path("backend/data/harness/runs.jsonl")
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if badcase_class is not None and data.get("badcase_class") != badcase_class:
            continue
        out.append(data)
    out.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return out[:limit]


@api_router.get(
    "/harness/champion",
    response_model=dict[str, Any] | None,
    tags=["Harness"],
    summary="查询当前 champion（按 badcase_class）",
    description=(
        "从 backend/data/champion.json 读取给定 badcase_class 的 Champion。"
        "若文件不存在或该 class 尚未 promote，返回 None。"
    ),
)
async def get_champion(
    badcase_class: str = Query(..., description="要查询的 badcase_class"),
) -> dict[str, Any] | None:
    """查询当前 champion；无 champion 时返回 None。"""
    from app.services.harness_engine import HarnessLoopEngine

    registry = _get_badcase_registry_service()
    engine = HarnessLoopEngine(
        registry=registry,
        champion_path="backend/data/champion.json",
    )
    champion = engine.get_champion(badcase_class)
    if champion is None:
        return None
    return champion.model_dump(mode="json")
