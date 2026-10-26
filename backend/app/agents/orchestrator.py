"""
AIOps Agent Platform - LangGraph Orchestrator

使用 LangGraph 编排多个 Agent 的协作流程，定义故障处理的完整状态机。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from app.agents.base import AgentExecutionContext
from app.agents.change_agent import ChangeAgent, ChangeInput
from app.agents.heal_agent import HealAgent, HealInput
from app.agents.rca_agent import RCAAgent, RCAInput
from app.agents._evolution import EvolutionPipeline, Feedback
from app.models.agent import AgentState, AgentStatus
from app.models.events import (
    AlertEvent,
    ChangeEvent,
    HealEvent,
    RCAEvent,
    SeverityLevel,
)
from app.models.incident import Incident, IncidentPhase, IncidentState
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OrchestratorState(str, Enum):
    """编排器状态"""
    IDLE = "idle"                    # 空闲等待
    RECEIVING_ALERT = "receiving_alert"   # 接收告警
    TRIAGING = "triaging"            # 分类分级
    RUNNING_RCA = "running_rca"      # 根因分析
    DECIDING_ACTION = "deciding_action"   # 决策
    EXECUTING_HEAL = "executing_heal"     # 执行自愈
    AWAITING_APPROVAL = "awaiting_approval"  # 等待审批
    VERIFYING = "verifying"          # 验证结果
    COMPLETED = "completed"          # 完成
    ESCALATED = "escalated"          # 已升级
    ERROR = "error"                  # 错误


class Orchestrator:
    """
    LangGraph 编排器

    使用 LangGraph 定义故障处理的完整状态机和 Agent 协作流程。

    流程图:
    
    [Alert] -> RECEIVING_ALERT -> TRIAGING -> RUNNING_RCA -> DECIDING_ACTION
                                                                   |
                    +----------------------------------------------+
                    |                                              |
              [低风险] -> EXECUTING_HEAL -> VERIFYING -> COMPLETED  |
                    |                                              |
              [高风险] -> AWAITING_APPROVAL -> ...                 |
                    |                                              |
              [无法处理] -> ESCALATED                              |
    """

    def __init__(self) -> None:
        self._state = OrchestratorState.IDLE
        self._current_incident: Incident | None = None
        self._graph: Any = None  # LangGraph 图实例
        self._compiled_graph: Any = None  # 编译后的图
        # v15 M1：自进化 pipeline（3 路径：Playbook 沉淀 / Badcase 捕获 / 反馈调优）
        self._evolution_pipeline = EvolutionPipeline()

    @property
    def state(self) -> OrchestratorState:
        """当前编排器状态"""
        return self._state

    @property
    def current_incident(self) -> Incident | None:
        """当前处理的故障"""
        return self._current_incident

    def build_graph(self) -> Any:
        """
        构建 LangGraph 状态机图

        定义各节点（Agent）和边（流转条件）。

        Returns:
            Any: 构建好的图
        """
        try:
            from langgraph.graph import StateGraph, END

            # 定义状态类型
            class GraphState:
                """LangGraph 状态定义"""
                def __init__(self) -> None:
                    self.incident: Incident | None = None
                    self.current_phase: str = ""
                    self.agent_results: dict[str, Any] = {}
                    self.errors: list[str] = []
                    self.completed: bool = False
                    self.escalated: bool = False

            # 创建图
            workflow = StateGraph(GraphState)

            # 添加节点
            workflow.add_node("receive_alert", self._node_receive_alert)
            workflow.add_node("triage", self._node_triage)
            workflow.add_node("run_rca", self._node_run_rca)
            workflow.add_node("decide_action", self._node_decide_action)
            workflow.add_node("execute_heal", self._node_execute_heal)
            workflow.add_node("request_approval", self._node_request_approval)
            workflow.add_node("verify_fix", self._node_verify_fix)
            workflow.add_node("escalate", self._node_escalate)
            workflow.add_node("complete", self._node_complete)

            # 添加边 - 定义流转逻辑
            workflow.set_entry_point("receive_alert")
            workflow.add_edge("receive_alert", "triage")
            workflow.add_edge("triage", "run_rca")
            workflow.add_edge("run_rca", "decide_action")

            # 决策节点条件分支
            workflow.add_conditional_edges(
                "decide_action",
                self._edge_decide_action,
                {
                    "heal": "execute_heal",
                    "approve": "request_approval",
                    "escalate": "escalate",
                    "complete": "complete",
                },
            )

            workflow.add_edge("execute_heal", "verify_fix")

            # 验证节点条件分支
            workflow.add_conditional_edges(
                "verify_fix",
                self._edge_verify_fix,
                {
                    "completed": "complete",
                    "retry": "decide_action",
                    "escalate": "escalate",
                },
            )

            workflow.add_edge("request_approval", "verify_fix")
            workflow.add_edge("escalate", END)
            workflow.add_edge("complete", END)

            self._graph = workflow
            self._compiled_graph = workflow.compile()

            logger.info("LangGraph orchestrator built successfully")
            return self._compiled_graph

        except ImportError:
            logger.warning("LangGraph not available, running in fallback mode")
            return None

    async def process_alert(self, alert: AlertEvent) -> Incident:
        """
        处理告警入口

        Args:
            alert: 告警事件

        Returns:
            Incident: 创建的故障实例
        """
        logger.info(
            "Orchestrator processing alert",
            service=alert.service,
            severity=alert.severity.value,
        )

        self._state = OrchestratorState.RECEIVING_ALERT

        # 创建故障实例
        incident = Incident.from_alert(alert)
        self._current_incident = incident

        if self._compiled_graph:
            try:
                initial_state = {
                    "incident": incident,
                    "current_phase": "detection",
                    "agent_results": {},
                    "errors": [],
                    "completed": False,
                    "escalated": False,
                }
                # 真正调用 LangGraph
                result = await self._compiled_graph.ainvoke(initial_state)
                # 把 agent_results 写回 incident context
                for key, value in result.get("agent_results", {}).items():
                    incident.context[f"orchestrator.{key}"] = value
                if result.get("escalated"):
                    incident.transition_to(IncidentState.ESCALATED, actor="orchestrator")
                    self._state = OrchestratorState.ESCALATED
                elif result.get("completed"):
                    incident.transition_to(IncidentState.RESOLVED, actor="orchestrator")
                    self._state = OrchestratorState.COMPLETED
                else:
                    # graph 执行未明确完成，转 sequential 兜底
                    await self._sequential_process(incident)
            except Exception as e:
                logger.error("LangGraph execution failed, falling back to sequential", error=str(e))
                await self._sequential_process(incident)
        else:
            # LangGraph 不可用，sequential 兜底
            await self._sequential_process(incident)

        return incident

    async def _sequential_process(self, incident: Incident) -> None:
        """
        顺序处理流程（Fallback 模式）

        当 LangGraph 不可用时，按顺序执行各 Agent。
        每个 Agent 的真实结果写回 incident 对应字段。

        Args:
            incident: 故障实例
        """
        ctx = AgentExecutionContext(
            incident_id=incident.incident_id,
            metadata={"correlation_id": incident.alert_event.correlation_id if incident.alert_event else ""},
        )

        # Step 1: 分类分级（incident.from_alert 已完成，此处仅 transition）
        self._state = OrchestratorState.TRIAGING
        incident.transition_to(IncidentState.ACKNOWLEDGED, actor="orchestrator")
        incident.add_timeline_entry(
            phase=IncidentPhase.TRIAGE,
            state=IncidentState.ACKNOWLEDGED,
            actor="orchestrator",
            action="Alert triaged",
        )

        # Step 2: 根因分析
        self._state = OrchestratorState.RUNNING_RCA
        incident.transition_to(IncidentState.RCA_IN_PROGRESS, actor="rca_agent")
        incident.add_timeline_entry(
            phase=IncidentPhase.RCA,
            state=IncidentState.RCA_IN_PROGRESS,
            actor="rca_agent",
            action="Running root cause analysis",
        )

        try:
            rca_agent = RCAAgent()
            rca_result = await rca_agent.process(
                RCAInput(
                    alert=incident.alert_event,
                    incident_id=incident.incident_id,
                    lookback_minutes=60,
                    max_hops=3,
                ),
                context=ctx,
            )
            if rca_result.success and rca_result.output_data:
                rca_event_dict = rca_result.output_data.get("rca_event", {})
                if rca_event_dict:
                    incident.rca_event = (
                        RCAEvent(**rca_event_dict)
                        if isinstance(rca_event_dict, dict)
                        else rca_event_dict
                    )
                    incident.context["rca_root_cause"] = rca_result.output_data.get("root_cause", "unknown")
                    # RCA 成功后自动生成 Runbook 草案（含日志证据引用）
                    try:
                        from app.services.runbook_service import runbook_service
                        incident.context["runbook_draft"] = runbook_service.generate(
                            rca_event=incident.rca_event,
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
            incident.transition_to(IncidentState.RCA_COMPLETED, actor="rca_agent")
        except Exception as e:
            logger.error("RCA agent failed in sequential", error=str(e))
            incident.transition_to(IncidentState.ESCALATED, actor="rca_agent")

        # Step 3: 决策 + 自愈（基于 severity）
        self._state = OrchestratorState.DECIDING_ACTION
        if incident.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH):
            # 高风险 → 必须审批，跳过直接自愈
            self._state = OrchestratorState.AWAITING_APPROVAL
            incident.transition_to(IncidentState.AWAITING_APPROVAL, actor="change_agent")
        else:
            # 低/中风险 → 自愈
            self._state = OrchestratorState.EXECUTING_HEAL
            incident.transition_to(IncidentState.HEALING, actor="heal_agent")
            try:
                heal_agent = HealAgent()
                heal_result = await heal_agent.process(
                    HealInput(
                        rca_event=incident.rca_event or RCAEvent(incident_id=incident.incident_id),
                        incident_id=incident.incident_id,
                        dry_run=True,
                    ),
                    context=ctx,
                )
                if heal_result.success and heal_result.output_data:
                    heal_event_dict = heal_result.output_data.get("heal_event", {})
                    if heal_event_dict:
                        heal_event = (
                            HealEvent(**heal_event_dict)
                            if isinstance(heal_event_dict, dict)
                            else heal_event_dict
                        )
                        incident.heal_events.append(heal_event)
                        incident.context["heal_action"] = heal_result.output_data.get("action", "")
                incident.transition_to(IncidentState.HEALED, actor="heal_agent")
            except Exception as e:
                logger.error("Heal agent failed in sequential", error=str(e))
                incident.transition_to(IncidentState.AWAITING_APPROVAL, actor="heal_agent")

        # Step 4: 变更审批（无论前面是否走过 heal，都需要 Change Agent 决策）
        try:
            change_agent = ChangeAgent()
            latest_heal = incident.heal_events[-1] if incident.heal_events else None
            change_input = ChangeInput(
                heal_event=latest_heal or HealEvent(incident_id=incident.incident_id),
                incident_id=incident.incident_id,
                change_type="auto_heal",
                requester="orchestrator",
            )
            change_result = await change_agent.process(change_input, context=ctx)
            if change_result.success and change_result.output_data:
                change_event_dict = change_result.output_data.get("change_event", {})
                if change_event_dict:
                    change_event = (
                        ChangeEvent(**change_event_dict)
                        if isinstance(change_event_dict, dict)
                        else change_event_dict
                    )
                    incident.change_events.append(change_event)
                    incident.context["approval_status"] = change_result.output_data.get(
                        "approval_status", "pending"
                    )
        except Exception as e:
            logger.error("Change agent failed in sequential", error=str(e))

        # Step 5: 验证（标记完成）
        self._state = OrchestratorState.VERIFYING
        incident.transition_to(IncidentState.RESOLVED, actor="orchestrator")
        self._state = OrchestratorState.COMPLETED

        incident.add_timeline_entry(
            phase=IncidentPhase.RESOLUTION,
            state=IncidentState.RESOLVED,
            actor="orchestrator",
            action="Incident resolved (sequential fallback)",
        )

        # ===== W7 末尾反思（与 LangGraph _node_verify_fix 一致） =====
        await self.run_sequential_with_reflection(incident)

    # === LangGraph 节点方法（真实调用 Agent）===

    async def _node_receive_alert(self, state: dict[str, Any]) -> dict[str, Any]:
        """接收告警节点"""
        self._state = OrchestratorState.RECEIVING_ALERT
        logger.info("Graph node: receive_alert")
        if state.get("incident"):
            state["incident"].transition_to(IncidentState.ACKNOWLEDGED, actor="orchestrator")
        return state

    async def _node_triage(self, state: dict[str, Any]) -> dict[str, Any]:
        """分类分级节点 — 标记 incident 已被 orchestrator 接收"""
        self._state = OrchestratorState.TRIAGING
        logger.info("Graph node: triage")
        incident = state.get("incident")
        if incident:
            incident.transition_to(IncidentState.ACKNOWLEDGED, actor="orchestrator")
        return state

    async def _node_run_rca(self, state: dict[str, Any]) -> dict[str, Any]:
        """根因分析节点"""
        self._state = OrchestratorState.RUNNING_RCA
        logger.info("Graph node: run_rca")
        incident = state.get("incident")
        if not incident:
            return state
        incident.transition_to(IncidentState.RCA_IN_PROGRESS, actor="rca_agent")
        try:
            rca_agent = RCAAgent()
            ctx = AgentExecutionContext(incident_id=incident.incident_id)
            result = await rca_agent.process(
                RCAInput(alert=incident.alert_event, incident_id=incident.incident_id),
                context=ctx,
            )
            state["agent_results"]["rca"] = result.output_data or {}
            rca_event_dict = result.output_data.get("rca_event", {}) if result.output_data else {}
            if rca_event_dict:
                incident.rca_event = (
                    RCAEvent(**rca_event_dict)
                    if isinstance(rca_event_dict, dict)
                    else rca_event_dict
                )
            incident.transition_to(IncidentState.RCA_COMPLETED, actor="rca_agent")

            # ===== v15 M1：三路径自进化（路径 1: RCA→Playbook） =====
            self._evolve_from_rca(incident, rca_event_dict)
        except Exception as e:
            logger.error("RCA node error", error=str(e))
            state["errors"].append(f"rca: {e}")
        return state

    # ==================== v15 M1：三路径自进化 ====================

    def _evolve_from_rca(
        self,
        incident: Incident,
        rca_event_dict: dict[str, Any],
    ) -> None:
        """M1 路径 1: 高置信度 RCA 自动沉淀为 Playbook

        借鉴 Devix §6.3 - 成功的 RCA 应该自动沉淀，避免下次重复劳动。
        """
        if not rca_event_dict:
            return

        root_cause = rca_event_dict.get("root_cause", "")
        confidence = rca_event_dict.get("confidence", 0.0)
        evidence = rca_event_dict.get("evidence", {})
        evidence_list = []
        if isinstance(evidence, dict):
            for k, v in evidence.items():
                if isinstance(v, str):
                    evidence_list.append(f"{k}:{v}")
                elif isinstance(v, list):
                    for item in v[:3]:
                        if isinstance(item, str):
                            evidence_list.append(f"{k}:{item}")

        # heal_action（如果存在）
        heal_action = ""
        suggested_actions = rca_event_dict.get("suggested_actions", [])
        if suggested_actions and isinstance(suggested_actions, list):
            heal_action = str(suggested_actions[0]) if suggested_actions else ""

        rca_result = {
            "incident_id": incident.incident_id,
            "root_cause": root_cause,
            "confidence": confidence,
            "evidence": evidence_list,
            "heal_action": heal_action,
        }

        evolution_result = self._evolution_pipeline.evolve(
            rca_result=rca_result,
            feedback=None,
            failed=False,
        )

        if evolution_result.path1_executed and evolution_result.playbook:
            logger.info(
                "Auto-promoted RCA to Playbook",
                incident_id=incident.incident_id,
                playbook_name=evolution_result.playbook.name,
                confidence=confidence,
            )

    def record_incident_failure(self, incident_id: str, reason: str) -> None:
        """M1 路径 2: 记录失败的 incident 用于 Badcase 分析

        调用方：incident 失败时调用
        """
        self._evolution_pipeline.badcase_capture.capture_failed(
            incident_id=incident_id,
            root_cause=reason,
            confidence=0.0,
            failure_reason="execution_failed",
        )

    def record_human_feedback(self, feedback: Feedback) -> None:
        """M1 路径 3: 记录人工反馈用于 prompt 调优

        调用方：人工评分接口收到反馈时调用
        """
        self._evolution_pipeline.prompt_optimizer.apply_feedback(feedback)

    async def _node_decide_action(self, state: dict[str, Any]) -> dict[str, Any]:
        """决策节点 — 不调 Agent，仅记录决策依据"""
        self._state = OrchestratorState.DECIDING_ACTION
        logger.info("Graph node: decide_action")
        return state

    async def _node_execute_heal(self, state: dict[str, Any]) -> dict[str, Any]:
        """执行自愈节点"""
        self._state = OrchestratorState.EXECUTING_HEAL
        logger.info("Graph node: execute_heal")
        incident = state.get("incident")
        if not incident:
            return state
        try:
            heal_agent = HealAgent()
            ctx = AgentExecutionContext(incident_id=incident.incident_id)
            rca_event = incident.rca_event or RCAEvent(incident_id=incident.incident_id)
            result = await heal_agent.process(
                HealInput(rca_event=rca_event, incident_id=incident.incident_id, dry_run=True),
                context=ctx,
            )
            state["agent_results"]["heal"] = result.output_data or {}
            heal_event_dict = result.output_data.get("heal_event", {}) if result.output_data else {}
            if heal_event_dict:
                incident.heal_events.append(
                    HealEvent(**heal_event_dict)
                    if isinstance(heal_event_dict, dict)
                    else heal_event_dict
                )
        except Exception as e:
            logger.error("Heal node error", error=str(e))
            state["errors"].append(f"heal: {e}")
        return state

    async def _node_request_approval(self, state: dict[str, Any]) -> dict[str, Any]:
        """请求审批节点"""
        self._state = OrchestratorState.AWAITING_APPROVAL
        logger.info("Graph node: request_approval")
        incident = state.get("incident")
        if not incident:
            return state
        try:
            change_agent = ChangeAgent()
            ctx = AgentExecutionContext(incident_id=incident.incident_id)
            latest_heal = incident.heal_events[-1] if incident.heal_events else None
            result = await change_agent.process(
                ChangeInput(
                    heal_event=latest_heal or HealEvent(incident_id=incident.incident_id),
                    incident_id=incident.incident_id,
                    change_type="auto_heal",
                    requester="orchestrator",
                ),
                context=ctx,
            )
            state["agent_results"]["change"] = result.output_data or {}
            change_event_dict = result.output_data.get("change_event", {}) if result.output_data else {}
            if change_event_dict:
                incident.change_events.append(
                    ChangeEvent(**change_event_dict)
                    if isinstance(change_event_dict, dict)
                    else change_event_dict
                )
        except Exception as e:
            logger.error("Change node error", error=str(e))
            state["errors"].append(f"change: {e}")
        return state

    async def _node_verify_fix(self, state: dict[str, Any]) -> dict[str, Any]:
        """验证修复节点（带反思）

        关键改造：从"默认 completed"升级为"结构化反思"：
        - 收集 verify 阶段的证据（置信度、症状恢复比例、工具失败次数）
        - 调用 reflect_on_verify() 做决策
        - 把反思结果（reason / suggestions / duration_ms）写入 audit_trail
        """
        from app.agents.reflection import (
            VerifyEvidence,
            append_audit_trail,
            reflect_on_verify,
        )

        self._state = OrchestratorState.VERIFYING
        logger.info("Graph node: verify_fix (with reflection)")
        incident = state.get("incident")

        # 1. 收集 verify 证据
        rca_result = state.get("agent_results", {}).get("rca", {})
        heal_result = state.get("agent_results", {}).get("heal", {})

        confidence = float(rca_result.get("confidence", 0.0) or 0.0)
        symptoms_resolved = int(heal_result.get("symptoms_resolved", 0))
        symptoms_total = int(heal_result.get("symptoms_total", 0))
        tool_failures = list(state.get("errors", []))

        evidence = VerifyEvidence(
            confidence=confidence,
            symptoms_resolved=symptoms_resolved,
            symptoms_total=symptoms_total,
            tool_failures=tool_failures,
        )

        # 2. 反思
        retry_count = int(state.get("verify_retry_count", 0))
        reflection = reflect_on_verify(evidence, retry_count=retry_count)

        # 3. 写入 audit_trail
        if incident:
            append_audit_trail(
                incident.context,
                step="verify_reflection",
                trigger=f"confidence={confidence:.2f}, resolved={symptoms_resolved}/{symptoms_total}",
                actions_taken=[
                    f"next_action={reflection.next_action}",
                    f"reason={reflection.reason}",
                    f"suggestions={reflection.suggestions}",
                ],
                outcome=reflection.reason_detail,
                duration_ms=reflection.duration_ms,
            )

        # 4. 应用反思结果
        if reflection.next_action == "complete":
            if incident:
                incident.transition_to(IncidentState.RESOLVED, actor="orchestrator")
            state["completed"] = True
        elif reflection.next_action == "retry":
            state["verify_retry_count"] = retry_count + 1
            state["reflection"] = reflection.reason_detail
            # 让 LangGraph 重走 decide_action 边
            state["completed"] = False
            state["escalated"] = False
        elif reflection.next_action == "escalate":
            if incident:
                incident.transition_to(IncidentState.ESCALATED, actor="orchestrator")
            state["escalated"] = True
            state["reflection"] = reflection.reason_detail

        logger.info(
            "Verify reflection",
            next_action=reflection.next_action,
            reason=reflection.reason,
            retry_count=state.get("verify_retry_count", 0),
            detail=reflection.reason_detail,
        )
        return state

    async def _node_escalate(self, state: dict[str, Any]) -> dict[str, Any]:
        """升级节点（审查报告中期补 #9：集成 oncall 排班 + 升级链路）"""
        self._state = OrchestratorState.ESCALATED
        logger.info("Graph node: escalate")
        state["escalated"] = True
        if state.get("incident"):
            incident = state["incident"]
            incident.transition_to(IncidentState.ESCALATED, actor="orchestrator")

            # 按 incident severity 找值班人（仅 CRITICAL/HIGH 主动通知）
            oncall_scheduler = state.get("oncall_scheduler")
            if oncall_scheduler is not None:
                severity = (
                    state.get("alert", {}).severity.value
                    if state.get("alert") and hasattr(state.get("alert"), "severity")
                    else "MEDIUM"
                )
                service_name = state.get("alert", {}).service if state.get("alert") else None
                assignment = oncall_scheduler.get_current_oncall(
                    service_name=service_name,
                    severity=severity,
                )
                if assignment is not None:
                    state["oncall_assignment"] = {
                        **assignment.to_dict(),
                        "incident_id": state.get("incident_id"),
                        "service_name": service_name,
                    }
                    logger.info(
                        "oncall notification",
                        user_id=assignment.user_id,
                        role=assignment.role,
                        escalation_level=assignment.escalation_level,
                        reason=assignment.reason,
                        incident_id=state.get("incident_id"),
                    )
                else:
                    state["oncall_assignment"] = None
        return state

    async def _node_complete(self, state: dict[str, Any]) -> dict[str, Any]:
        """完成节点"""
        self._state = OrchestratorState.COMPLETED
        logger.info("Graph node: complete")
        state["completed"] = True
        return state

    # === 条件边方法 ===

    def _edge_decide_action(self, state: dict[str, Any]) -> str:
        """
        决策条件边

        根据 RCA 结果和 severity 决定下一步：
        - 没有 incident → escalate
        - severity=CRITICAL 或 HIGH → approve（需审批）
        - 否则 → heal（直接自愈）
        - 已 completed → complete
        """
        incident = state.get("incident")
        if not incident:
            return "escalate"

        if state.get("completed"):
            return "complete"

        # 基于 severity 决策
        if incident.severity == SeverityLevel.CRITICAL:
            return "approve"
        elif incident.severity == SeverityLevel.LOW:
            return "heal"
        else:
            # HIGH/MEDIUM 默认走 heal（带 dry-run）
            return "heal"

    def _edge_verify_fix(self, state: dict[str, Any]) -> str:
        """
        验证条件边（已升级为基于反思结果）

        优先级：escalated > retry > completed
        """
        if state.get("escalated"):
            return "escalate"
        if state.get("verify_retry_count", 0) > 0:
            # 反思节点决定重试 → 回到 decide_action 重新规划
            return "retry"
        if state.get("completed"):
            return "completed"
        # 兜底：默认 completed（与 v2.0 行为兼容）
        return "completed"

    # === W7 顺序路径反思（fallback path）===

    async def run_sequential_with_reflection(self, incident: Incident) -> None:
        """
        在顺序处理流程末尾跑反思 + 写 audit_trail。

        与 LangGraph ``_node_verify_fix`` 行为一致，但不真正循环重试
        （sequential 语义：保证流程一次跑完，反思只决定是否升级）。

        决策映射：
        - ``complete`` → 保持已 transition 的 RESOLVED（无副作用）
        - ``retry``    → 仅记录 warning（顺序路径不支持真重试）
        - ``escalate`` → 覆盖状态为 ESCALATED 并写 warning
        """
        from app.agents.reflection import (
            VerifyEvidence,
            append_audit_trail,
            reflect_on_verify,
        )

        # 收集 verify 证据（顺序路径下没有"真实恢复比例"，用 confidence 兜底）
        rca = incident.rca_event
        heal = incident.heal_events[-1] if incident.heal_events else None
        confidence = getattr(rca, "confidence", 0.0) if rca else 0.0
        symptoms_resolved = 0
        symptoms_total = 0
        tool_failures: list[str] = []
        heal_passed = bool(
            heal and heal.dry_run_result.get("all_executable") if heal and heal.dry_run_result else False
        )
        if heal and not heal_passed:
            symptoms_total = 1  # 把 dry_run_failed 视作 1 个待恢复症状

        evidence = VerifyEvidence(
            confidence=float(confidence or 0.0),
            symptoms_resolved=symptoms_resolved,
            symptoms_total=symptoms_total,
            tool_failures=tool_failures,
        )
        reflection = reflect_on_verify(evidence, retry_count=0)

        append_audit_trail(
            incident.context,
            step="verify_reflection",
            trigger=f"confidence={evidence.confidence:.2f}",
            actions_taken=[
                f"next_action={reflection.next_action}",
                f"reason={reflection.reason}",
            ],
            outcome=reflection.reason_detail,
            duration_ms=reflection.duration_ms,
        )

        # 升级：sequential 也支持 escalate
        if reflection.next_action == "escalate":
            incident.transition_to(IncidentState.ESCALATED, actor="orchestrator")
            logger.warning(
                "Sequential verify escalated",
                incident_id=incident.incident_id,
                reason=reflection.reason,
            )
        elif reflection.next_action == "retry":
            logger.warning(
                "Sequential verify recommended retry (not looping)",
                incident_id=incident.incident_id,
                reason=reflection.reason,
            )

