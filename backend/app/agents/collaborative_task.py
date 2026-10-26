"""
AIOps Agent Platform - Collaborative Task Model (Plan+ReAct)

协同任务模型：把"诊断/证据/变更/Runbook"4 个子任务组织成
Plan → Act → Observe → Reflect 闭环，支持重试、中断恢复。

参考 sxdevops AIOpsExternalTask 设计，但保持本仓库单进程语义。
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

from app.utils.logging import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    DRAFT = "draft"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# ===== 任务数据结构 =====


@dataclass
class TaskStep:
    """协同任务中的一个步骤（可被并发/串行调度）。"""

    step_id: str
    name: str
    agent: str  # "diagnosis" / "evidence" / "change" / "runbook"
    action: str  # 调用方法名
    params: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: TaskStepStatus = TaskStepStatus.PENDING
    result: dict[str, Any] | None = None
    error: str = ""
    retries: int = 0
    max_retries: int = 2
    # 反思
    reflection: str = ""
    permanent_failure: bool = False  # 标记永久错误（不再重试）

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "agent": self.agent,
            "action": self.action,
            "params": self.params,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "reflection": self.reflection,
            "permanent_failure": self.permanent_failure,
        }


@dataclass
class CollaborativeTask:
    """协同任务：多个步骤按依赖关系执行。"""

    task_id: str
    incident_id: str
    title: str
    steps: list[TaskStep]
    status: TaskStatus = TaskStatus.DRAFT
    react_trace: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "incident_id": self.incident_id,
            "title": self.title,
            "status": self.status.value,
            "step_count": len(self.steps),
            "completed": sum(1 for s in self.steps if s.status == TaskStepStatus.SUCCESS),
            "failed": sum(1 for s in self.steps if s.status == TaskStepStatus.FAILED),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_summary(),
            "steps": [s.to_dict() for s in self.steps],
            "react_trace": self.react_trace,
            "metadata": self.metadata,
        }


# ===== Plan 阶段：默认步骤模板 =====


def default_rca_plan(incident_id: str) -> CollaborativeTask:
    """默认 RCA 协同任务计划：诊断 → 证据 → 变更评估 → Runbook 起草。

    步骤可并发/串行：
    - diagnosis → evidence / runbook (串行：先诊断再补充证据)
    - runbook 依赖 diagnosis 和 evidence（都要完成后才起草）
    - change 独立（任何时候都可评估风险）
    """
    s_diag = TaskStep(
        step_id="step-diagnosis",
        name="诊断",
        agent="rca_agent",
        action="process",
        params={},
    )
    s_evidence = TaskStep(
        step_id="step-evidence",
        name="补充证据",
        agent="log_agent",
        action="gather",
        params={"lookback_minutes": 60},
        depends_on=["step-diagnosis"],
    )
    s_change = TaskStep(
        step_id="step-change",
        name="变更风险评估",
        agent="change_agent",
        action="evaluate",
        params={},
    )
    s_runbook = TaskStep(
        step_id="step-runbook",
        name="Runbook 起草",
        agent="runbook_service",
        action="generate",
        params={},
        depends_on=["step-diagnosis", "step-evidence"],
    )
    return CollaborativeTask(
        task_id=f"ct-{uuid.uuid4().hex[:12]}",
        incident_id=incident_id,
        title=f"RCA 协同任务 #{incident_id[:8]}",
        steps=[s_diag, s_evidence, s_change, s_runbook],
    )


# ===== Plan+ReAct 主循环 =====


class PlanReActRunner:
    """执行协同任务：Plan 阶段已固定，Act/Observe/Reflect 循环到完成或终止。"""

    PERMANENT_ERROR_KEYWORDS = {
        "permission denied",
        "not found",
        "invalid input",
    }
    TERMINATION_CONDITIONS = {
        "all_steps_success",
        "critical_step_failed",
        "max_replan_reached",
        "user_interrupted",
    }

    def __init__(
        self,
        task: CollaborativeTask,
        step_executor: Callable[[TaskStep], Awaitable[dict[str, Any]]],
        max_iterations: int = 20,
    ) -> None:
        self.task = task
        self.step_executor = step_executor
        self.max_iterations = max_iterations
        self._interrupted = False
        self._iteration = 0

    def interrupt(self) -> None:
        """外部信号：中断任务（保留 react_trace 供后续恢复）。"""
        self._interrupted = True
        self.task.status = TaskStatus.INTERRUPTED
        logger.info("Task interrupted", task_id=self.task.task_id)

    async def run(self) -> CollaborativeTask:
        """Plan+ReAct 主循环：执行 → 观察 → 反思 → 重试 / 终止。"""
        # 注意：如果 interrupt() 在 run() 之前已调用，保留 INTERRUPTED 状态
        if self.task.status != TaskStatus.INTERRUPTED:
            self.task.status = TaskStatus.RUNNING
        self._append_trace("plan_generated", {
            "step_count": len(self.task.steps),
            "plan": [s.to_dict() for s in self.task.steps],
        })

        while self._iteration < self.max_iterations:
            if self._interrupted:
                return self.task

            # 1. 找到下一个可执行的步骤（依赖已满足 + 未完成）
            ready = self._ready_steps()
            if not ready:
                # 没有 ready 的步骤 → 检查终止条件
                if self._is_all_success():
                    self.task.status = TaskStatus.COMPLETED
                    return self.task
                # 还有 pending / failed 的步骤但不可执行 → 死锁
                if self._has_blocked_steps():
                    self.task.status = TaskStatus.FAILED
                    self._append_trace("termination", {"reason": "deadlock"})
                    return self.task
                # 没失败也没 pending → 全部完成
                self.task.status = TaskStatus.COMPLETED
                return self.task

            # 2. Act：并发执行所有 ready 的步骤
            self._append_trace("act", {"steps": [s.step_id for s in ready]})
            await asyncio.gather(
                *[self._execute_step(s) for s in ready],
                return_exceptions=False,
            )

            # 3. Observe + Reflect：观察结果并决定下一步
            self._append_trace("observe", {
                "iteration": self._iteration,
                "step_results": [
                    {"step_id": s.step_id, "status": s.status.value, "error": s.error}
                    for s in self.task.steps
                ],
            })

            self._iteration += 1
            self.task.updated_at = datetime.now(timezone.utc).isoformat()

        # 超过 max_iterations
        self.task.status = TaskStatus.FAILED
        self._append_trace("termination", {"reason": "max_iterations_reached"})
        return self.task

    # ----- 内部方法 -----

    def _ready_steps(self) -> list[TaskStep]:
        """返回当前可执行的步骤（依赖已满足 + 未完成 + 未超重试 + 非永久失败）。"""
        ready = []
        completed_ids = {
            s.step_id for s in self.task.steps
            if s.status == TaskStepStatus.SUCCESS
        }
        for step in self.task.steps:
            if step.status in (
                TaskStepStatus.SUCCESS,
                TaskStepStatus.SKIPPED,
                TaskStepStatus.RUNNING,
            ):
                continue
            if step.permanent_failure:
                continue  # 永久失败：不重试
            if step.status == TaskStepStatus.FAILED and step.retries >= step.max_retries:
                continue
            if all(dep in completed_ids for dep in step.depends_on):
                ready.append(step)
        return ready

    async def _execute_step(self, step: TaskStep) -> None:
        """执行单个步骤并写反思。"""
        step.status = TaskStepStatus.RUNNING
        try:
            result = await self.step_executor(step)
            step.result = result
            step.status = TaskStepStatus.SUCCESS
            step.reflection = self._reflect_on_result(step, result)
        except Exception as exc:
            step.error = str(exc)[:500]
            is_permanent = any(
                kw in step.error.lower() for kw in self.PERMANENT_ERROR_KEYWORDS
            )
            if is_permanent:
                # 永久错误：标 failed + permanent_failure=True（不再重试）
                step.retries += 1
                step.status = TaskStepStatus.FAILED
                step.permanent_failure = True
                step.reflection = "permanent_error_skip_remaining"
                logger.warning(
                    "Permanent error in step, no retry",
                    step_id=step.step_id, error=step.error,
                )
            else:
                step.retries += 1
                if step.retries >= step.max_retries:
                    step.status = TaskStepStatus.FAILED
                    step.reflection = f"retry_exhausted_{step.retries}"
                else:
                    step.status = TaskStepStatus.PENDING
                    step.reflection = f"will_retry_{step.retries + 1}"

            logger.warning(
                "Step failed",
                step_id=step.step_id,
                retries=step.retries,
                permanent=is_permanent,
                reflection=step.reflection,
                error=step.error[:200],
            )

    def _reflect_on_result(self, step: TaskStep, result: dict[str, Any]) -> str:
        """基于执行结果做反思（v1 简单规则版，可替换为 LLM）。"""
        if not result:
            return "empty_result"
        if result.get("status") == "success":
            return "ok"
        if result.get("confidence", 1.0) < 0.5:
            return "low_confidence_check_input_quality"
        return "ok"

    def _is_all_success(self) -> bool:
        return all(
            s.status in (TaskStepStatus.SUCCESS, TaskStepStatus.SKIPPED)
            for s in self.task.steps
        )

    def _has_blocked_steps(self) -> bool:
        """是否有 pending / failed 但依赖未满足的步骤（=死锁）。"""
        completed = {
            s.step_id for s in self.task.steps
            if s.status == TaskStepStatus.SUCCESS
        }
        for step in self.task.steps:
            if step.status in (
                TaskStepStatus.PENDING, TaskStepStatus.FAILED,
            ) and not all(d in completed for d in step.depends_on):
                return True
        return False

    def _append_trace(self, thought: str, observation: dict[str, Any]) -> None:
        self.task.react_trace.append({
            "iteration": self._iteration,
            "thought": thought,
            "observation": observation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # 限制 trace 长度防止内存爆
        if len(self.task.react_trace) > 200:
            self.task.react_trace = self.task.react_trace[-200:]
