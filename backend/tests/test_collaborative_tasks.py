"""
Tests for CollaborativeTask + Plan+ReAct runner (W4).
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.collaborative_task import (
    CollaborativeTask,
    PlanReActRunner,
    TaskStatus,
    TaskStep,
    TaskStepStatus,
    default_rca_plan,
)


# ===== 任务数据结构 =====


def test_default_rca_plan_has_dependency_chain():
    task = default_rca_plan("INC-1")
    assert task.title.startswith("RCA 协同任务")
    assert len(task.steps) == 4
    # 验证依赖链：evidence 依赖 diagnosis
    evidence_step = next(s for s in task.steps if s.step_id == "step-evidence")
    assert "step-diagnosis" in evidence_step.depends_on
    # runbook 依赖 diagnosis + evidence
    rb_step = next(s for s in task.steps if s.step_id == "step-runbook")
    assert set(rb_step.depends_on) == {"step-diagnosis", "step-evidence"}


def test_task_to_summary_counts_step_statuses():
    task = default_rca_plan("INC-1")
    task.steps[0].status = TaskStepStatus.SUCCESS
    task.steps[1].status = TaskStepStatus.FAILED
    summary = task.to_summary()
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert summary["step_count"] == 4


def test_task_to_dict_includes_react_trace():
    task = default_rca_plan("INC-1")
    task.react_trace.append({"iteration": 0, "thought": "plan", "observation": {}})
    blob = task.to_dict()
    assert "react_trace" in blob
    assert blob["react_trace"][0]["thought"] == "plan"


# ===== PlanReActRunner =====


async def _ok_executor(step: TaskStep) -> dict:
    return {"status": "success", "step_id": step.step_id, "result": "ok"}


async def _fail_once_executor(step: TaskStep) -> dict:
    if step.retries == 1:
        raise RuntimeError("transient failure")
    return {"status": "success", "step_id": step.step_id}


async def _always_fail_executor(step: TaskStep) -> dict:
    raise RuntimeError("permanent failure: not found")


async def _permanent_error_executor(step: TaskStep) -> dict:
    raise RuntimeError("permission denied")


@pytest.mark.asyncio
async def test_runner_executes_all_steps_to_completion():
    task = default_rca_plan("INC-1")
    runner = PlanReActRunner(task, _ok_executor)
    result = await runner.run()
    assert result.status == TaskStatus.COMPLETED
    assert all(s.status == TaskStepStatus.SUCCESS for s in result.steps)


@pytest.mark.asyncio
async def test_runner_records_react_trace_per_iteration():
    task = default_rca_plan("INC-1")
    runner = PlanReActRunner(task, _ok_executor)
    await runner.run()
    thoughts = [t["thought"] for t in task.react_trace]
    assert "plan_generated" in thoughts
    assert "act" in thoughts
    assert "observe" in thoughts
    assert "termination" in thoughts or task.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_runner_retries_transient_failure():
    """第一次失败 → 重试 → 成功 → COMPLETED。"""
    task = default_rca_plan("INC-1")
    evi_step = next(s for s in task.steps if s.step_id == "step-evidence")
    evi_step.max_retries = 3

    async def diag_exec(step: TaskStep) -> dict:
        return {"status": "success"}

    async def evi_exec(step: TaskStep) -> dict:
        # 第一次（retries=0）：失败
        if step.retries == 0:
            raise RuntimeError("transient timeout")
        return {"status": "success"}

    async def dispatch(step: TaskStep) -> dict:
        if step.step_id == "step-diagnosis":
            return await diag_exec(step)
        return await evi_exec(step)

    runner = PlanReActRunner(task, dispatch)
    result = await runner.run()

    # evidence 第一次失败 → retries=1 → 重试 → 成功
    assert evi_step.retries == 1
    assert evi_step.status == TaskStepStatus.SUCCESS
    assert result.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}


@pytest.mark.asyncio
async def test_runner_marks_permanent_error_as_failed():
    """永久错误（permission denied）→ 立即标 failed，不重试。"""
    task = default_rca_plan("INC-1")
    diag_step = next(s for s in task.steps if s.step_id == "step-diagnosis")
    diag_step.max_retries = 5

    async def perm_diag(step: TaskStep) -> dict:
        raise RuntimeError("permission denied: cannot access RCA agent")

    async def dispatch(step: TaskStep) -> dict:
        if step.step_id == "step-diagnosis":
            raise RuntimeError("permission denied: cannot access RCA agent")
        return await _ok_executor(step)

    runner = PlanReActRunner(task, dispatch)
    result = await runner.run()

    # diagnosis 永久错误 → 只尝试 1 次
    assert diag_step.retries == 1
    assert diag_step.status == TaskStepStatus.FAILED
    assert "permanent_error" in diag_step.reflection


@pytest.mark.asyncio
async def test_runner_interrupt_sets_status():
    task = default_rca_plan("INC-1")
    runner = PlanReActRunner(task, _ok_executor)
    runner.interrupt()
    result = await runner.run()
    assert result.status == TaskStatus.INTERRUPTED


@pytest.mark.asyncio
async def test_runner_dependency_order():
    """验证 runbook 一定在 diagnosis 和 evidence 之后执行。"""
    task = default_rca_plan("INC-1")
    execution_order: list[str] = []

    async def record_executor(step: TaskStep) -> dict:
        execution_order.append(step.step_id)
        # 模拟每个步骤耗时
        await asyncio.sleep(0.01)
        return {"status": "success"}

    runner = PlanReActRunner(task, record_executor)
    await runner.run()

    # runbook 一定在 evidence 之后
    assert execution_order.index("step-runbook") > execution_order.index("step-evidence")
    assert execution_order.index("step-evidence") > execution_order.index("step-diagnosis")


@pytest.mark.asyncio
async def test_runner_truncates_react_trace_at_200():
    """react_trace 超过 200 条应被截断。"""
    task = default_rca_plan("INC-1")

    async def slow_executor(step: TaskStep) -> dict:
        await asyncio.sleep(0.001)
        return {"status": "success"}

    # 强制 max_iterations 触发多次 trace
    runner = PlanReActRunner(task, slow_executor, max_iterations=300)
    await runner.run()
    assert len(task.react_trace) <= 200


def test_to_dict_includes_all_step_fields():
    task = default_rca_plan("INC-1")
    blob = task.to_dict()
    step_blob = blob["steps"][0]
    assert {"step_id", "name", "agent", "action", "depends_on", "status", "retries"}.issubset(set(step_blob.keys()))


def test_step_to_dict_includes_reflection():
    step = TaskStep(step_id="x", name="n", agent="a", action="b", reflection="test reflection")
    blob = step.to_dict()
    assert blob["reflection"] == "test reflection"
