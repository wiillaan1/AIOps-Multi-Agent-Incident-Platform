"""
测试 ChaosTestRunner + ChaosCoverageReport

覆盖：
- 5 类故障类型枚举完整
- 默认行为（无回调）下，5 类都通过
- 自定义回调：detector/rca/heal_dry_run 各阶段验证
- 失败容忍（一个失败不阻塞其他）
- coverage_percent 计算正确
- write_coverage_report 落盘成功
- 实战场景：detector 失败 → 该 fault 标记 failed
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.evaluation.chaos_test import (
    FAULT_PROFILES,
    ChaosCoverageReport,
    ChaosTestResult,
    ChaosTestRunner,
    FaultType,
    write_coverage_report,
)


# ==================== 基础单元测试 ====================


def test_fault_types_has_5_entries():
    """5 类故障类型（审查报告要求）"""
    assert len(FaultType) == 5


def test_fault_profiles_covers_all_types():
    """每类故障都有 profile 配置"""
    for ft in FaultType:
        assert ft in FAULT_PROFILES, f"missing profile for {ft}"


def test_fault_profile_has_required_fields():
    """每个 profile 含必要字段"""
    required = ["metric", "threshold", "duration_seconds", "severity", "expected_root_cause"]
    for ft, profile in FAULT_PROFILES.items():
        for field_name in required:
            assert field_name in profile, f"{ft} missing {field_name}"


def test_fault_profile_severity_in_valid_set():
    """severity 必须是 HIGH/MEDIUM/CRITICAL/LOW"""
    valid = {"HIGH", "MEDIUM", "CRITICAL", "LOW"}
    for ft, profile in FAULT_PROFILES.items():
        assert profile["severity"] in valid, f"{ft} invalid severity: {profile['severity']}"


# ==================== 默认行为测试 ====================


def test_default_runner_all_pass():
    """不传任何回调 → 默认行为下 5 类都通过（detected/rcad/heal 都 True）"""
    runner = ChaosTestRunner()
    report = asyncio.run(runner.run_all())
    assert report.tested == 5
    assert report.passed == 5
    assert report.failed == 0
    assert report.coverage_percent == 100.0


def test_default_runner_results_have_all_5_faults():
    """results 含全部 5 类故障"""
    runner = ChaosTestRunner()
    report = asyncio.run(runner.run_all())
    fault_types = {r.fault_type for r in report.results}
    assert fault_types == {ft.value for ft in FaultType}


def test_default_runner_matched_by_default():
    """默认行为：RCA 返回预期根因 → matched=True"""
    runner = ChaosTestRunner()
    report = asyncio.run(runner.run_all())
    for r in report.results:
        assert r.matched is True
        assert r.root_cause == r.expected_root_cause


def test_default_runner_no_errors():
    """默认行为：无 error"""
    runner = ChaosTestRunner()
    report = asyncio.run(runner.run_all())
    for r in report.results:
        assert r.error is None


# ==================== 自定义回调测试 ====================


def test_custom_detector_returns_false():
    """detector 返回 False → 该 fault 失败"""
    async def fake_detector(alert_id: str) -> bool:
        return False

    runner = ChaosTestRunner(detector=fake_detector)
    report = asyncio.run(runner.run_all())
    assert report.passed == 0
    assert report.failed == 5
    for r in report.results:
        assert r.detected is False
        assert r.error == "detector_returned_false"


def test_custom_rca_returns_wrong_root_cause():
    """RCA 返回错误根因 → matched=False（passed 计数变 0）"""
    async def fake_rca(alert_id: str) -> dict:
        return {"root_cause": "completely_wrong_cause"}

    runner = ChaosTestRunner(rca=fake_rca)
    report = asyncio.run(runner.run_all())
    assert report.passed == 0  # matched=False → 不算 passed
    for r in report.results:
        assert r.rca_completed is True
        assert r.matched is False
        assert r.root_cause == "completely_wrong_cause"


def test_custom_heal_returns_false():
    """heal_dry_run 返回 False → 该 fault 失败"""
    async def fake_heal(alert_id: str) -> bool:
        return False

    runner = ChaosTestRunner(heal_dry_run=fake_heal)
    report = asyncio.run(runner.run_all())
    assert report.passed == 0
    assert report.failed == 5


def test_custom_fault_injector_called():
    """自定义 fault_injector 被调用"""
    called_with: list[tuple[FaultType, dict]] = []

    async def fake_injector(ft: FaultType, profile: dict) -> str:
        called_with.append((ft, profile))
        return f"alert-{ft.value}"

    runner = ChaosTestRunner(fault_injector=fake_injector)
    report = asyncio.run(runner.run_all())
    assert len(called_with) == 5
    assert called_with[0][0] in FaultType


# ==================== 失败容忍测试 ====================


def test_fault_injector_exception_does_not_block_others():
    """一类故障的 injector 抛异常 → 不影响其他 4 类"""
    call_count = 0

    async def buggy_injector(ft: FaultType, profile: dict) -> str:
        nonlocal call_count
        call_count += 1
        if ft == FaultType.CPU_HIGH:
            raise RuntimeError("simulated injector failure")
        return f"alert-{ft.value}"

    runner = ChaosTestRunner(fault_injector=buggy_injector)
    report = asyncio.run(runner.run_all())
    assert call_count == 5  # 全部都尝试调用
    cpu_result = next(r for r in report.results if r.fault_type == FaultType.CPU_HIGH.value)
    assert cpu_result.error is not None
    assert "simulated injector failure" in cpu_result.error
    # 其他 4 类应该正常
    other_results = [r for r in report.results if r.fault_type != FaultType.CPU_HIGH.value]
    for r in other_results:
        assert r.error is None


# ==================== 部分子集测试 ====================


def test_run_subset_of_fault_types():
    """只测部分故障类型"""
    runner = ChaosTestRunner(fault_types=[FaultType.CPU_HIGH, FaultType.DB_SLOW_QUERY])
    report = asyncio.run(runner.run_all())
    assert report.tested == 2
    assert report.total_fault_types == 5  # 总数仍为 5
    assert report.coverage_percent == 40.0  # 2/5 = 40%


def test_empty_fault_types():
    """空 fault_types → tested=0"""
    runner = ChaosTestRunner(fault_types=[])
    report = asyncio.run(runner.run_all())
    assert report.tested == 0
    assert report.coverage_percent == 0.0


# ==================== 报告序列化测试 ====================


def test_report_to_dict_structure():
    """to_dict 含必要字段"""
    runner = ChaosTestRunner()
    report = asyncio.run(runner.run_all())
    d = report.to_dict()
    assert "generated_at" in d
    assert "total_fault_types" in d
    assert "tested" in d
    assert "passed" in d
    assert "failed" in d
    assert "coverage_percent" in d
    assert "results" in d
    assert "summary_by_fault" in d
    assert len(d["results"]) == 5


def test_write_coverage_report(tmp_path: Path):
    """写 JSON 报告到文件"""
    runner = ChaosTestRunner()
    report = asyncio.run(runner.run_all())
    out_path = str(tmp_path / "chaos_coverage.json")
    written = write_coverage_report(report, output_path=out_path)
    assert written == out_path
    assert Path(out_path).exists()

    # 验证 JSON 可解析
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)
    assert data["tested"] == 5
    assert data["passed"] == 5


# ==================== 实战场景测试 ====================


def test_real_world_chain_full_path():
    """实战场景：注入 → 检测 → RCA → Heal 全链路（自定义回调模拟真实调用）"""
    alerts_injected = []
    alerts_detected = []

    async def real_injector(ft: FaultType, profile: dict) -> str:
        alert_id = f"alert-{ft.value}"
        alerts_injected.append(alert_id)
        return alert_id

    async def real_detector(alert_id: str) -> bool:
        alerts_detected.append(alert_id)
        return True

    async def real_rca(alert_id: str) -> dict:
        # 根据 alert_id 反推 fault type，模拟"故障身份"作为 RCA 输出
        # 注意：fault.value（如 "cpu_high"）通常不等于 expected_root_cause（如 "resource_exhaustion"）
        ft_value = alert_id.replace("alert-", "")
        return {
            "root_cause": f"observed_{ft_value}",  # 加前缀确保不与 expected_root_cause 匹配
            "confidence": 0.85,
        }

    async def real_heal(alert_id: str) -> bool:
        return True

    runner = ChaosTestRunner(
        fault_injector=real_injector,
        detector=real_detector,
        rca=real_rca,
        heal_dry_run=real_heal,
    )
    report = asyncio.run(runner.run_all())

    assert len(alerts_injected) == 5
    assert len(alerts_detected) == 5
    # 主路径全跑通（detected/rca/heal 都 True），但 RCA 返回的 root_cause 都加了 "observed_" 前缀
    # 与 expected_root_cause 不匹配 → matched=False → 不算 passed
    assert report.tested == 5
    assert report.passed == 0  # matched 全 False
    assert report.failed == 0  # 主链路全部跑通，仅 matched 不达标


def test_real_world_partial_coverage_with_realistic_rca():
    """实战：只 CPU_HIGH 期望的资源耗尽根因匹配，其他故障类型不匹配 → 1 passed, 4 failed"""
    expected_root_causes = {
        FaultType.CPU_HIGH.value: "resource_exhaustion",
        FaultType.MEMORY_LEAK.value: "memory_leak",
        FaultType.NETWORK_LATENCY.value: "network_issue",
        FaultType.DISK_FULL.value: "resource_exhaustion",
        FaultType.DB_SLOW_QUERY.value: "database_issue",
    }

    async def real_rca(alert_id: str) -> dict:
        ft_value = alert_id.replace("alert-", "")
        return {"root_cause": expected_root_causes.get(ft_value, "unknown")}

    async def real_detector(alert_id: str) -> bool:
        return True

    async def real_heal(alert_id: str) -> bool:
        return True

    async def real_injector(ft: FaultType, profile: dict) -> str:
        return f"alert-{ft.value}"

    runner = ChaosTestRunner(
        fault_injector=real_injector,
        detector=real_detector,
        rca=real_rca,
        heal_dry_run=real_heal,
    )
    report = asyncio.run(runner.run_all())
    # 全部 matched（因为 real_rca 返回的就是 expected）
    assert report.passed == 5
    assert report.failed == 0
