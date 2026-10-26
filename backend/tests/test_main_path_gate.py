"""Tests for W8.2 MainPathGate.

验证 4 问启发式 Gate 在主链路场景下的行为。
"""

from __future__ import annotations

import os

import pytest

from app.services.main_path_gate import (
    MainPathGate,
    MainPathGateResult,
    get_main_path_gate,
    is_w8_2_gate_enabled,
)


# ===== Feature Flag =====


def test_feature_flag_default_disabled(monkeypatch):
    """未配置 AIOPS_USE_W8_2_GATE 时默认关闭。"""
    monkeypatch.delenv("AIOPS_USE_W8_2_GATE", raising=False)
    assert is_w8_2_gate_enabled() is False


def test_feature_flag_enable_variants(monkeypatch):
    """'true'/'1'/'yes'/'on' 都能启用。"""
    for val in ["true", "1", "yes", "on", "TRUE"]:
        monkeypatch.setenv("AIOPS_USE_W8_2_GATE", val)
        assert is_w8_2_gate_enabled() is True, f"failed for {val}"


def test_feature_flag_disabled_variants(monkeypatch):
    """'false'/'0'/'no'/'off'/'garbage' 都是关闭。"""
    for val in ["false", "0", "no", "off", "garbage"]:
        monkeypatch.setenv("AIOPS_USE_W8_2_GATE", val)
        assert is_w8_2_gate_enabled() is False, f"failed for {val}"


# ===== 成功路径 =====


def _success_context() -> dict:
    """构造一个全通过的 happy path context。"""
    return {
        "incident_id": "INC-001",
        "service": "order-service",
        "rca_event": {
            "root_cause": "database_issue",
            "confidence": 0.85,
            "evidence": {
                "bayesian": {"posterior": 0.45},
                "bfs": {"chain": ["payment-service"]},
                "rag": {"match_score": 0.78},
                "history": {"similarity": 0.82},
                "log": {"samples": ["lock wait timeout"]},
            },
        },
        "audit_trail": [
            {"step": "triage", "trigger": "x", "outcome": "y", "timestamp": 1722000000.0},
            {"step": "rca_bayesian", "outcome": "ok", "timestamp": 1722000001.0},
            {"step": "heal_dry_run", "outcome": "ok", "timestamp": 1722000002.0},
            {"step": "verify_reflection", "outcome": "ok", "timestamp": 1722000003.0},
        ],
        "verification": {
            "rca_match_status": "match",
            "heal_recovered": True,
            "plan_b_triggered": False,
        },
    }


def test_gate_pass_on_happy_path(monkeypatch):
    """成功路径：所有 4 问通过。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    result = gate.check(_success_context())
    assert result.passed is True
    assert "all 4 checks passed" in result.reason
    assert len(result.failed_checks) == 0
    for check_str in result.checks.values():
        assert check_str.startswith("ok")


def test_gate_disabled_returns_pass(monkeypatch):
    """feature flag 关闭时直接返回 passed。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "false")
    gate = MainPathGate()
    result = gate.check({})
    assert result.passed is True
    assert "disabled" in result.reason


# ===== 4 问分别测试 =====


def test_check_time_ordering_fails_gap_too_large(monkeypatch):
    """时间顺序：audit_trail 时间间隔超出 1 小时。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["audit_trail"] = [
        {"step": "triage", "timestamp": 1722000000.0},
        {"step": "verify", "timestamp": 1722005000.0},  # +5000s
    ]
    result = gate.check(ctx)
    assert result.checks["time_ordering"].startswith("fail")
    assert "time gap" in result.checks["time_ordering"]


def test_check_time_ordering_pass_normal(monkeypatch):
    """时间顺序：正常间隔通过。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["audit_trail"] = [
        {"step": "triage", "timestamp": 1722000000.0},
        {"step": "verify", "timestamp": 1722000060.0},  # +60s
    ]
    result = gate.check(ctx)
    assert result.checks["time_ordering"].startswith("ok")


def test_check_time_ordering_empty(monkeypatch):
    """时间顺序：空 audit_trail 返回 ok 兜底。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["audit_trail"] = []
    result = gate.check(ctx)
    assert result.checks["time_ordering"].startswith("ok")
    assert "no audit_trail" in result.checks["time_ordering"]


def test_check_time_ordering_handles_invalid_timestamps(monkeypatch):
    """时间顺序：无效时间戳被忽略。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["audit_trail"] = [
        {"step": "triage", "timestamp": "not a number"},
        {"step": "verify", "timestamp": None},
    ]
    result = gate.check(ctx)
    assert result.checks["time_ordering"].startswith("ok")


def test_check_alternatives_plan_b_plus_mismatch_fails(monkeypatch):
    """替代解释：plan_b + mismatch 双重信号 → fail。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["verification"]["plan_b_triggered"] = True
    ctx["verification"]["rca_match_status"] = "mismatch"
    result = gate.check(ctx)
    assert result.checks["alternatives"].startswith("fail")
    assert "plan_b" in result.checks["alternatives"]


def test_check_alternatives_plan_b_only_warn(monkeypatch):
    """替代解释：仅 plan_b → ok。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["verification"]["plan_b_triggered"] = True
    result = gate.check(ctx)
    assert result.checks["alternatives"].startswith("ok")


def test_check_alternatives_mismatch_only_warn(monkeypatch):
    """替代解释：仅 mismatch → warn。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["verification"]["rca_match_status"] = "mismatch"
    result = gate.check(ctx)
    assert result.checks["alternatives"].startswith("warn")


def test_check_evidence_5_routes_pass(monkeypatch):
    """证据充分性：5 路证据全有 → ok。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    # 默认已经有 5 路证据
    result = gate.check(ctx)
    assert result.checks["evidence"].startswith("ok")
    assert "5 evidence routes" in result.checks["evidence"]


def test_check_evidence_min_2_routes_pass(monkeypatch):
    """证据充分性：仅 2 路证据 → ok。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"]["evidence"] = {
        "bayesian": {"posterior": 0.45},
        "rag": {"match_score": 0.78},
    }
    result = gate.check(ctx)
    assert result.checks["evidence"].startswith("ok")
    assert "2 evidence routes" in result.checks["evidence"]


def test_check_evidence_only_1_route_fails(monkeypatch):
    """证据充分性：仅 1 路证据 → fail。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"]["evidence"] = {
        "bayesian": {"posterior": 0.45},
    }
    result = gate.check(ctx)
    assert result.checks["evidence"].startswith("fail")
    assert "1 evidence routes" in result.checks["evidence"]


def test_check_evidence_no_evidence_fails(monkeypatch):
    """证据充分性：完全无证据 → fail。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"]["evidence"] = {}
    result = gate.check(ctx)
    assert result.checks["evidence"].startswith("fail")
    assert "0 evidence routes" in result.checks["evidence"]


def test_check_counterfactual_low_confidence_fails(monkeypatch):
    """反事实可行性：低 confidence → fail。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"]["confidence"] = 0.4
    result = gate.check(ctx)
    assert result.checks["counterfactual"].startswith("fail")
    assert "0.40" in result.checks["counterfactual"]


def test_check_counterfactual_not_recovered_fails(monkeypatch):
    """反事实可行性：heal 未恢复 → fail。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["verification"]["heal_recovered"] = False
    result = gate.check(ctx)
    assert result.checks["counterfactual"].startswith("fail")
    assert "not recovered" in result.checks["counterfactual"]


def test_check_counterfactual_pass(monkeypatch):
    """反事实可行性：高 confidence + 已恢复 → ok。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    result = gate.check(ctx)
    assert result.checks["counterfactual"].startswith("ok")
    assert "confidence=0.85" in result.checks["counterfactual"]


def test_check_counterfactual_uses_rca_result_fallback(monkeypatch):
    """反事实可行性：rca_event 缺失时从 rca_result 取 confidence。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"] = "not a dict"  # 故意给字符串
    ctx["rca_result"] = {"confidence": 0.80}
    result = gate.check(ctx)
    assert result.checks["counterfactual"].startswith("ok")


# ===== 失败聚合 =====


def test_failed_checks_aggregated(monkeypatch):
    """失败聚合：4 问都失败时，failed_checks 列出全部。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = {
        "incident_id": "INC-FAIL",
        "rca_event": {
            "root_cause": "unknown",
            "confidence": 0.3,
            "evidence": {},
        },
        "audit_trail": [
            {"step": "triage", "timestamp": 1722000000.0},
            {"step": "verify", "timestamp": 1722005000.0},  # +5000s gap
        ],
        "verification": {
            "rca_match_status": "mismatch",
            "heal_recovered": False,
            "plan_b_triggered": True,
        },
    }
    result = gate.check(ctx)
    assert result.passed is False
    assert "failed checks:" in result.reason
    # 4 问应当都失败
    assert len(result.failed_checks) == 4
    assert "time_ordering" in result.failed_checks


def test_severity_warn_on_fail(monkeypatch):
    """失败时 severity=warn（不强制 retry）。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"]["confidence"] = 0.3  # 触发失败
    result = gate.check(ctx)
    assert result.passed is False
    assert result.severity == "warn"


def test_severity_info_on_disabled(monkeypatch):
    """feature flag 关闭时 severity=info。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "false")
    gate = MainPathGate()
    result = gate.check({})
    assert result.severity == "info"


# ===== 异常处理 =====


def test_gate_returns_pass_on_crash(monkeypatch):
    """Gate 内部异常时返回 passed=True（不影响主链路）。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()

    # 强制让 _check_evidence 抛 AttributeError
    # 通过 mock context 让 evidence 不是 dict（触发 None.items() 之类的）
    # 实际上 rca_event 不为 None 时我们用 isinstance(rca_event, dict) 保护
    # 所以特意构造一个会让某个 check 抛 AttributeError 的场景
    class _ExplodingDict(dict):
        """访问任何 key 时抛 AttributeError 的 dict。"""

        def get(self, key, default=None):
            raise AttributeError("simulated crash in evidence")

    ctx = {"rca_event": _ExplodingDict(), "audit_trail": []}
    result = gate.check(ctx)
    # 异常时不影响主链路
    assert result.passed is True
    assert "crashed" in result.reason


def test_gate_handles_empty_context(monkeypatch):
    """完全空的 context 不报错。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    result = gate.check({})
    # 4 问都"ok"（空数据兜底）
    assert isinstance(result, MainPathGateResult)


def test_gate_ignores_unknown_evidence_keys(monkeypatch):
    """evidence 字段出现未知 key 时不影响判断。"""
    monkeypatch.setenv("AIOPS_USE_W8_2_GATE", "true")
    gate = MainPathGate()
    ctx = _success_context()
    ctx["rca_event"]["evidence"] = {
        "bayesian": {"x": 1},
        "unknown_route": {"y": 2},
        "another_unknown": {"z": 3},
    }
    result = gate.check(ctx)
    # 只有 1 路证据（bayesian 算），应当 fail
    assert result.checks["evidence"].startswith("fail")


# ===== 工厂函数 =====


def test_factory_returns_instance():
    """工厂函数返回 MainPathGate 实例。"""
    gate = get_main_path_gate()
    assert isinstance(gate, MainPathGate)


# ===== 配置参数 =====


def test_min_evidence_routes_override(monkeypatch):
    """MIN_EVIDENCE_ROUTES 配置可被读取。"""
    # 当前默认 2
    assert MainPathGate.MIN_EVIDENCE_ROUTES == 2


def test_min_rca_confidence_override(monkeypatch):
    """MIN_RCA_CONFIDENCE 配置可被读取。"""
    assert MainPathGate.MIN_RCA_CONFIDENCE == 0.6
