"""Tests for W8.3 DecisionContext.

验证跨 stage 共享、refinement_history 审计、序列化兼容性。
"""

from __future__ import annotations

import pytest

from app.services.decision_context import (
    DecisionContext,
    RefinementEntry,
    StageAdapter,
    create_stage_adapter,
    is_w8_3_context_enabled,
)


# ===== Feature Flag =====


def test_feature_flag_default_disabled(monkeypatch):
    """未配置 AIOPS_USE_W8_3_CONTEXT 时默认关闭。"""
    monkeypatch.delenv("AIOPS_USE_W8_3_CONTEXT", raising=False)
    assert is_w8_3_context_enabled() is False


def test_feature_flag_enable(monkeypatch):
    """'true' 启用。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    assert is_w8_3_context_enabled() is True


def test_feature_flag_disabled(monkeypatch):
    """'false' 关闭。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "false")
    assert is_w8_3_context_enabled() is False


# ===== DecisionContext 单元测试 =====


def test_decision_context_default_empty():
    """默认 DecisionContext 4 个分类都是空 dict。"""
    dc = DecisionContext()
    assert dc.feature_flags == {}
    assert dc.resource_limits == {}
    assert dc.version_info == {}
    assert dc.cross_stage_decisions == {}
    assert dc.refinement_history == []


def test_set_feature_flag():
    """set feature_flags 字段。"""
    dc = DecisionContext()
    dc.set("feature_flags", "use_log", True, stage="rca", reason="log rich")
    assert dc.feature_flags["use_log"] is True
    assert len(dc.refinement_history) == 1
    assert dc.refinement_history[0]["stage"] == "rca"
    assert dc.refinement_history[0]["reason"] == "log rich"


def test_set_resource_limit():
    """set resource_limits 字段。"""
    dc = DecisionContext()
    dc.set("resource_limits", "max_api_calls", 100.0, stage="heal")
    assert dc.resource_limits["max_api_calls"] == 100.0


def test_set_version_info():
    """set version_info 字段。"""
    dc = DecisionContext()
    dc.set("version_info", "rca_agent", "v2.5", stage="rca")
    assert dc.version_info["rca_agent"] == "v2.5"


def test_set_cross_stage_decision():
    """set cross_stage_decisions（A → B 决策）。"""
    dc = DecisionContext()
    dc.set("cross_stage_decisions", "heal_hint", "database_issue", stage="rca")
    assert dc.cross_stage_decisions["heal_hint"] == "database_issue"


def test_set_invalid_category_raises():
    """set 无效 category 抛 ValueError。"""
    dc = DecisionContext()
    with pytest.raises(ValueError, match="unknown category"):
        dc.set("invalid_category", "key", "value", stage="rca")


def test_get_feature_flag():
    """get feature_flag 字段。"""
    dc = DecisionContext()
    dc.set("feature_flags", "use_log", True, stage="rca")
    assert dc.get("feature_flags", "use_log") is True
    assert dc.get("feature_flags", "missing") is None
    assert dc.get("feature_flags", "missing", default=False) is False


def test_get_refinements_filtered_by_key():
    """get_refinements 按 key 过滤。"""
    dc = DecisionContext()
    dc.set("feature_flags", "use_log", True, stage="rca")
    dc.set("resource_limits", "max_api", 100.0, stage="rca")
    dc.set("feature_flags", "use_log", False, stage="heal")  # 覆盖

    log_refinements = dc.get_refinements("use_log")
    assert len(log_refinements) == 2
    assert log_refinements[0]["value"] is True
    assert log_refinements[1]["value"] is False


def test_get_refinements_all():
    """get_refinements 不传 key 返回全部。"""
    dc = DecisionContext()
    dc.set("feature_flags", "a", True, stage="rca")
    dc.set("feature_flags", "b", False, stage="heal")
    assert len(dc.get_refinements()) == 2


def test_to_dict_serialization():
    """to_dict 序列化 4 个分类。"""
    dc = DecisionContext()
    dc.set("feature_flags", "use_log", True, stage="rca")
    dc.set("resource_limits", "max_api", 100.0, stage="heal")
    data = dc.to_dict()
    assert "feature_flags" in data
    assert "resource_limits" in data
    assert "version_info" in data
    assert "cross_stage_decisions" in data
    assert "refinement_history" in data


def test_from_dict_deserialization():
    """from_dict 还原 DecisionContext。"""
    data = {
        "feature_flags": {"a": True},
        "resource_limits": {"b": 1.0},
        "version_info": {"c": "v1"},
        "cross_stage_decisions": {"d": "x"},
        "refinement_history": [{"key": "a", "value": True, "stage": "rca", "timestamp": "2026-01-01"}],
    }
    dc = DecisionContext.from_dict(data)
    assert dc.feature_flags == {"a": True}
    assert dc.resource_limits == {"b": 1.0}
    assert dc.refinement_history[0]["key"] == "a"


def test_round_trip_serialization():
    """to_dict → from_dict round-trip 不丢数据。"""
    dc = DecisionContext()
    dc.set("feature_flags", "use_log", True, stage="rca", reason="log rich")
    dc.set("resource_limits", "max_api", 100.0, stage="heal")
    data = dc.to_dict()
    dc2 = DecisionContext.from_dict(data)
    assert dc2.feature_flags == dc.feature_flags
    assert dc2.resource_limits == dc.resource_limits
    assert len(dc2.refinement_history) == len(dc.refinement_history)


# ===== StageAdapter 测试 =====


def test_stage_adapter_disabled_no_persist(monkeypatch):
    """feature flag 关闭时，StageAdapter 写入不持久化。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "false")
    incident_ctx = {"incident_id": "INC-001"}
    adapter = create_stage_adapter(incident_ctx, "rca")
    adapter.set_feature_flag("use_log", True, reason="x")
    assert "decision_ctx" not in incident_ctx


def test_stage_adapter_enabled_persist(monkeypatch):
    """feature flag 启用时，StageAdapter 写入 incident.context。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {"incident_id": "INC-001"}
    adapter = create_stage_adapter(incident_ctx, "rca")
    adapter.set_feature_flag("use_log", True, reason="x")

    assert "decision_ctx" in incident_ctx
    assert incident_ctx["decision_ctx"]["feature_flags"]["use_log"] is True
    assert len(incident_ctx["decision_ctx"]["refinement_history"]) == 1
    assert incident_ctx["decision_ctx"]["refinement_history"][0]["reason"] == "x"


def test_stage_adapter_cross_stage_decision(monkeypatch):
    """RCA 写 cross_stage_decisions，Heal 读出来。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}

    # RCA 写 hint
    rca_adapter = create_stage_adapter(incident_ctx, "rca")
    rca_adapter.set_cross_stage("heal_hint", "database_issue", reason="rca said DB")

    # Heal 读 hint
    heal_adapter = create_stage_adapter(incident_ctx, "heal")
    assert heal_adapter.get_cross_stage("heal_hint") == "database_issue"


def test_stage_adapter_set_resource_limit(monkeypatch):
    """set_resource_limit 写 resource_limits。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}
    adapter = create_stage_adapter(incident_ctx, "heal")
    adapter.set_resource_limit("max_api_calls", 50.0, reason="rate limit")
    assert incident_ctx["decision_ctx"]["resource_limits"]["max_api_calls"] == 50.0


def test_stage_adapter_set_version(monkeypatch):
    """set_version 写 version_info。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}
    adapter = create_stage_adapter(incident_ctx, "rca")
    adapter.set_version("rca_agent", "v2.5", reason="upgrade")
    assert incident_ctx["decision_ctx"]["version_info"]["rca_agent"] == "v2.5"


def test_stage_adapter_recovers_existing_context(monkeypatch):
    """StageAdapter 能从 incident.context 还原已有 DecisionContext。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {
        "decision_ctx": {
            "feature_flags": {"existing": True},
            "resource_limits": {},
            "version_info": {},
            "cross_stage_decisions": {},
            "refinement_history": [],
        }
    }
    adapter = create_stage_adapter(incident_ctx, "rca")
    # 已有字段能被读到
    assert adapter.get_feature_flag("existing") is True
    # 新写入不会清空已有
    adapter.set_feature_flag("new_flag", True)
    assert adapter.get_feature_flag("existing") is True
    assert adapter.get_feature_flag("new_flag") is True


def test_stage_adapter_get_feature_flag_default(monkeypatch):
    """get_feature_flag 不存在时返回 default。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}
    adapter = create_stage_adapter(incident_ctx, "rca")
    assert adapter.get_feature_flag("nonexistent") is False
    assert adapter.get_feature_flag("nonexistent", default=True) is True


def test_stage_adapter_get_refinements(monkeypatch):
    """get_refinements 返回 refinement_history。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}
    adapter = create_stage_adapter(incident_ctx, "rca")
    adapter.set_feature_flag("a", True)
    adapter.set_feature_flag("b", False)

    refinements = adapter.get_refinements()
    assert len(refinements) == 2

    a_refinements = adapter.get_refinements("a")
    assert len(a_refinements) == 1
    assert a_refinements[0]["value"] is True


def test_stage_adapter_records_stage_in_history(monkeypatch):
    """refinement_history 记录 stage 标识。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}
    adapter = create_stage_adapter(incident_ctx, "rca")
    adapter.set_feature_flag("test", True)
    refinements = adapter.get_refinements("test")
    assert refinements[0]["stage"] == "rca"


# ===== 跨 stage 协同测试 =====


def test_cross_stage_full_flow(monkeypatch):
    """完整跨 stage 流程：Monitor → RCA → Heal → Change。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {}

    # 1. Monitor stage：设 feature_flag
    monitor = create_stage_adapter(incident_ctx, "monitor")
    monitor.set_feature_flag("enable_log_query", True, reason="severity high")

    # 2. RCA stage：读 Monitor 设的 flag + 写 cross_stage_decisions
    rca = create_stage_adapter(incident_ctx, "rca")
    assert rca.get_feature_flag("enable_log_query") is True
    rca.set_cross_stage("heal_hint", "database_issue", reason="5路证据共识")
    rca.set_version("rca_agent", "v2.5", reason="upgrade")

    # 3. Heal stage：读 RCA 写的 hint
    heal = create_stage_adapter(incident_ctx, "heal")
    assert heal.get_cross_stage("heal_hint") == "database_issue"
    assert heal.get_feature_flag("enable_log_query") is True  # 跨 stage 传递

    # 4. Change stage：读所有
    change = create_stage_adapter(incident_ctx, "change")
    assert change.get_cross_stage("heal_hint") == "database_issue"
    assert change.get_feature_flag("enable_log_query") is True

    # 最终 decision_ctx 字段存在
    assert "decision_ctx" in incident_ctx
    assert len(incident_ctx["decision_ctx"]["refinement_history"]) == 3


def test_cross_stage_no_interference_with_other_fields(monkeypatch):
    """DecisionContext 不污染 incident.context 的其他字段。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "true")
    incident_ctx = {
        "incident_id": "INC-001",
        "service": "order-service",
        "audit_trail": [{"step": "triage"}],
    }
    adapter = create_stage_adapter(incident_ctx, "rca")
    adapter.set_feature_flag("test", True)

    # 其他字段没被修改
    assert incident_ctx["incident_id"] == "INC-001"
    assert incident_ctx["service"] == "order-service"
    assert incident_ctx["audit_trail"] == [{"step": "triage"}]
    # 新字段仅追加
    assert "decision_ctx" in incident_ctx


def test_decision_context_disabled_keeps_feature_flag_false(monkeypatch):
    """feature flag 关闭时，get_feature_flag 永远返回 default。"""
    monkeypatch.setenv("AIOPS_USE_W8_3_CONTEXT", "false")
    incident_ctx = {}
    adapter = create_stage_adapter(incident_ctx, "rca")
    # 即使手动设了，关闭时不写
    adapter.set_feature_flag("test", True)
    # 读时按 default
    assert adapter.get_feature_flag("test", default=False) is False
    # 关闭时也没写
    assert "decision_ctx" not in incident_ctx


# ===== 工厂函数 =====


def test_factory_returns_stage_adapter(monkeypatch):
    """工厂函数返回 StageAdapter 实例。"""
    adapter = create_stage_adapter({}, "rca")
    assert isinstance(adapter, StageAdapter)
