"""Tests for W8.4 IncidentCorrelator.

验证跨 incident 关联分析的 4 维度相似度计算和时间窗口过滤。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.incident_correlator import (
    CorrelationResult,
    IncidentCorrelator,
    get_incident_correlator,
    is_w8_4_correlate_enabled,
)


# ===== Feature Flag =====


def test_feature_flag_default_disabled(monkeypatch):
    """未配置 AIOPS_USE_W8_4_CORRELATE 时默认关闭。"""
    monkeypatch.delenv("AIOPS_USE_W8_4_CORRELATE", raising=False)
    assert is_w8_4_correlate_enabled() is False


def test_feature_flag_enable(monkeypatch):
    """'true' 启用。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    assert is_w8_4_correlate_enabled() is True


def test_feature_flag_disabled(monkeypatch):
    """'false' 关闭。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "false")
    assert is_w8_4_correlate_enabled() is False


# ===== Test Data 工厂 =====


def _snapshot_v1(
    services: list[str] | None = None,
    root_cause: str | None = None,
    steps: list[str] | None = None,
    evidence_kinds: list[str] | None = None,
    incident_id: str = "INC-X",
) -> dict:
    """构造一个 W8.1 entity_snapshot。"""
    return {
        "schema_version": "1.0",
        "incident_id": incident_id,
        "services": services or [],
        "root_cause": root_cause,
        "confidence": 0.8,
        "steps": steps or [],
        "evidence_blocks": [
            {
                "block_id": f"eb-{i}",
                "source_kind": kind,
                "mechanism": kind,
            }
            for i, kind in enumerate(evidence_kinds or [])
        ],
        "created_at": "2026-07-25T10:00:00+00:00",
    }


def _incident(
    incident_id: str,
    snapshot: dict | None = None,
    created_at: str = "2026-07-25T10:00:00+00:00",
) -> dict:
    """构造一个带 entity_snapshot 的 incident 字典。"""
    return {
        "incident_id": incident_id,
        "created_at": created_at,
        "entity_snapshot": snapshot,
    }


# ===== correlate() 主流程测试 =====


def test_correlate_disabled_returns_empty(monkeypatch):
    """feature flag 关闭时返回空列表。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "false")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    candidate = _incident("INC-002", _snapshot_v1(services=["order-service"]))
    results = correlator.correlate(target, [candidate])
    assert results == []


def test_correlate_excludes_self(monkeypatch):
    """correlate 排除 target 自身。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    snap = _snapshot_v1(services=["order-service"], root_cause="db")
    target = _incident("INC-001", snap)
    results = correlator.correlate(target, [target])  # 把 target 自己也放进去
    assert results == []


def test_correlate_no_snapshot_returns_empty(monkeypatch):
    """target 没有 entity_snapshot 时返回空。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = {"incident_id": "INC-001", "created_at": "2026-07-25T10:00:00Z"}  # 无 snapshot
    candidate = _incident("INC-002", _snapshot_v1(services=["order-service"]))
    results = correlator.correlate(target, [candidate])
    assert results == []


def test_correlate_no_time_returns_empty(monkeypatch):
    """target 没有时间戳时返回空。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = {"incident_id": "INC-001", "entity_snapshot": _snapshot_v1()}  # 无 created_at
    candidate = _incident("INC-002", _snapshot_v1(services=["order-service"]))
    results = correlator.correlate(target, [candidate])
    assert results == []


# ===== 时间窗口过滤 =====


def test_correlate_filters_by_time_window(monkeypatch):
    """超出时间窗口的 candidate 被过滤。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(services=["order-service"]),
        created_at="2026-07-25T10:00:00Z",
    )
    # 2 小时前的 candidate（默认 60 分钟窗口外）
    candidate = _incident(
        "INC-002",
        _snapshot_v1(services=["order-service"]),
        created_at="2026-07-25T08:00:00Z",  # 2 小时前
    )
    results = correlator.correlate(target, [candidate])
    assert results == []


def test_correlate_includes_within_window(monkeypatch):
    """时间窗口内的 candidate 被保留。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
            steps=["triage", "rca"],
        ),
        created_at="2026-07-25T10:00:00Z",
    )
    # 30 分钟内的 candidate
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
            steps=["triage", "rca"],
        ),
        created_at="2026-07-25T10:30:00Z",
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1


def test_correlate_custom_window(monkeypatch):
    """自定义时间窗口生效。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
        created_at="2026-07-25T10:00:00Z",
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
        created_at="2026-07-25T11:30:00Z",  # 1.5 小时后
    )
    # 默认 60 分钟窗口外 → 排除
    results = correlator.correlate(target, [candidate], window_minutes=60)
    assert results == []
    # 改成 120 分钟窗口 → 返回
    results = correlator.correlate(target, [candidate], window_minutes=120)
    assert len(results) == 1


# ===== 4 维度相似度 =====


def test_similarity_service_match_only(monkeypatch):
    """仅 service 匹配，similarity=0.4 — 低于阈值 0.5，应被过滤。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    candidate = _incident("INC-002", _snapshot_v1(services=["order-service"]))
    results = correlator.correlate(target, [candidate])
    # 0.4 < 0.5 (阈值) → 排除
    assert results == []


def test_similarity_service_and_root_cause(monkeypatch):
    """service + root_cause 匹配，similarity=0.7。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(services=["order-service"], root_cause="db"),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(services=["order-service"], root_cause="db"),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    assert abs(results[0].similarity - 0.7) < 0.01


def test_similarity_full_match(monkeypatch):
    """4 维度全匹配。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    steps = ["triage", "rca_bayesian", "rca_rag", "heal_dry_run", "verify_reflection"]
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service", "payment-service"],
            root_cause="db",
            steps=steps,
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service", "payment-service"],
            root_cause="db",
            steps=steps,
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    # service(0.4) + rc(0.3) + evidence(0.2) + steps(0.1) = 1.0
    assert abs(results[0].similarity - 1.0) < 0.01


def test_similarity_below_threshold_excluded(monkeypatch):
    """相似度低于阈值 0.5 不返回。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    candidate = _incident("INC-002", _snapshot_v1(services=["payment-service"]))  # 不同 service
    results = correlator.correlate(target, [candidate])
    # service 不同 → similarity=0.0 → 低于 0.5 → 排除
    assert results == []


def test_similarity_different_service_no_correlation(monkeypatch):
    """不同 service 不关联。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    candidate = _incident("INC-002", _snapshot_v1(services=["inventory-service"]))
    results = correlator.correlate(target, [candidate])
    assert results == []


def test_similarity_different_root_cause(monkeypatch):
    """不同 root_cause 时，仅 service 匹配，similarity=0.4（< 0.5 阈值）。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"], root_cause="db"))
    candidate = _incident("INC-002", _snapshot_v1(services=["order-service"], root_cause="code_bug"))
    results = correlator.correlate(target, [candidate])
    # service 0.4 + rc 0.0 + evidence 0.0 + steps 0.0 = 0.4 < 0.5 → 排除
    assert results == []


def test_similarity_service_and_root_cause_at_threshold(monkeypatch):
    """service + root_cause 匹配，similarity=0.7（> 0.5 阈值）→ 返回。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(services=["order-service"], root_cause="db"),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(services=["order-service"], root_cause="db"),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    assert abs(results[0].similarity - 0.7) < 0.01


# ===== 排序与结果 =====


def test_results_sorted_by_similarity_desc(monkeypatch):
    """结果按 similarity 降序排序。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(services=["order-service"], root_cause="db"),
    )
    # 候选 1: 仅 service 相同 → 0.4（应被过滤）
    # 候选 2: service + rc → 0.7
    # 候选 3: service + rc + 3 evidence → 0.9
    c1 = _incident("INC-A", _snapshot_v1(services=["order-service"]))
    c2 = _incident(
        "INC-B",
        _snapshot_v1(services=["order-service"], root_cause="db"),
    )
    c3 = _incident(
        "INC-C",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    results = correlator.correlate(target, [c1, c2, c3])
    # 至少 2 个超过阈值
    assert len(results) >= 2
    # 排序降序
    for i in range(len(results) - 1):
        assert results[i].similarity >= results[i + 1].similarity


def test_correlation_result_common_services(monkeypatch):
    """CorrelationResult 包含 common_services。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service", "payment-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service", "inventory-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    assert "order-service" in results[0].common_services
    assert "payment-service" not in results[0].common_services
    assert "inventory-service" not in results[0].common_services


def test_correlation_result_shared_root_cause(monkeypatch):
    """CorrelationResult 包含 shared_root_cause。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="database_issue",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="database_issue",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    assert results[0].shared_root_cause == "database_issue"


def test_correlation_result_matched_evidence_kinds(monkeypatch):
    """CorrelationResult 包含 matched_evidence_kinds。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log", "history"],
        ),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    assert set(results[0].matched_evidence_kinds) >= {"bayesian", "rag", "log"}


def test_correlation_result_step_overlap(monkeypatch):
    """CorrelationResult 包含 step_overlap_count。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
            steps=["triage", "rca", "heal", "verify"],
        ),
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
            steps=["triage", "rca", "verify"],  # 重叠 3 个
        ),
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1
    assert results[0].step_overlap_count == 3


# ===== correlate_by_entity_snapshot 直接接口 =====


def test_correlate_by_snapshot_basic(monkeypatch):
    """直接对 snapshot 列表算关联。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _snapshot_v1(
        services=["order-service"],
        root_cause="db",
        evidence_kinds=["bayesian", "rag", "log"],
    )
    candidates = [
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
            incident_id="INC-A",
        ),
    ]
    results = correlator.correlate_by_entity_snapshot(target, candidates)
    assert len(results) == 1
    assert results[0].incident_id == "INC-A"


def test_correlate_by_snapshot_skips_invalid_version(monkeypatch):
    """schema_version 不匹配的 snapshot 被跳过。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _snapshot_v1(
        services=["order-service"],
        root_cause="db",
        evidence_kinds=["bayesian", "rag", "log"],
    )
    candidates = [
        {"schema_version": "0.9", "incident_id": "INC-OLD"},  # 版本不匹配
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
            incident_id="INC-A",
        ),
    ]
    results = correlator.correlate_by_entity_snapshot(target, candidates)
    # 0.9 版本被跳过，只返回 INC-A
    assert len(results) == 1
    assert results[0].incident_id == "INC-A"


def test_correlate_by_snapshot_disabled(monkeypatch):
    """feature flag 关闭时返回空。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "false")
    correlator = IncidentCorrelator()
    target = _snapshot_v1(services=["order-service"])
    candidates = [_snapshot_v1(services=["order-service"])]
    results = correlator.correlate_by_entity_snapshot(target, candidates)
    assert results == []


# ===== 时间戳解析 =====


def test_parse_unix_timestamp(monkeypatch):
    """Unix 时间戳解析。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(services=["order-service"]),
        created_at="2026-07-25T10:00:00Z",
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(services=["order-service"]),
        created_at=1753437600,  # 2025-07-25 时间戳
    )
    # 时间差距很大（> 1 年），应当被过滤
    results = correlator.correlate(target, [candidate])
    assert results == []


def test_parse_iso_with_z(monkeypatch):
    """ISO 8601 with Z suffix 解析。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
        created_at="2026-07-25T10:00:00Z",
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
        created_at="2026-07-25T10:30:00Z",
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1


def test_parse_iso_with_offset(monkeypatch):
    """ISO 8601 with +00:00 offset 解析。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
        created_at="2026-07-25T10:00:00+00:00",
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(
            services=["order-service"],
            root_cause="db",
            evidence_kinds=["bayesian", "rag", "log"],
        ),
        created_at="2026-07-25T10:30:00+00:00",
    )
    results = correlator.correlate(target, [candidate])
    assert len(results) == 1


def test_invalid_time_string_handled(monkeypatch):
    """无效时间字符串被忽略（cand_time=None → 跳过）。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident(
        "INC-001",
        _snapshot_v1(services=["order-service"]),
        created_at="2026-07-25T10:00:00Z",
    )
    candidate = _incident(
        "INC-002",
        _snapshot_v1(services=["order-service"]),
        created_at="not a date",
    )
    results = correlator.correlate(target, [candidate])
    assert results == []


# ===== 异常处理 =====


def test_correlate_handles_malformed_snapshot(monkeypatch):
    """malformed snapshot 不导致异常。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    # candidate 的 snapshot 是字符串而非 dict
    candidate = {
        "incident_id": "INC-002",
        "created_at": "2026-07-25T10:00:00Z",
        "entity_snapshot": "not a dict",
    }
    results = correlator.correlate(target, [candidate])
    assert results == []  # 优雅跳过


def test_correlate_empty_candidates(monkeypatch):
    """空 candidates 列表返回空。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    results = correlator.correlate(target, [])
    assert results == []


def test_correlate_candidate_missing_snapshot(monkeypatch):
    """candidate 没有 entity_snapshot 时跳过。"""
    monkeypatch.setenv("AIOPS_USE_W8_4_CORRELATE", "true")
    correlator = IncidentCorrelator()
    target = _incident("INC-001", _snapshot_v1(services=["order-service"]))
    candidate = {"incident_id": "INC-002", "created_at": "2026-07-25T10:00:00Z"}  # 无 snapshot
    results = correlator.correlate(target, [candidate])
    assert results == []


# ===== 工厂函数 =====


def test_factory_returns_correlator():
    """工厂函数返回 IncidentCorrelator 实例。"""
    correlator = get_incident_correlator()
    assert isinstance(correlator, IncidentCorrelator)


# ===== 配置参数 =====


def test_similarity_threshold_default():
    """SIMILARITY_THRESHOLD 默认 0.5。"""
    assert IncidentCorrelator.SIMILARITY_THRESHOLD == 0.5


def test_default_window_minutes():
    """DEFAULT_WINDOW_MINUTES 默认 60。"""
    assert IncidentCorrelator.DEFAULT_WINDOW_MINUTES == 60


def test_weight_sum_is_1():
    """4 维度权重之和 = 1.0。"""
    total = (
        IncidentCorrelator.WEIGHT_SERVICE
        + IncidentCorrelator.WEIGHT_ROOT_CAUSE
        + IncidentCorrelator.WEIGHT_EVIDENCE
        + IncidentCorrelator.WEIGHT_STEPS
    )
    assert abs(total - 1.0) < 0.001
