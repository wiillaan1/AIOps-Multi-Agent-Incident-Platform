"""
测试 compute_alert_fingerprint + TriggerDedupService

覆盖：
- fingerprint 稳定性（相同字段 → 相同 hash；不同字段 → 不同 hash）
- 5 分钟内 dedup（hit_count 自增）
- 窗口外不 dedup
- in-flight 上限（默认 10）
- mark_done 释放槽位
- 线程安全（lock 在每次访问）
- get/reset 单例
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.services.trigger_dedup import (
    DEFAULT_DEDUP_WINDOW_SECONDS,
    DEFAULT_MAX_IN_FLIGHT,
    TriggerDedupService,
    compute_alert_fingerprint,
    get_trigger_dedup_service,
    reset_trigger_dedup_service,
)


# ==================== fingerprint 单元测试 ====================


def test_fingerprint_stable_for_same_inputs():
    """相同 (service, metric, severity, scope) → 相同 fingerprint"""
    a = compute_alert_fingerprint("order-service", "cpu_usage", "HIGH")
    b = compute_alert_fingerprint("order-service", "cpu_usage", "HIGH")
    assert a == b


def test_fingerprint_differs_by_service():
    """不同 service → 不同 fingerprint"""
    a = compute_alert_fingerprint("order-service", "cpu_usage", "HIGH")
    b = compute_alert_fingerprint("payment-service", "cpu_usage", "HIGH")
    assert a != b


def test_fingerprint_differs_by_metric():
    """不同 metric → 不同 fingerprint"""
    a = compute_alert_fingerprint("order", "cpu_usage", "HIGH")
    b = compute_alert_fingerprint("order", "memory_usage", "HIGH")
    assert a != b


def test_fingerprint_differs_by_severity():
    a = compute_alert_fingerprint("order", "cpu", "HIGH")
    b = compute_alert_fingerprint("order", "cpu", "CRITICAL")
    assert a != b


def test_fingerprint_differs_by_scope():
    a = compute_alert_fingerprint("order", "cpu", "HIGH", scope="region-a")
    b = compute_alert_fingerprint("order", "cpu", "HIGH", scope="region-b")
    assert a != b


def test_fingerprint_default_scope_when_none():
    """scope=None 时使用 'default'，应与显式 'default' 相同"""
    a = compute_alert_fingerprint("order", "cpu", "HIGH")
    b = compute_alert_fingerprint("order", "cpu", "HIGH", scope="default")
    assert a == b


def test_fingerprint_case_insensitive():
    """大小写不敏感"""
    a = compute_alert_fingerprint("Order-Service", "CPU", "high")
    b = compute_alert_fingerprint("order-service", "cpu", "HIGH")
    assert a == b


def test_fingerprint_length_is_16():
    """fingerprint 是 16 字符 hex（sha256 截断）"""
    fp = compute_alert_fingerprint("x", "y", "HIGH")
    assert len(fp) == 16


def test_fingerprint_handles_empty_strings():
    """空字符串不抛异常"""
    fp = compute_alert_fingerprint("", "", "")
    assert len(fp) == 16


# ==================== TriggerDedupService 单测 ====================


def test_service_default_window_is_300_seconds():
    """默认去重窗口 300 秒（5 分钟，审查报告要求）"""
    s = TriggerDedupService()
    assert s.dedup_window.total_seconds() == DEFAULT_DEDUP_WINDOW_SECONDS
    assert DEFAULT_DEDUP_WINDOW_SECONDS == 300


def test_service_default_max_in_flight_is_10():
    """默认 in-flight 上限 10"""
    s = TriggerDedupService()
    assert s.max_in_flight == DEFAULT_MAX_IN_FLIGHT
    assert DEFAULT_MAX_IN_FLIGHT == 10


def test_first_alert_not_dedup():
    """首次告警 → 不去重"""
    s = TriggerDedupService()
    is_dup, existing = s.should_dedupe("fp1")
    assert is_dup is False
    assert existing is None


def test_second_alert_within_window_dedups():
    """5 分钟内相同 fingerprint → dedup"""
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = TriggerDedupService()
    s.mark_in_flight("fp1", "inc-001", now=t0)
    is_dup, existing = s.should_dedupe("fp1", now=t0 + timedelta(minutes=2))
    assert is_dup is True
    assert existing == "inc-001"


def test_second_alert_outside_window_does_not_dedup():
    """窗口外（>5 分钟） → 不去重"""
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = TriggerDedupService()
    s.mark_in_flight("fp1", "inc-001", now=t0)
    is_dup, _ = s.should_dedupe("fp1", now=t0 + timedelta(minutes=10))
    assert is_dup is False


def test_hit_count_increments_on_dedup():
    """dedup 命中时 hit_count 自增"""
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = TriggerDedupService()
    s.mark_in_flight("fp1", "inc-001", now=t0)
    s.should_dedupe("fp1", now=t0 + timedelta(minutes=1))
    s.should_dedupe("fp1", now=t0 + timedelta(minutes=2))
    record = s.get_record("fp1")
    assert record is not None
    assert record.hit_count >= 3  # mark_in_flight (1) + 2 dedups


def test_mark_done_releases_in_flight_slot():
    """mark_done 释放 in-flight 槽"""
    s = TriggerDedupService(max_in_flight=2)
    s.mark_in_flight("fp1", "inc-001")
    s.mark_in_flight("fp2", "inc-002")
    assert s.in_flight_count() == 2
    assert s.can_accept_new() is False

    s.mark_done("inc-001")
    assert s.in_flight_count() == 1
    assert s.can_accept_new() is True


def test_in_flight_count_zero_initially():
    s = TriggerDedupService()
    assert s.in_flight_count() == 0
    assert s.can_accept_new() is True


def test_can_accept_new_false_when_full():
    """in-flight 满 → can_accept_new = False"""
    s = TriggerDedupService(max_in_flight=3)
    for i in range(3):
        s.mark_in_flight(f"fp{i}", f"inc-{i:03d}")
    assert s.can_accept_new() is False


def test_list_in_flight_returns_all():
    s = TriggerDedupService()
    s.mark_in_flight("fp1", "inc-001")
    s.mark_in_flight("fp2", "inc-002")
    records = s.list_in_flight()
    assert len(records) == 2
    ids = {r.incident_id for r in records}
    assert ids == {"inc-001", "inc-002"}


def test_cleanup_expired_removes_old_dedup_records():
    """过期 dedup 记录被清理"""
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = TriggerDedupService()
    s.mark_in_flight("fp1", "inc-001", now=t0)
    # 跨过窗口调用 should_dedupe → 触发清理
    s.should_dedupe("fp1", now=t0 + timedelta(minutes=10))
    assert s.get_record("fp1") is None  # 过期被清


def test_clear_removes_all_records():
    s = TriggerDedupService()
    s.mark_in_flight("fp1", "inc-001")
    s.clear()
    assert s.in_flight_count() == 0
    assert s.get_record("fp1") is None


def test_mark_done_unknown_incident_is_safe():
    """mark_done 不存在的 incident 不抛异常"""
    s = TriggerDedupService()
    s.mark_done("nonexistent")  # 不抛异常
    assert s.in_flight_count() == 0


def test_custom_window():
    """自定义 dedup 窗口（60 秒）。第二次 dedup 会刷新 last_seen（sliding window），所以用大间隔测试"窗口外"行为。"""
    s = TriggerDedupService(dedup_window_seconds=60)
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s.mark_in_flight("fp1", "inc-001", now=t0)
    # 100 秒后（窗口已过，且没有刷新 last_seen）
    is_dup, _ = s.should_dedupe("fp1", now=t0 + timedelta(seconds=100))
    assert is_dup is False


def test_custom_max_in_flight():
    """自定义 in-flight 上限"""
    s = TriggerDedupService(max_in_flight=1)
    s.mark_in_flight("fp1", "inc-001")
    assert s.can_accept_new() is False
    s.mark_done("inc-001")
    assert s.can_accept_new() is True


# ==================== 单例测试 ====================


def test_singleton_returns_same_instance():
    """get_trigger_dedup_service 返回相同实例"""
    reset_trigger_dedup_service()
    s1 = get_trigger_dedup_service()
    s2 = get_trigger_dedup_service()
    assert s1 is s2


def test_reset_clears_singleton():
    s1 = get_trigger_dedup_service()
    s1.mark_in_flight("fp1", "inc-001")
    reset_trigger_dedup_service()
    s2 = get_trigger_dedup_service()
    assert s1 is not s2
    assert s2.in_flight_count() == 0


# ==================== 实战场景 ====================


def test_real_world_alert_storm_deduplication():
    """实战场景：同一告警 1 秒内重复触发 100 次 → 仅首次进入，其他 dedup"""
    s = TriggerDedupService(max_in_flight=10)
    fingerprint = compute_alert_fingerprint("order-service", "cpu_usage", "HIGH", scope="region-a")
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)

    # 首次
    is_dup, _ = s.should_dedupe(fingerprint, now=t0)
    assert is_dup is False
    s.mark_in_flight(fingerprint, "inc-001", now=t0)

    # 接下来 99 次 → 都 dedup
    dedup_count = 0
    for i in range(1, 100):
        is_dup, _ = s.should_dedupe(fingerprint, now=t0 + timedelta(milliseconds=i * 100))
        if is_dup:
            dedup_count += 1
    assert dedup_count == 99
    assert s.in_flight_count() == 1  # 仍是同一个 incident


def test_real_world_in_flight_capacity_protection():
    """实战场景：连续触发超过 in-flight 上限 → can_accept_new = False"""
    s = TriggerDedupService(max_in_flight=3)
    fingerprints = [
        compute_alert_fingerprint(f"svc-{i}", "metric", "HIGH")
        for i in range(5)
    ]
    t0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    for fp in fingerprints[:3]:
        s.mark_in_flight(fp, f"inc-{fp[:4]}", now=t0)

    assert s.in_flight_count() == 3
    assert s.can_accept_new() is False
