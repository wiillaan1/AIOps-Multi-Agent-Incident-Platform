"""
测试 OncallScheduler + OncallScheduleStore

覆盖：
- OncallSchedule 时间窗口 + service_scope 匹配
- JSONL 持久化 + startup 恢复
- get_current_oncall 策略（CRITICAL/HIGH 找 L1；MEDIUM/LOW 返回 None）
- escalate 链路（L1 → L2 → LEAD → None）
- L1 不在时降级找 L2（不在时找 LEAD）
- 默认演示排班安装
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.services.oncall_scheduler import (
    DEFAULT_ESCALATION_ROLES,
    OncallAssignment,
    OncallSchedule,
    OncallScheduleStore,
    OncallScheduler,
    install_default_demo_schedules,
)


# ==================== OncallSchedule 单测 ====================


def test_schedule_covers_when_in_window_and_service_match():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = OncallSchedule(
        user_id="alice",
        user_email="alice@example.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        service_scope=["order-service"],
    )
    assert s.covers("order-service", now=now) is True


def test_schedule_not_cover_outside_window():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=3),
        end_time=now - timedelta(hours=1),  # 已过期
        service_scope=["order-service"],
    )
    assert s.covers("order-service", now=now) is False


def test_schedule_not_cover_wrong_service():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        service_scope=["order-service"],
    )
    assert s.covers("payment-service", now=now) is False


def test_schedule_empty_scope_covers_all_services():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        service_scope=[],  # 空 scope = 全服务
    )
    assert s.covers("any-service", now=now) is True
    assert s.covers(None, now=now) is True


def test_schedule_covers_when_service_none_and_scope_set():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    s = OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        service_scope=["order-service"],
    )
    # service_name=None 时 → True（任意 scope 都行）
    assert s.covers(None, now=now) is True


# ==================== Store 单测 ====================


def test_store_persistence_roundtrip(tmp_path: Path):
    store_path = str(tmp_path / "oncall.jsonl")
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)

    s1 = OncallScheduleStore(store_path=store_path)
    s1.add(OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    ))

    s2 = OncallScheduleStore(store_path=store_path)
    assert len(s2.all()) == 1
    assert s2.all()[0].user_id == "alice"


def test_store_ignores_corrupted_lines(tmp_path: Path):
    path = tmp_path / "oncall.jsonl"
    valid = OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=datetime(2026, 9, 10, 10, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 9, 10, 18, 0, tzinfo=timezone.utc),
    )
    path.write_text(
        f"{json.dumps(valid.to_dict(), ensure_ascii=False)}\n"
        "not-json\n",
        encoding="utf-8",
    )
    s = OncallScheduleStore(store_path=str(path))
    assert len(s.all()) == 1  # 损坏行跳过


# ==================== Scheduler 单测 ====================


def test_get_current_oncall_critical_finds_l1():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_a.jsonl")
    store.add(OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        service_scope=["order-service"],
    ))
    scheduler = OncallScheduler(store)
    a = scheduler.get_current_oncall(service_name="order-service", severity="CRITICAL", now=now)
    assert a is not None
    assert a.user_id == "alice"
    assert a.role == "L1"
    assert a.escalation_level == 0


def test_get_current_oncall_low_returns_none():
    """MEDIUM/LOW 静默记录，不主动通知"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_b.jsonl")
    store.add(OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    ))
    scheduler = OncallScheduler(store)
    assert scheduler.get_current_oncall(severity="LOW", now=now) is None
    assert scheduler.get_current_oncall(severity="MEDIUM", now=now) is None


def test_get_current_oncall_falls_back_to_l2_when_no_l1():
    """L1 不在 → 找 L2"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_c.jsonl")
    store.add(OncallSchedule(
        user_id="bob",
        user_email="b@x.com",
        role="L2",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    ))
    scheduler = OncallScheduler(store)
    a = scheduler.get_current_oncall(severity="CRITICAL", now=now)
    assert a is not None
    assert a.role == "L2"
    assert a.escalation_level == 1


def test_get_current_oncall_falls_back_to_lead():
    """L1/L2 都不在 → 找 LEAD"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_d.jsonl")
    store.add(OncallSchedule(
        user_id="carol",
        user_email="c@x.com",
        role="LEAD",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    ))
    scheduler = OncallScheduler(store)
    a = scheduler.get_current_oncall(severity="HIGH", now=now)
    assert a is not None
    assert a.role == "LEAD"
    assert a.escalation_level == 2


def test_get_current_oncall_no_match_returns_none():
    """只有 LEAD 在岗，但 escalation_chain 不含 LEAD → 返回 None"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_e.jsonl")
    store.add(OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="observer",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
    ))
    scheduler = OncallScheduler(store, escalation_chain=["L1", "L2"])
    a = scheduler.get_current_oncall(severity="CRITICAL", now=now)
    assert a is None


def test_escalate_l1_to_l2():
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_f.jsonl")
    store.add(OncallSchedule(user_id="alice", user_email="a@x.com", role="L1",
                              start_time=now - timedelta(hours=1),
                              end_time=now + timedelta(hours=1)))
    store.add(OncallSchedule(user_id="bob", user_email="b@x.com", role="L2",
                              start_time=now - timedelta(hours=1),
                              end_time=now + timedelta(hours=1)))
    scheduler = OncallScheduler(store)
    l1 = scheduler.get_current_oncall(severity="CRITICAL", now=now)
    l2 = scheduler.escalate(l1, now=now)
    assert l2 is not None
    assert l2.user_id == "bob"
    assert l2.role == "L2"
    assert l2.escalation_level == 1
    assert "escalated_from:alice:L1" in l2.reason


def test_escalate_from_lead_returns_none():
    """LEAD 已是顶层，再 escalate 返回 None"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_g.jsonl")
    store.add(OncallSchedule(user_id="carol", user_email="c@x.com", role="LEAD",
                              start_time=now - timedelta(hours=1),
                              end_time=now + timedelta(hours=1)))
    scheduler = OncallScheduler(store)
    lead = scheduler.get_current_oncall(severity="CRITICAL", now=now)
    assert lead.role == "LEAD"
    assert scheduler.escalate(lead, now=now) is None


def test_escalate_when_next_role_unavailable_returns_none():
    """L1 在，L2 不在 → escalate 返回 None"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_h.jsonl")
    store.add(OncallSchedule(user_id="alice", user_email="a@x.com", role="L1",
                              start_time=now - timedelta(hours=1),
                              end_time=now + timedelta(hours=1)))
    scheduler = OncallScheduler(store)
    l1 = scheduler.get_current_oncall(severity="CRITICAL", now=now)
    assert scheduler.escalate(l1, now=now) is None


def test_default_escalation_chain():
    """默认升级链是 L1 → L2 → LEAD"""
    assert DEFAULT_ESCALATION_ROLES == ["L1", "L2", "LEAD"]


def test_install_default_demo_schedules(tmp_path: Path):
    """默认演示排班安装成功"""
    store = OncallScheduleStore(store_path=str(tmp_path / "demo.jsonl"))
    install_default_demo_schedules(store)
    assert len(store.all()) >= 1


def test_get_current_oncall_respects_service_scope():
    """service_scope 不匹配的服务不返回该值班人"""
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    store = OncallScheduleStore(store_path="/tmp/_test_oncall_i.jsonl")
    store.add(OncallSchedule(
        user_id="alice",
        user_email="a@x.com",
        role="L1",
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1),
        service_scope=["order-service"],  # 不含 payment-service
    ))
    scheduler = OncallScheduler(store)
    # 不在 scope 内 → 找不到 → None
    a = scheduler.get_current_oncall(service_name="payment-service", severity="CRITICAL", now=now)
    assert a is None
    # 在 scope 内 → 找到
    a = scheduler.get_current_oncall(service_name="order-service", severity="CRITICAL", now=now)
    assert a is not None
