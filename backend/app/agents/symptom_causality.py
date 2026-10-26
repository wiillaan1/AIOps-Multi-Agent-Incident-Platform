"""多症状告警的朴素因果推断（基于历史频率 + 时序先后）。

业务动机：当一个 incident 携带 N 个告警（例如 payment 延迟 + account
线程等待 + redis 缓存命中率下降），LLM 可以综合给出"共享根因 =
mysql-primary 慢查询"，但**成本高且不可解释**。

本模块给出**朴素可解释**的因果打分：

    naive_causal_prob(A_i → A_j)
      = 0.7 × P(co_occurrence in history with shared root)
        + 0.3 × P(time sequence matches)

两路信号互补：
- co_occurrence：经验性——"历史上这俩告警经常一起出，且共享同一根因"
- time_sequence：结构性——"i 先发生、j 后发生"，符合因果时序

阈值 ≥ MIN_SHARED_CAUSALITY 的 pairs 聚类成 SharedRootCauseHypothesis。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Sequence

# —— 阈值常量（公开，方便测试和外部调整） ——————————————————
MIN_SHARED_CAUSALITY = 0.5  # 高于此值认为"有共享根因嫌疑"
CO_OCCURRENCE_WEIGHT = 0.7
TIME_SEQUENCE_WEIGHT = 0.3
MIN_HISTORICAL_INCIDENTS = 3  # 历史不足时跳过 co_occurrence，纯靠时序


@dataclass
class SymptomCausalityPair:
    """一对告警的朴素因果关系（i → j 方向）。"""

    alert_i_id: str
    alert_j_id: str
    naive_causal_prob: float  # 0-1
    co_occurrence_prob: float  # 0-1，co-occurrence 信号单独打分
    time_sequence_match: bool  # i 早于 j（在容差内）
    co_occurrence_count: int  # 历史同时出现次数
    historical_shared_root: str | None = None  # 历史最常出现的共享根因
    notes: list[str] = field(default_factory=list)


@dataclass
class SharedRootCauseHypothesis:
    """从多个因果对聚类出来的"共享根因"假设。"""

    root_cause_service: str
    root_cause_metric: str
    support_pairs: list[tuple[str, str]]  # 支持该假设的 (alert_i_id, alert_j_id) 对
    aggregated_prob: float  # 综合概率
    contributing_alert_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SymptomCausalityResult:
    """_infer_symptom_causality 的完整输出。"""

    pairs: list[SymptomCausalityPair]
    shared_hypotheses: list[SharedRootCauseHypothesis]
    alerts_processed: int
    historical_incidents_used: int
    computation_method: str = "naive_cooccurrence_timing_v1"
    notes: list[str] = field(default_factory=list)


def _alert_key(alert: Any) -> str:
    """提取告警的稳定 key：(service, metric)。

    支持：
    - AlertEvent 对象（有 .service / .metric 属性）
    - dict（含 "service" / "metric" key）
    """
    if isinstance(alert, dict):
        svc = alert.get("service", "") or ""
        mtr = alert.get("metric", "") or ""
    else:
        svc = getattr(alert, "service", "") or ""
        mtr = getattr(alert, "metric", "") or ""
    return f"{svc}::{mtr}"


def _alert_time(alert: Any) -> datetime | None:
    """提取告警时间戳。AlertEvent 通常有 timestamp / created_at 字段。"""
    if isinstance(alert, dict):
        for key in ("timestamp", "created_at", "time"):
            ts = alert.get(key)
            if isinstance(ts, datetime):
                return ts
    else:
        for attr in ("timestamp", "created_at", "time"):
            ts = getattr(alert, attr, None)
            if isinstance(ts, datetime):
                return ts
    return None


def _incident_alert_key(incident: Any) -> set[str]:
    """提取历史 incident 涉及的 alert keys。

    来源（按优先级）：
    1. incident.alert_event（主告警）
    2. incident.context["additional_alerts"]（多症状场景：incident 携带的额外告警列表）
    3. incident.context["symptoms"]（备用：症状字段）
    """
    keys: set[str] = set()
    ae = getattr(incident, "alert_event", None)
    if ae is not None:
        keys.add(_alert_key(ae))

    ctx = getattr(incident, "context", None) or {}
    if isinstance(ctx, dict):
        # 优先查 additional_alerts（标准字段）
        additional = ctx.get("additional_alerts") or []
        for a in additional:
            keys.add(_alert_key(a))
        # 兜底：symptoms（部分老数据）
        symptoms = ctx.get("symptoms") or []
        for s in symptoms:
            if isinstance(s, dict):
                svc = s.get("service", "")
                mtr = s.get("metric", "")
                if svc or mtr:
                    keys.add(f"{svc}::{mtr}")

    return keys


def _incident_root_cause(incident: Any) -> str | None:
    """提取历史 incident 的根因 service（来自 rca_event.root_cause 或类似）。"""
    rca = getattr(incident, "rca_event", None)
    if rca is None:
        return None
    # 兼容两种字段：root_cause: str（service）或 root_cause_service: str
    rc = getattr(rca, "root_cause", None)
    if rc:
        return rc
    return getattr(rca, "root_cause_service", None)


def _incident_time(incident: Any) -> datetime | None:
    return getattr(incident, "created_at", None)


def _compute_pair_score(
    *,
    pair: tuple[str, str],
    co_occurrence_count: int,
    shared_root: str | None,
    total_history: int,
    alerts_by_key: dict[str, Any],
    time_slop_seconds: int,
) -> tuple[float, float, bool]:
    """
    计算一对告警 (i, j) 的朴素因果概率。

    内部按**时间顺序**确定 i/j：先发生的当 i，后发生的当 j。
    `pair` 是 alphabetical 排序的 key tuple，但语义上我们只关心"是否有时序"。

    Returns:
        (naive_causal_prob, co_occurrence_prob, time_match)
    """
    a_alpha_0 = alerts_by_key[pair[0]]
    a_alpha_1 = alerts_by_key[pair[1]]
    t_alpha_0 = _alert_time(a_alpha_0)
    t_alpha_1 = _alert_time(a_alpha_1)

    # 按时间排序：早的当 i，晚的当 j（缺时间戳则保持 alphabetical 顺序）
    if t_alpha_0 is not None and t_alpha_1 is not None and t_alpha_0 > t_alpha_1:
        a_i, a_j = a_alpha_1, a_alpha_0
        t_i, t_j = t_alpha_1, t_alpha_0
    else:
        a_i, a_j = a_alpha_0, a_alpha_1
        t_i, t_j = t_alpha_0, t_alpha_1

    # co_occurrence_prob（0-1）
    if total_history <= 0:
        co_occ_prob = 0.0
    else:
        # Laplace smoothing：分子 + 1，分母 + 2，避免零概率
        co_occ_prob = (co_occurrence_count + 1) / (total_history + 2)

    # time_sequence_match
    if t_i is None or t_j is None:
        time_match = False  # 缺时间戳 → 视为无序，不参与时序打分
    else:
        delta = (t_j - t_i).total_seconds()
        # i 先 j 后（i 在 j 前 ≥ slop 秒）→ match
        time_match = delta >= time_slop_seconds

    # 综合：加权平均（time_match → 1.0 否则 0.5）
    time_score = 1.0 if time_match else 0.5
    naive_prob = (
        CO_OCCURRENCE_WEIGHT * co_occ_prob
        + TIME_SEQUENCE_WEIGHT * time_score
    )

    return naive_prob, co_occ_prob, time_match


def _infer_symptom_causality(
    alerts: Sequence[Any],
    historical_incidents: Sequence[Any],
    *,
    co_occurrence_window_minutes: int = 30,
    time_sequence_slop_seconds: int = 60,
    min_co_occurrence: int = 1,
) -> SymptomCausalityResult:
    """
    多症状告警的朴素因果推断。

    Args:
        alerts: 当前 incident 携带的多个告警（至少 2 个才有意义）
        historical_incidents: 历史 incident 列表，用于计算 co-occurrence
        co_occurrence_window_minutes: 视为"同时出现"的时间窗口
        time_sequence_slop_seconds: 时序匹配的容差
        min_co_occurrence: 历史 co-occurrence 至少这么多次才纳入打分

    Returns:
        SymptomCausalityResult 含所有 pairs + 共享根因假设列表
    """
    notes: list[str] = []

    # 防御：少于 2 个告警 → 朴素因果无意义
    if len(alerts) < 2:
        notes.append("alerts < 2 → 朴素因果无意义，返回空结果")
        return SymptomCausalityResult(
            pairs=[],
            shared_hypotheses=[],
            alerts_processed=len(alerts),
            historical_incidents_used=len(historical_incidents),
            notes=notes,
        )

    alerts_now = {a.event_id: a for a in alerts if getattr(a, "event_id", None)}
    alerts_by_key = {_alert_key(a): a for a in alerts_now.values()}
    if len(alerts_now) < 2:
        notes.append("缺少 event_id → 无法配对")
        return SymptomCausalityResult(
            pairs=[],
            shared_hypotheses=[],
            alerts_processed=len(alerts),
            historical_incidents_used=len(historical_incidents),
            notes=notes,
        )

    # Step 1: 统计历史 co-occurrence
    # key: (alert_key_i, alert_key_j) → (count, most_common_shared_root)
    co_occurrence_table: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "roots": defaultdict(int)}
    )

    total_incidents_with_root = 0
    history_window = timedelta(minutes=co_occurrence_window_minutes)

    for inc in historical_incidents:
        inc_keys = _incident_alert_key(inc)
        if not inc_keys:
            continue
        root = _incident_root_cause(inc)
        if root is None:
            continue
        total_incidents_with_root += 1

        # 同一个 incident 内所有 alert keys 配对
        sorted_keys = sorted(inc_keys)
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                pair_key = (sorted_keys[i], sorted_keys[j])
                co_occurrence_table[pair_key]["count"] += 1
                co_occurrence_table[pair_key]["roots"][root] += 1

    # Step 2: 计算当前所有 alert pair 的朴素因果概率
    alert_id_list = list(alerts_now.keys())
    pairs_out: list[SymptomCausalityPair] = []

    for i_idx in range(len(alert_id_list)):
        for j_idx in range(i_idx + 1, len(alert_id_list)):
            a_i = alerts_now[alert_id_list[i_idx]]
            a_j = alerts_now[alert_id_list[j_idx]]
            key_i = _alert_key(a_i)
            key_j = _alert_key(a_j)
            pair_key = (min(key_i, key_j), max(key_i, key_j))

            entry = co_occurrence_table.get(pair_key, {"count": 0, "roots": {}})
            co_occurrence_count = entry["count"]
            # 找出最常见的共享根因
            roots_counter = entry.get("roots", {})
            shared_root = (
                max(roots_counter.items(), key=lambda kv: kv[1])[0]
                if roots_counter
                else None
            )

            if (
                len(historical_incidents) >= MIN_HISTORICAL_INCIDENTS
                and co_occurrence_count < min_co_occurrence
            ):
                # 历史不足且当前 pair 历史也少 → 跳过（避免噪声）
                continue

            naive_prob, co_occ_prob, time_match = _compute_pair_score(
                pair=pair_key,
                co_occurrence_count=co_occurrence_count,
                shared_root=shared_root,
                total_history=total_incidents_with_root,
                alerts_by_key=alerts_by_key,
                time_slop_seconds=time_sequence_slop_seconds,
            )

            pair_notes: list[str] = []
            if co_occurrence_count == 0:
                pair_notes.append("历史未同时出现")
            if not time_match:
                pair_notes.append("时序无序或缺时间戳")

            pairs_out.append(
                SymptomCausalityPair(
                    alert_i_id=a_i.event_id,
                    alert_j_id=a_j.event_id,
                    naive_causal_prob=round(naive_prob, 4),
                    co_occurrence_prob=round(co_occ_prob, 4),
                    time_sequence_match=time_match,
                    co_occurrence_count=co_occurrence_count,
                    historical_shared_root=shared_root,
                    notes=pair_notes,
                )
            )

    # Step 3: 聚类共享根因假设
    # 按 shared_root 分组（共享同一个根因的 pairs 聚到一起）
    by_root: dict[str, list[SymptomCausalityPair]] = defaultdict(list)
    for p in pairs_out:
        if p.naive_causal_prob >= MIN_SHARED_CAUSALITY and p.historical_shared_root:
            by_root[p.historical_shared_root].append(p)

    hypotheses: list[SharedRootCauseHypothesis] = []
    for root_service, ps in by_root.items():
        # aggregated_prob = 该 root 下所有 pair 的平均 naive_causal_prob
        avg_prob = sum(p.naive_causal_prob for p in ps) / len(ps)
        # 拆 root_cause_service 为 service + metric（"service::metric" 格式）
        root_parts = root_service.split("::", 1)
        root_svc = root_parts[0]
        root_metric = root_parts[1] if len(root_parts) > 1 else ""

        contributing = set()
        for p in ps:
            contributing.add(p.alert_i_id)
            contributing.add(p.alert_j_id)

        hypotheses.append(
            SharedRootCauseHypothesis(
                root_cause_service=root_svc,
                root_cause_metric=root_metric,
                support_pairs=[(p.alert_i_id, p.alert_j_id) for p in ps],
                aggregated_prob=round(avg_prob, 4),
                contributing_alert_ids=sorted(contributing),
                evidence={
                    "co_occurrence_counts": [p.co_occurrence_count for p in ps],
                    "time_matches": [p.time_sequence_match for p in ps],
                },
            )
        )

    # 按 aggregated_prob 降序
    hypotheses.sort(key=lambda h: h.aggregated_prob, reverse=True)

    # Step 4: 备注
    if len(historical_incidents) < MIN_HISTORICAL_INCIDENTS:
        notes.append(
            f"历史 incidents 仅 {len(historical_incidents)} 个（< {MIN_HISTORICAL_INCIDENTS}），"
            "co_occurrence 信号弱，结果以时序为主"
        )

    return SymptomCausalityResult(
        pairs=pairs_out,
        shared_hypotheses=hypotheses,
        alerts_processed=len(alerts),
        historical_incidents_used=len(historical_incidents),
        notes=notes,
    )


__all__ = [
    "SymptomCausalityPair",
    "SharedRootCauseHypothesis",
    "SymptomCausalityResult",
    "_infer_symptom_causality",
    "MIN_SHARED_CAUSALITY",
]