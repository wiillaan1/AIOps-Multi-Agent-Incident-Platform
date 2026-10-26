"""Checklist and dynamic evidence rule matching services."""

from __future__ import annotations

from app.models.rules import ChecklistItem, Rule, RuleMatch
from app.models.umodel import EntityType, InvestigationGraph


class Checklist:
    """Provide minimum required dimensions for each entity type."""

    def __init__(self) -> None:
        self._items: dict[EntityType, list[ChecklistItem]] = {
            EntityType.SERVICE: [
                ChecklistItem(entity_type=EntityType.SERVICE, dimension="error_rate", rationale="服务报错率"),
                ChecklistItem(entity_type=EntityType.SERVICE, dimension="latency_p95", rationale="尾延迟"),
                ChecklistItem(entity_type=EntityType.SERVICE, dimension="qps", rationale="流量"),
                ChecklistItem(entity_type=EntityType.SERVICE, dimension="recent_deployment", rationale="最近发布"),
                ChecklistItem(entity_type=EntityType.SERVICE, dimension="saturation_cpu", rationale="CPU 饱和度"),
            ],
            EntityType.POD: [
                ChecklistItem(entity_type=EntityType.POD, dimension="oomkilled", rationale="OOM"),
                ChecklistItem(entity_type=EntityType.POD, dimension="liveness_probe", rationale="存活探针"),
                ChecklistItem(entity_type=EntityType.POD, dimension="readiness_probe", rationale="就绪探针"),
                ChecklistItem(entity_type=EntityType.POD, dimension="restart_count", rationale="重启次数"),
                ChecklistItem(entity_type=EntityType.POD, dimension="memory_gc", rationale="GC 暂停"),
            ],
            EntityType.TOPIC: [
                ChecklistItem(entity_type=EntityType.TOPIC, dimension="consumer_lag", rationale="消费者延迟"),
                ChecklistItem(entity_type=EntityType.TOPIC, dimension="produce_rate", rationale="生产速率"),
                ChecklistItem(entity_type=EntityType.TOPIC, dimension="backlog", rationale="积压"),
            ],
            EntityType.DB: [
                ChecklistItem(entity_type=EntityType.DB, dimension="connection_pool", rationale="连接池"),
                ChecklistItem(entity_type=EntityType.DB, dimension="replication_lag", rationale="复制延迟"),
                ChecklistItem(entity_type=EntityType.DB, dimension="slow_query", rationale="慢查询"),
                ChecklistItem(entity_type=EntityType.DB, dimension="cpu_saturation", rationale="DB CPU"),
            ],
            EntityType.NODE: [
                ChecklistItem(entity_type=EntityType.NODE, dimension="cpu", rationale="节点 CPU"),
                ChecklistItem(entity_type=EntityType.NODE, dimension="memory", rationale="节点内存"),
                ChecklistItem(entity_type=EntityType.NODE, dimension="disk_pressure", rationale="磁盘压力"),
                ChecklistItem(entity_type=EntityType.NODE, dimension="network", rationale="网络"),
            ],
            EntityType.RUNBOOK: [
                ChecklistItem(entity_type=EntityType.RUNBOOK, dimension="matched_service", rationale="匹配服务"),
                ChecklistItem(entity_type=EntityType.RUNBOOK, dimension="last_validated", rationale="最近验证时间"),
            ],
        }

    def for_entity(self, entity_type: EntityType) -> list[ChecklistItem]:
        """Return a copy of the required dimensions for an entity type."""
        return list(self._items.get(entity_type, []))


class RuleEngine:
    """Match entity-scoped evidence against dynamic observation rules."""

    def __init__(self) -> None:
        self._rules: list[Rule] = [
            Rule(rule_id="high_cpu_service", trigger_entity_type=EntityType.SERVICE, condition_evidence_pattern="cpu", produces_observation_plan=["check_cpu_by_pod", "check_gc_pause", "check_thread_dump"], priority=80),
            Rule(rule_id="high_latency_service", trigger_entity_type=EntityType.SERVICE, condition_evidence_pattern="latency", produces_observation_plan=["check_downstream_latency", "check_db_lag", "check_redis_hit"], priority=75),
            Rule(rule_id="pod_oom", trigger_entity_type=EntityType.POD, condition_evidence_pattern="oomkilled", produces_observation_plan=["check_memory_limit", "check_oom_kills_total", "check_jvm_heap"], priority=90),
            Rule(rule_id="pod_restart", trigger_entity_type=EntityType.POD, condition_evidence_pattern="restart", produces_observation_plan=["check_previous_logs", "check_lifecycle", "check_config_change"], priority=70),
            Rule(rule_id="topic_lag", trigger_entity_type=EntityType.TOPIC, condition_evidence_pattern="lag", produces_observation_plan=["check_consumer_count", "check_partition_balance", "check_producer_rate"], priority=60),
            Rule(rule_id="db_slow_query", trigger_entity_type=EntityType.DB, condition_evidence_pattern="slow", produces_observation_plan=["check_index_usage", "check_lock_wait", "check_connections"], priority=65),
        ]

    def add_rule(self, rule: Rule) -> None:
        """Register a rule for subsequent matching."""
        self._rules.append(rule)

    def match(self, graph: InvestigationGraph, entity_id: str) -> list[RuleMatch]:
        """Return priority-ordered rules matching an entity's evidence."""
        entity = graph.entities.get(entity_id)
        if entity is None:
            return []
        evidence = [block for block in graph.evidence_blocks if block.object_ref == entity_id]
        matches: list[RuleMatch] = []
        for rule in self._rules:
            if rule.trigger_entity_type != entity.entity_type:
                continue
            matched = [block for block in evidence if rule.condition_evidence_pattern.lower() in block.observation.lower()]
            if matched:
                matches.append(RuleMatch(rule=rule, matched_evidence_ids=[block.block_id for block in matched], produced_observations=list(rule.produces_observation_plan)))
        matches.sort(key=lambda item: item.rule.priority, reverse=True)
        return matches
