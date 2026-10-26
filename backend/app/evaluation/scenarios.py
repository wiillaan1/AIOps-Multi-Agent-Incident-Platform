"""
AIOps Agent Platform - Scenario Registry

把 `datasets.FAULT_SCENARIOS` 与 `business_monitor_agent.BUSINESS_RULES`
统一映射为 19 条 `ScenarioSpec`，每条带评测维度选择 + ground_truth +
relevant_docs + expected_action 字段。供 `ScenarioDrivenEvaluator` 与
`GoldenSetRunner` 共同消费。

设计原则：
1. 一个 scenario id 对应一个评测单元；
2. scenario_id 优先用 FAULT_SCENARIOS.id；BUSINESS_RULES 单独构成的条目
   （4 条无对应 FS）用 business_rule id 作为 key；
3. 每条 spec.dimensions 决定跑哪几个底层 evaluator，不强制跑满 4 个；
4. relevant_docs 全部引用 KNOWLEDGE_BASE 中真实存在的 doc.id。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvalDimension = Literal["e2e", "reasoning", "tool", "rag"]
EVAL_DIMENSIONS: list[str] = ["e2e", "reasoning", "tool", "rag"]

ScenarioSource = Literal["fault_scenario", "business_rule", "both"]


class ScenarioSpec(BaseModel):
    """单个评测场景的规约"""

    scenario_id: str = Field(description="场景 id（FS_* 或 br_*）")
    source: ScenarioSource = Field(description="数据来源")
    name: str = Field(description="场景名称")
    description: str = Field(default="", description="场景描述")
    category: str = Field(description="infrastructure / business / business_logic / financial / inventory / order / user")
    service: str = Field(default="", description="相关服务")
    severity: Literal["critical", "high", "medium", "low"] = Field(description="严重级别")

    # 评测维度：决定跑哪几个 evaluator
    dimensions: list[EvalDimension] = Field(description="必跑的评测维度")

    # 评测 ground truth
    ground_truth: dict[str, Any] = Field(description="根因/动作/审批等真值")
    relevant_docs: list[str] = Field(
        default_factory=list, description="RAG 评测期望命中的知识库条目 id"
    )

    # 期望行为字段（来自 FAULT_SCENARIOS.expected_action / BUSINESS_RULES）
    expected_actions: list[str] = Field(
        default_factory=list, description="期望动作列表"
    )
    expected_approval: str = Field(default="", description="期望审批路径")
    expected_level: str = Field(default="", description="期望自愈等级 L0/L1/L2")
    blast_radius: float = Field(default=0.0, description="期望爆炸半径")

    # 业务规则关联（仅 business_logic 类场景）
    business_rule: str = Field(default="", description="关联的业务规则 id")

    # 可选：变更/历史上下文
    context: dict[str, Any] = Field(
        default_factory=dict, description="额外上下文"
    )


# ============================================================================
# 19 条场景注册表
# ============================================================================
# 编号约定：
#   fs_001..fs_005           — 基础设施层（5 条）
#   fs_biz_001..fs_biz_006   — 指标驱动业务层（6 条）
#   fs_biz_007..fs_biz_010   — 业务规则驱动层（4 条，已关联 BUSINESS_RULES）
#   br_coupon_abuse / br_order_stuck / br_inventory_discrepancy /
#   br_user_permission_error — 仅 BUSINESS_RULES，无 FS（4 条）
# ============================================================================

SCENARIO_REGISTRY: dict[str, ScenarioSpec] = {
    # ----------------------------- 基础设施层 -----------------------------
    "fs_001": ScenarioSpec(
        scenario_id="fs_001",
        source="fault_scenario",
        name="Deployment-induced CPU spike",
        description="Recent deployment causes CPU usage to spike",
        category="infrastructure",
        service="order-service",
        severity="high",
        dimensions=["e2e", "reasoning", "tool"],
        ground_truth={
            "root_cause": "recent_deployment",
            "confidence": 0.85,
        },
        relevant_docs=["kb_001"],
        expected_actions=["rollback_deployment"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.15,
        context={"trigger_metric": "cpu_usage_percent"},
    ),
    "fs_002": ScenarioSpec(
        scenario_id="fs_002",
        source="fault_scenario",
        name="Memory leak leading to OOM",
        description="Gradual memory leak eventually causes OOM kills",
        category="infrastructure",
        service="payment-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool"],
        ground_truth={
            "root_cause": "memory_leak",
            "confidence": 0.9,
        },
        relevant_docs=["kb_002"],
        expected_actions=["restart_pod"],
        expected_approval="auto",
        expected_level="L0",
        blast_radius=0.05,
        context={"trigger_metric": "memory_usage_percent"},
    ),
    "fs_003": ScenarioSpec(
        scenario_id="fs_003",
        source="fault_scenario",
        name="Database connection pool exhausted",
        description="DB connection pool depletion causing cascading failures",
        category="infrastructure",
        service="user-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "database_connection_pool_exhausted",
            "confidence": 0.85,
        },
        relevant_docs=["kb_002", "kb_003"],
        expected_actions=["scale_up"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.20,
        context={"trigger_metric": "p99_latency_ms"},
    ),
    "fs_004": ScenarioSpec(
        scenario_id="fs_004",
        source="fault_scenario",
        name="Cache failure causing DB overload",
        description="Redis cache failure leads to direct DB queries",
        category="infrastructure",
        service="inventory-service",
        severity="high",
        dimensions=["e2e", "reasoning"],
        ground_truth={
            "root_cause": "cache_failure",
            "confidence": 0.8,
        },
        relevant_docs=["kb_003"],
        expected_actions=["restart_pod"],
        expected_approval="auto",
        expected_level="L0",
        blast_radius=0.10,
        context={"trigger_metric": "p99_latency_ms"},
    ),
    "fs_005": ScenarioSpec(
        scenario_id="fs_005",
        source="fault_scenario",
        name="Configuration error causing high error rate",
        description="Wrong configuration causes services to return errors",
        category="infrastructure",
        service="api-gateway",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool"],
        ground_truth={
            "root_cause": "configuration_error",
            "confidence": 0.85,
        },
        relevant_docs=["kb_001"],
        expected_actions=["rollback_config"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.25,
        context={"trigger_metric": "error_rate_percent"},
    ),
    # -------------------------- 指标驱动业务层 ----------------------------
    "fs_biz_001": ScenarioSpec(
        scenario_id="fs_biz_001",
        source="fault_scenario",
        name="Payment gateway timeout — orders piling up",
        description="Third-party payment gateway timeout causes payment failures and order backlog",
        category="business",
        service="payment-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "third_party_timeout",
            "confidence": 0.88,
        },
        relevant_docs=["kb_biz_001"],
        expected_actions=["enable_payment_fallback"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.30,
        context={"trigger_metric": "payment_failure_rate"},
    ),
    "fs_biz_002": ScenarioSpec(
        scenario_id="fs_biz_002",
        source="fault_scenario",
        name="Order processing stuck — inventory sync failure",
        description="Inventory service returns stale data causing order-processor to deadlock",
        category="business",
        service="order-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "inventory_data_stale",
            "confidence": 0.82,
        },
        relevant_docs=["kb_biz_002"],
        expected_actions=["reconcile_inventory"],
        expected_approval="team_lead",
        expected_level="L1",
        blast_radius=0.25,
        context={"trigger_metric": "order_backlog_count"},
    ),
    "fs_biz_003": ScenarioSpec(
        scenario_id="fs_biz_003",
        source="fault_scenario",
        name="User login failure spike — SSO token cache expired",
        description="Redis cache expires all SSO tokens causing mass login failures",
        category="business",
        service="user-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "token_cache_expired",
            "confidence": 0.9,
        },
        relevant_docs=["kb_biz_003"],
        expected_actions=["enable_static_auth"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.40,
        context={"trigger_metric": "auth_failure_rate"},
    ),
    "fs_biz_004": ScenarioSpec(
        scenario_id="fs_biz_004",
        source="fault_scenario",
        name="Rate limiter false-positive — blocking legitimate traffic",
        description="API gateway rate limiter misconfigured, blocking valid user requests",
        category="business",
        service="api-gateway",
        severity="high",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "rate_limit_misconfig",
            "confidence": 0.85,
        },
        relevant_docs=["kb_biz_004"],
        expected_actions=["adjust_rate_limit"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.10,
        context={"trigger_metric": "rate_limit_hit_rate"},
    ),
    "fs_biz_005": ScenarioSpec(
        scenario_id="fs_biz_005",
        source="fault_scenario",
        name="Message queue backlog — consumer group stuck",
        description="Kafka consumer group stuck on corrupted message causing lag",
        category="business",
        service="order-service",
        severity="high",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "corrupted_message",
            "confidence": 0.8,
        },
        relevant_docs=["kb_biz_005"],
        expected_actions=["skip_corrupted_message"],
        expected_approval="auto",
        expected_level="L0",
        blast_radius=0.12,
        context={"trigger_metric": "mq_consumer_lag"},
    ),
    "fs_biz_006": ScenarioSpec(
        scenario_id="fs_biz_006",
        source="fault_scenario",
        name="Data inconsistency — DB-ES replication lag",
        description="MySQL-to-Elasticsearch replication lag causes search to show wrong counts",
        category="business",
        service="inventory-service",
        severity="medium",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "replication_lag",
            "confidence": 0.78,
        },
        relevant_docs=["kb_biz_006"],
        expected_actions=["force_reindex"],
        expected_approval="team_lead",
        expected_level="L1",
        blast_radius=0.15,
        context={"trigger_metric": "replication_lag_seconds"},
    ),
    # -------------------------- 业务规则驱动层 ----------------------------
    # 这 4 条同时来自 FAULT_SCENARIOS 和 BUSINESS_RULES，source="both"
    "fs_biz_007": ScenarioSpec(
        scenario_id="fs_biz_007",
        source="both",
        name="重复扣款 — 支付幂等性失败",
        description="支付回调因网络抖动重试 3 次，导致同一订单被扣款 3 次",
        category="business_logic",
        service="payment-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "payment_idempotency_failure",
            "confidence": 0.95,
        },
        relevant_docs=["kb_biz_007"],
        expected_actions=["refund_duplicates"],
        expected_approval="team_lead",
        expected_level="L2",
        blast_radius=0.05,
        business_rule="br_duplicate_charge",
    ),
    "fs_biz_008": ScenarioSpec(
        scenario_id="fs_biz_008",
        source="both",
        name="库存超卖 — 并发下单竞态条件",
        description="秒杀活动期间库存缓存未及时更新，超卖订单被生成",
        category="business_logic",
        service="inventory-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "inventory_sync_failure",
            "confidence": 0.92,
        },
        relevant_docs=["kb_biz_008"],
        expected_actions=["cancel_oversold_orders"],
        expected_approval="team_lead",
        expected_level="L2",
        blast_radius=0.15,
        business_rule="br_oversell",
    ),
    "fs_biz_009": ScenarioSpec(
        scenario_id="fs_biz_009",
        source="both",
        name="订单金额对账不平 — 优惠券计算错误",
        description="满减券在多 SKU 订单中比例分摊计算有精度问题",
        category="business_logic",
        service="order-service",
        severity="high",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "accounting_reconciliation_error",
            "confidence": 0.85,
        },
        relevant_docs=["kb_biz_009"],
        expected_actions=["reconcile_transactions"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.08,
        business_rule="br_amount_mismatch",
    ),
    "fs_biz_010": ScenarioSpec(
        scenario_id="fs_biz_010",
        source="both",
        name="支付回调丢失 — 订单状态不一致",
        description="支付网关回调 webhook 因证书过期返回 500，订单状态未更新",
        category="business_logic",
        service="payment-service",
        severity="critical",
        dimensions=["e2e", "reasoning", "tool", "rag"],
        ground_truth={
            "root_cause": "third_party_callback_loss",
            "confidence": 0.9,
        },
        relevant_docs=["kb_biz_010"],
        expected_actions=["reconcile_payment_status"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.10,
        business_rule="br_payment_callback_loss",
    ),
    # ------------------------ 仅 BUSINESS_RULES 条目 ------------------------
    # 这 4 条仅来自 BUSINESS_RULES，无对应 FS_*；主要靠业务监控通道触发。
    # 评测时仅做推理 + 端到端（缺失脚本触发的 metric/log 上下文）。
    "br_coupon_abuse": ScenarioSpec(
        scenario_id="br_coupon_abuse",
        source="business_rule",
        name="优惠券滥用检测",
        description="同一用户重复使用已消费的优惠券",
        category="financial",
        service="order-service",
        severity="high",
        dimensions=["e2e", "reasoning"],
        ground_truth={
            "root_cause": "coupon_idempotency_failure",
            "confidence": 0.85,
        },
        relevant_docs=["kb_biz_007"],
        expected_actions=["revoke_duplicate_coupons", "fix_coupon_state_machine"],
        expected_approval="team_lead",
        expected_level="L1",
        blast_radius=0.05,
        business_rule="br_coupon_abuse",
    ),
    "br_order_stuck": ScenarioSpec(
        scenario_id="br_order_stuck",
        source="business_rule",
        name="订单卡单检测",
        description="订单长时间停留在中间状态不流转",
        category="order",
        service="order-service",
        severity="high",
        dimensions=["e2e", "reasoning"],
        ground_truth={
            "root_cause": "order_state_machine_bug",
            "confidence": 0.8,
        },
        relevant_docs=["kb_biz_002"],
        expected_actions=["retry_stuck_orders", "check_order_processor_health"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.10,
        business_rule="br_order_stuck",
    ),
    "br_inventory_discrepancy": ScenarioSpec(
        scenario_id="br_inventory_discrepancy",
        source="business_rule",
        name="库存数据不一致",
        description="MySQL 与 Elasticsearch 中的库存数据不一致",
        category="inventory",
        service="inventory-service",
        severity="medium",
        dimensions=["e2e", "reasoning"],
        ground_truth={
            "root_cause": "replication_lag",
            "confidence": 0.78,
        },
        relevant_docs=["kb_biz_006"],
        expected_actions=["force_reindex", "check_replication_lag"],
        expected_approval="team_lead",
        expected_level="L1",
        blast_radius=0.12,
        business_rule="br_inventory_discrepancy",
    ),
    "br_user_permission_error": ScenarioSpec(
        scenario_id="br_user_permission_error",
        source="business_rule",
        name="用户权限异常",
        description="用户被拒绝访问应有权限的资源",
        category="user",
        service="user-service",
        severity="medium",
        dimensions=["e2e", "reasoning"],
        ground_truth={
            "root_cause": "permission_sync_failure",
            "confidence": 0.75,
        },
        relevant_docs=["kb_biz_003"],
        expected_actions=["sync_user_permissions", "check_permission_service"],
        expected_approval="oncall",
        expected_level="L1",
        blast_radius=0.05,
        business_rule="br_user_permission_error",
    ),
}


# ============================================================================
# 访问辅助函数
# ============================================================================


def get_scenario(scenario_id: str) -> ScenarioSpec | None:
    """根据 id 查找场景，未找到返回 None"""
    return SCENARIO_REGISTRY.get(scenario_id)


def scenarios_for_dimension(dimension: EvalDimension) -> list[ScenarioSpec]:
    """返回需要在指定维度上跑评测的所有场景"""
    return [
        spec for spec in SCENARIO_REGISTRY.values()
        if dimension in spec.dimensions
    ]


def scenarios_by_category(category: str) -> list[ScenarioSpec]:
    """按 category 过滤场景"""
    return [
        spec for spec in SCENARIO_REGISTRY.values()
        if spec.category == category
    ]


def scenarios_by_severity(severity: str) -> list[ScenarioSpec]:
    """按 severity 过滤场景"""
    return [
        spec for spec in SCENARIO_REGISTRY.values()
        if spec.severity == severity
    ]


def all_scenarios() -> list[ScenarioSpec]:
    """返回所有场景的列表形式"""
    return list(SCENARIO_REGISTRY.values())


def coverage_summary() -> dict[str, Any]:
    """评测维度覆盖率统计，用于前端/报告展示"""
    summary: dict[str, Any] = {
        "total": len(SCENARIO_REGISTRY),
        "by_category": {},
        "by_severity": {},
        "by_dimension": {d: 0 for d in EVAL_DIMENSIONS},
        "by_source": {},
    }
    for spec in SCENARIO_REGISTRY.values():
        summary["by_category"][spec.category] = (
            summary["by_category"].get(spec.category, 0) + 1
        )
        summary["by_severity"][spec.severity] = (
            summary["by_severity"].get(spec.severity, 0) + 1
        )
        summary["by_source"][spec.source] = (
            summary["by_source"].get(spec.source, 0) + 1
        )
        for d in spec.dimensions:
            summary["by_dimension"][d] += 1
    return summary
