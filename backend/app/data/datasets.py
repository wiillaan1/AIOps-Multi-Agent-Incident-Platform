"""
AIOps Agent Platform - Test Datasets

评估和测试用的数据集定义。
包含模拟指标数据、故障场景、变更记录等。
所有数据集支持动态扩展。
"""

from __future__ import annotations

from typing import Any

# ============================================================
# 模拟指标数据集
# ============================================================

METRICS_DATASETS: dict[str, dict[str, dict[str, list[float]]]] = {
    "order-service": {
        "cpu_usage_percent": {
            "normal": [45, 48, 52, 50, 47, 49, 51, 46, 48, 50],
            "anomaly": [45, 48, 52, 50, 47, 49, 51, 46, 48, 50, 55, 60, 75, 85, 92, 95, 93, 88, 80, 70],
            "spike": [45, 48, 52, 50, 47, 95, 93, 88],
        },
        "memory_usage_percent": {
            "normal": [60, 62, 61, 63, 60, 62, 61, 63, 60, 62],
            "leak": [60, 62, 65, 68, 72, 78, 85, 90, 92, 95],
        },
        "p99_latency_ms": {
            "normal": [100, 105, 98, 102, 100, 103, 99, 101, 100, 102],
            "degradation": [100, 105, 150, 200, 350, 500, 800, 1200, 1500, 2000],
        },
        "error_rate_percent": {
            "normal": [0.1, 0.2, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            "spike": [0.1, 0.2, 0.1, 0.3, 5.0, 8.0, 12.0, 15.0, 10.0, 6.0],
        },
    },
    "payment-service": {
        "cpu_usage_percent": {
            "normal": [40, 42, 41, 43, 40, 42, 41, 43, 40, 42],
            "anomaly": [40, 42, 41, 43, 50, 65, 80, 88, 85, 78],
        },
        "memory_usage_percent": {
            "normal": [55, 57, 56, 58, 55, 57, 56, 58, 55, 57],
            "leak": [55, 57, 60, 65, 72, 80, 88, 92, 94, 96],
        },
        "p99_latency_ms": {
            "normal": [80, 82, 78, 85, 80, 83, 79, 81, 80, 84],
            "degradation": [80, 82, 150, 300, 600, 1000, 1500, 2000, 2500, 3000],
        },
        "error_rate_percent": {
            "normal": [0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1],
            "spike": [0.05, 0.1, 0.05, 0.1, 2.0, 5.0, 8.0, 10.0, 6.0, 3.0],
        },
    },
    "user-service": {
        "cpu_usage_percent": {
            "normal": [35, 37, 36, 38, 35, 37, 36, 38, 35, 37],
            "anomaly": [35, 37, 36, 38, 45, 55, 70, 82, 78, 65],
        },
        "memory_usage_percent": {
            "normal": [50, 52, 51, 53, 50, 52, 51, 53, 50, 52],
            "leak": [50, 52, 55, 60, 68, 75, 82, 88, 90, 92],
        },
        "p99_latency_ms": {
            "normal": [50, 52, 48, 55, 50, 53, 49, 51, 50, 54],
            "degradation": [50, 52, 100, 250, 500, 800, 1200, 1800, 2200, 2800],
        },
        "error_rate_percent": {
            "normal": [0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15, 0.1, 0.15],
            "spike": [0.1, 0.15, 0.1, 0.15, 3.0, 6.0, 10.0, 14.0, 8.0, 4.0],
        },
    },
    "api-gateway": {
        "cpu_usage_percent": {
            "normal": [30, 32, 31, 33, 30, 32, 31, 33, 30, 32],
            "anomaly": [30, 32, 31, 33, 40, 50, 65, 75, 70, 60],
        },
        "memory_usage_percent": {
            "normal": [45, 47, 46, 48, 45, 47, 46, 48, 45, 47],
            "leak": [45, 47, 50, 55, 62, 70, 78, 85, 88, 90],
        },
        "p99_latency_ms": {
            "normal": [30, 32, 28, 35, 30, 33, 29, 31, 30, 34],
            "degradation": [30, 32, 80, 150, 300, 500, 800, 1200, 1500, 1800],
        },
        "error_rate_percent": {
            "normal": [0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1, 0.05, 0.1],
            "spike": [0.05, 0.1, 0.05, 0.1, 10.0, 20.0, 30.0, 25.0, 15.0, 8.0],
        },
    },
    "inventory-service": {
        "cpu_usage_percent": {
            "normal": [25, 27, 26, 28, 25, 27, 26, 28, 25, 27],
            "anomaly": [25, 27, 26, 28, 35, 45, 60, 70, 65, 55],
        },
        "memory_usage_percent": {
            "normal": [40, 42, 41, 43, 40, 42, 41, 43, 40, 42],
            "leak": [40, 42, 45, 50, 58, 68, 78, 85, 88, 90],
        },
        "p99_latency_ms": {
            "normal": [60, 62, 58, 65, 60, 63, 59, 61, 60, 64],
            "degradation": [60, 62, 120, 250, 400, 600, 900, 1300, 1600, 2000],
        },
        "error_rate_percent": {
            "normal": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
            "spike": [0.1, 0.2, 0.1, 0.2, 4.0, 8.0, 12.0, 10.0, 6.0, 3.0],
        },
    },
}

# ============================================================
# 故障场景数据集
# ============================================================

FAULT_SCENARIOS: list[dict[str, Any]] = [
    # ========== 基础设施层 ==========
    {
        "id": "fs_001",
        "name": "Deployment-induced CPU spike",
        "description": "Recent deployment causes CPU usage to spike",
        "service": "order-service",
        "category": "infrastructure",
        "metrics": {"cpu_usage_percent": "anomaly"},
        "root_cause": "recent_deployment",
        "expected_action": {"type": "rollback_deployment", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.15,
        "severity": "high",
    },
    {
        "id": "fs_002",
        "name": "Memory leak leading to OOM",
        "description": "Gradual memory leak eventually causes OOM kills",
        "service": "payment-service",
        "category": "infrastructure",
        "metrics": {"memory_usage_percent": "leak"},
        "root_cause": "memory_leak",
        "expected_action": {"type": "restart_pod", "level": "L0"},
        "expected_approval": "auto",
        "blast_radius": 0.05,
        "severity": "critical",
    },
    {
        "id": "fs_003",
        "name": "Database connection pool exhausted",
        "description": "DB connection pool depletion causing cascading failures",
        "service": "user-service",
        "category": "infrastructure",
        "metrics": {"p99_latency_ms": "degradation", "error_rate_percent": "spike"},
        "root_cause": "database_connection_pool_exhausted",
        "expected_action": {"type": "scale_up", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.2,
        "severity": "critical",
    },
    {
        "id": "fs_004",
        "name": "Cache failure causing DB overload",
        "description": "Redis cache failure leads to direct DB queries",
        "service": "inventory-service",
        "category": "infrastructure",
        "metrics": {"p99_latency_ms": "degradation"},
        "root_cause": "cache_failure",
        "expected_action": {"type": "restart_pod", "level": "L0"},
        "expected_approval": "auto",
        "blast_radius": 0.1,
        "severity": "high",
    },
    {
        "id": "fs_005",
        "name": "Configuration error causing high error rate",
        "description": "Wrong configuration causes services to return errors",
        "service": "api-gateway",
        "category": "infrastructure",
        "metrics": {"error_rate_percent": "spike"},
        "root_cause": "configuration_error",
        "expected_action": {"type": "rollback_config", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.25,
        "severity": "critical",
    },
    # ========== 业务层 ==========
    {
        "id": "fs_biz_001",
        "name": "Payment gateway timeout — orders piling up",
        "description": "Third-party payment gateway timeout causes payment failures and order backlog in order-service",
        "service": "payment-service",
        "category": "business",
        "metrics": {"payment_failure_rate": "spike", "order_backlog_count": "spike"},
        "root_cause": "third_party_timeout",
        "expected_action": {"type": "enable_payment_fallback", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.3,
        "severity": "critical",
    },
    {
        "id": "fs_biz_002",
        "name": "Order processing stuck — inventory sync failure",
        "description": "Inventory service returns stale data causing order-processor to deadlock on stock validation",
        "service": "order-service",
        "category": "business",
        "metrics": {"order_backlog_count": "spike"},
        "root_cause": "inventory_data_stale",
        "expected_action": {"type": "reconcile_inventory", "level": "L1"},
        "expected_approval": "team_lead",
        "blast_radius": 0.25,
        "severity": "critical",
    },
    {
        "id": "fs_biz_003",
        "name": "User login failure spike — SSO token cache expired",
        "description": "Redis cache expires all SSO tokens causing mass login failures across user-service",
        "service": "user-service",
        "category": "business",
        "metrics": {"auth_failure_rate": "spike"},
        "root_cause": "token_cache_expired",
        "expected_action": {"type": "enable_static_auth", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.4,
        "severity": "critical",
    },
    {
        "id": "fs_biz_004",
        "name": "Rate limiter false-positive — blocking legitimate traffic",
        "description": "API gateway rate limiter misconfigured, blocking valid user requests during promotion event",
        "service": "api-gateway",
        "category": "business",
        "metrics": {"rate_limit_hit_rate": "spike", "error_rate_percent": "spike"},
        "root_cause": "rate_limit_misconfig",
        "expected_action": {"type": "adjust_rate_limit", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.1,
        "severity": "high",
    },
    {
        "id": "fs_biz_005",
        "name": "Message queue backlog — consumer group stuck",
        "description": "Kafka consumer group stuck on corrupted message, causing 5000+ message lag on order-processing topic",
        "service": "order-service",
        "category": "business",
        "metrics": {"mq_consumer_lag": "spike"},
        "root_cause": "corrupted_message",
        "expected_action": {"type": "skip_corrupted_message", "level": "L0"},
        "expected_approval": "auto",
        "blast_radius": 0.12,
        "severity": "high",
    },
    {
        "id": "fs_biz_006",
        "name": "Data inconsistency — DB-ES replication lag",
        "description": "MySQL-to-Elasticsearch replication lag causes search results to show incorrect inventory counts",
        "service": "inventory-service",
        "category": "business",
        "metrics": {"replication_lag_seconds": "spike"},
        "root_cause": "replication_lag",
        "expected_action": {"type": "force_reindex", "level": "L1"},
        "expected_approval": "team_lead",
        "blast_radius": 0.15,
        "severity": "medium",
    },
    # ========== 纯业务异常（新增，不依赖指标） ==========
    {
        "id": "fs_biz_007",
        "name": "重复扣款 — 支付幂等性失败",
        "description": "支付回调因网络抖动重试3次，导致用户同一笔订单被扣款3次。依赖 payment_idempotency_key 失效",
        "service": "payment-service",
        "category": "business_logic",
        "business_rules": ["br_duplicate_charge"],
        "root_cause": "payment_idempotency_failure",
        "expected_action": {"type": "refund_duplicates", "level": "L2"},
        "expected_approval": "team_lead",
        "blast_radius": 0.05,
        "severity": "critical",
    },
    {
        "id": "fs_biz_008",
        "name": "库存超卖 — 并发下单竞态条件",
        "description": "秒杀活动期间，库存缓存未及时更新，10个用户同时下单成功但实际只剩3件库存",
        "service": "inventory-service",
        "category": "business_logic",
        "business_rules": ["br_oversell"],
        "root_cause": "inventory_sync_failure",
        "expected_action": {"type": "cancel_oversold_orders", "level": "L2"},
        "expected_approval": "team_lead",
        "blast_radius": 0.15,
        "severity": "critical",
    },
    {
        "id": "fs_biz_009",
        "name": "订单金额对账不平 — 优惠券计算错误",
        "description": "满减券在多SKU订单中比例分摊计算有精度问题，导致订单总额与实付金额相差0.02元",
        "service": "order-service",
        "category": "business_logic",
        "business_rules": ["br_amount_mismatch"],
        "root_cause": "accounting_reconciliation_error",
        "expected_action": {"type": "reconcile_transactions", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.08,
        "severity": "high",
    },
    {
        "id": "fs_biz_010",
        "name": "支付回调丢失 — 订单状态不一致",
        "description": "支付网关回调 webhook 因证书过期返回 500，导致 50 笔已支付订单仍显示待支付",
        "service": "payment-service",
        "category": "business_logic",
        "business_rules": ["br_payment_callback_loss"],
        "root_cause": "third_party_callback_loss",
        "expected_action": {"type": "reconcile_payment_status", "level": "L1"},
        "expected_approval": "oncall",
        "blast_radius": 0.1,
        "severity": "critical",
    },
]

# ============================================================
# 变更记录数据集
# ============================================================

CHANGE_RECORDS: list[dict[str, Any]] = [
    {
        "service": "order-service",
        "type": "deployment",
        "timestamp": "2024-01-15T10:30:00Z",
        "author": "developer-a",
        "description": "Update order processing logic",
    },
    {
        "service": "payment-service",
        "type": "config_change",
        "timestamp": "2024-01-15T11:00:00Z",
        "author": "developer-b",
        "description": "Increase timeout settings",
    },
    {
        "service": "user-service",
        "type": "deployment",
        "timestamp": "2024-01-15T09:00:00Z",
        "author": "developer-c",
        "description": "Add new authentication flow",
    },
    {
        "service": "api-gateway",
        "type": "config_change",
        "timestamp": "2024-01-15T08:30:00Z",
        "author": "developer-a",
        "description": "Update rate limiting rules",
    },
    {
        "service": "inventory-service",
        "type": "deployment",
        "timestamp": "2024-01-14T16:00:00Z",
        "author": "developer-d",
        "description": "Optimize query performance",
    },
]


# ============================================================
# 数据集访问辅助函数
# ============================================================

def get_metric_data(
    service: str,
    metric: str,
    scenario: str = "normal",
) -> list[float]:
    """
    获取指定服务的指标数据

    Args:
        service: 服务名称
        metric: 指标名称
        scenario: 场景类型（normal/anomaly/spike/leak/degradation）

    Returns:
        指标数据点列表
    """
    service_data = METRICS_DATASETS.get(service, {})
    metric_data = service_data.get(metric, {})

    # 尝试获取指定场景的数据
    data = metric_data.get(scenario)
    if data is not None:
        return data

    # 回退到 normal
    data = metric_data.get("normal")
    if data is not None:
        return data

    # 最终回退
    return [50.0] * 10


def get_fault_scenario(scenario_id: str) -> dict[str, Any] | None:
    """
    根据 ID 获取故障场景

    Args:
        scenario_id: 场景ID

    Returns:
        故障场景字典，未找到返回 None
    """
    for scenario in FAULT_SCENARIOS:
        if scenario["id"] == scenario_id:
            return scenario
    return None


def get_change_records(service: str = "") -> list[dict[str, Any]]:
    """
    获取变更记录

    Args:
        service: 服务名称过滤，空字符串返回全部

    Returns:
        变更记录列表
    """
    if not service:
        return CHANGE_RECORDS.copy()
    return [r for r in CHANGE_RECORDS if r["service"] == service]


def add_fault_scenario(scenario: dict[str, Any]) -> None:
    """
    动态添加故障场景

    Args:
        scenario: 故障场景字典，必须包含 id 字段
    """
    if "id" not in scenario:
        raise ValueError("Scenario must have an 'id' field")

    # 如果已存在则更新，否则添加
    for i, existing in enumerate(FAULT_SCENARIOS):
        if existing["id"] == scenario["id"]:
            FAULT_SCENARIOS[i] = scenario
            return
    FAULT_SCENARIOS.append(scenario)


def add_metric_data(
    service: str,
    metric: str,
    scenario: str,
    data: list[float],
) -> None:
    """
    动态添加指标数据

    Args:
        service: 服务名称
        metric: 指标名称
        scenario: 场景类型
        data: 指标数据点列表
    """
    if service not in METRICS_DATASETS:
        METRICS_DATASETS[service] = {}
    if metric not in METRICS_DATASETS[service]:
        METRICS_DATASETS[service][metric] = {}
    METRICS_DATASETS[service][metric][scenario] = data


def add_change_record(record: dict[str, Any]) -> None:
    """
    动态添加变更记录

    Args:
        record: 变更记录字典
    """
    CHANGE_RECORDS.append(record)


# 注：原先 `EvaluationDatasets` 类已删除（2026-07-19 重构）。
# 评测数据集现由 `app.evaluation.layered_datasets` 提供：
#   - UnitDatasets        — 单 skill 测试
#   - IntegrationDatasets — 多 skill 协同
#   - E2EDatasets         — 端到端场景（来自 SCENARIO_REGISTRY × FAULT_SCENARIOS）
#   - GoldenDatasets      — 回归基线
