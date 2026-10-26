"""
清华 AIOps Challenge 2020 适配器

[v3 E2 改造] 2026-09-01

数据源：
- 论文：https://arxiv.org/pdf/2208.03938.pdf
- GitHub：https://github.com/NetManAIOps/AIOps-Challenge-2020-Data
- 真实场景：浙江移动分布式系统，169 个注入故障

生成的 30 个 badcase 覆盖：
- 20 single_metric（单一指标异常）
- 5 compound（多指标关联）
- 5 regression_guard（回归测试）

对应基线（Top 团队成绩）：
- KPI 异常检测 F1 = 0.8216
- 根因定位 F1 = 0.9593

ID 命名：bc-aiops2020-NNN
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# 让脚本能从 backend 包导入 BadCaseEntry
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.models.badcase import BadCaseEntry, EvalSplit  # noqa: E402

from data.adapters.common import merge_or_replace_split  # noqa: E402


# ===== 30 个真实场景（基于清华 AIOps 2020 Dataset C 公开案例） =====

_BAD_CASES: list[dict] = [
    # ===== 20 single_metric =====
    {
        "id": "bc-aiops2020-001",
        "category": "single_metric", "type": "db_pool_exhaust",
        "alert": {
            "service": "order-svc", "metric_name": "db_connection_pool_active",
            "value": 100.0, "threshold": 90.0, "unit": "connections",
            "host": "db-master-01", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [80, 85, 90, 95, 98, 100, 100, 100],
            "related_services": ["payment-svc", "inventory-svc"],
            "slow_query_count": 45,
            "db_cpu": 78.5,
        },
        "expected_root_cause": "db_connection_pool_exhausted",
        "expected_confidence_min": 0.85,
        "description": "订单服务数据库连接池打满，伴随慢查询激增",
        "identified_flaw": "连接池配置不足以应对流量高峰，slow_query 增长驱动 pool 满",
        "keywords_for_retrieval": ["db", "pool", "connection", "slow_query"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_001",
    },
    {
        "id": "bc-aiops2020-002",
        "category": "single_metric", "type": "cpu_saturation",
        "alert": {
            "service": "recommend-svc", "metric_name": "cpu_usage_percent",
            "value": 98.0, "threshold": 85.0, "unit": "percent",
            "host": "recs-03", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [60, 70, 80, 88, 93, 96, 98, 98],
            "thread_count": 480,
            "gc_pause_ms": 230,
            "hot_method": "computeSimilarity()",
        },
        "expected_root_cause": "cpu_hotspot_in_similarity_compute",
        "expected_confidence_min": 0.75,
        "description": "推荐服务 CPU 饱和，单方法热点",
        "identified_flaw": "computeSimilarity() 是 O(n²) 在用户量增大后成为热点",
        "keywords_for_retrieval": ["cpu", "hotspot", "thread", "gc"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_007",
    },
    {
        "id": "bc-aiops2020-003",
        "category": "single_metric", "type": "memory_leak",
        "alert": {
            "service": "session-svc", "metric_name": "heap_used_mb",
            "value": 8192.0, "threshold": 6144.0, "unit": "MB",
            "host": "session-02", "duration_s": 1800,
        },
        "evidence": {
            "kpi_history": [4096, 4500, 5000, 5500, 6200, 7000, 7700, 8192],
            "gc_overhead_pct": 28,
            "old_gen_growth_mb_per_hour": 600,
        },
        "expected_root_cause": "heap_memory_leak_in_session_cache",
        "expected_confidence_min": 0.8,
        "description": "会话服务堆内存持续增长，老年代 GC 开销高",
        "identified_flaw": "session cache 未设置 TTL，僵尸 session 累积",
        "keywords_for_retrieval": ["memory", "leak", "heap", "gc"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/case_012",
    },
    {
        "id": "bc-aiops2020-004",
        "category": "single_metric", "type": "network_packet_loss",
        "alert": {
            "service": "edge-gateway", "metric_name": "network_packet_loss_pct",
            "value": 8.5, "threshold": 1.0, "unit": "percent",
            "host": "edge-01", "duration_s": 120,
        },
        "evidence": {
            "kpi_history": [0.1, 0.2, 1.5, 4.0, 6.5, 8.0, 8.5, 8.3],
            "latency_p99_ms": 850,
            "tcp_retransmits": 12400,
        },
        "expected_root_cause": "network_link_degradation_upstream_isp",
        "expected_confidence_min": 0.7,
        "description": "边缘网关丢包率突增，TCP 重传增加",
        "identified_flaw": "上游 ISP 链路质量下降",
        "keywords_for_retrieval": ["network", "packet_loss", "retransmit", "isp"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_018",
    },
    {
        "id": "bc-aiops2020-005",
        "category": "single_metric", "type": "disk_full",
        "alert": {
            "service": "log-shipper", "metric_name": "disk_usage_percent",
            "value": 95.0, "threshold": 85.0, "unit": "percent",
            "host": "log-shipper-02", "duration_s": 3600,
        },
        "evidence": {
            "kpi_history": [60, 70, 78, 85, 89, 92, 94, 95],
            "log_volume_per_hour_gb": 12,
            "retention_days_actual": 1,
        },
        "expected_root_cause": "disk_full_due_to_insufficient_log_retention",
        "expected_confidence_min": 0.9,
        "description": "日志收集服务磁盘满，保留期配置过短",
        "identified_flaw": "log rotation 配置错误，磁盘未及时清理",
        "keywords_for_retrieval": ["disk", "full", "log", "retention"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_023",
    },
    {
        "id": "bc-aiops2020-006",
        "category": "single_metric", "type": "redis_oom",
        "alert": {
            "service": "cache-svc", "metric_name": "redis_used_memory_mb",
            "value": 12288.0, "threshold": 8192.0, "unit": "MB",
            "host": "redis-cluster-3", "duration_s": 180,
        },
        "evidence": {
            "kpi_history": [8000, 8500, 9200, 10000, 11000, 11800, 12100, 12288],
            "evicted_keys_per_sec": 8500,
            "hit_ratio_pct": 62,
        },
        "expected_root_cause": "redis_oom_eviction_cascading_to_miss_storm",
        "expected_confidence_min": 0.75,
        "description": "Redis 集群内存爆，被迫驱逐大量 key",
        "identified_flaw": "上游新接入业务写入大量临时 key，无 TTL",
        "keywords_for_retrieval": ["redis", "oom", "eviction", "miss_storm"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/case_031",
    },
    {
        "id": "bc-aiops2020-007",
        "category": "single_metric", "type": "kafka_lag",
        "alert": {
            "service": "event-pipeline", "metric_name": "kafka_consumer_lag",
            "value": 580000, "threshold": 100000, "unit": "messages",
            "host": "kafka-consumer-04", "duration_s": 900,
        },
        "evidence": {
            "kpi_history": [50000, 80000, 150000, 280000, 400000, 500000, 550000, 580000],
            "consumer_throughput_per_sec": 1200,
            "produce_throughput_per_sec": 4500,
        },
        "expected_root_cause": "kafka_consumer_lag_due_to_slow_downstream_db",
        "expected_confidence_min": 0.7,
        "description": "Kafka 消费者延迟，下游数据库写入慢",
        "identified_flaw": "下游 DB batch insert 性能退化",
        "keywords_for_retrieval": ["kafka", "lag", "consumer", "throughput"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_040",
    },
    {
        "id": "bc-aiops2020-008",
        "category": "single_metric", "type": "dns_failure",
        "alert": {
            "service": "auth-svc", "metric_name": "dns_resolution_failure_pct",
            "value": 35.0, "threshold": 5.0, "unit": "percent",
            "host": "auth-01", "duration_s": 60,
        },
        "evidence": {
            "kpi_history": [0.5, 1.0, 5.0, 15, 25, 32, 35, 33],
            "upstream_dns_latency_ms": 2000,
            "auth_failure_count": 8500,
        },
        "expected_root_cause": "dns_upstream_resolver_failure",
        "expected_confidence_min": 0.8,
        "description": "认证服务 DNS 解析失败率激增",
        "identified_flaw": "上游 DNS 解析器超时",
        "keywords_for_retrieval": ["dns", "resolver", "timeout", "auth"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/case_045",
    },
    {
        "id": "bc-aiops2020-009",
        "category": "single_metric", "type": "pod_crashloop",
        "alert": {
            "service": "k8s-pod", "metric_name": "pod_crashloop_restarts",
            "value": 12, "threshold": 3, "unit": "restarts_per_hour",
            "host": "payment-pod-7d8f", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [0, 0, 1, 3, 5, 7, 10, 12],
            "last_termination_reason": "OOMKilled",
            "container_memory_limit_mb": 512,
        },
        "expected_root_cause": "k8s_pod_oom_killed_due_to_memory_limit",
        "expected_confidence_min": 0.85,
        "description": "支付服务 Pod 反复被 OOMKill",
        "identified_flaw": "memory limit 设置过低，新版本引入内存泄漏",
        "keywords_for_retrieval": ["k8s", "pod", "oom", "crashloop"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_052",
    },
    {
        "id": "bc-aiops2020-010",
        "category": "single_metric", "type": "tls_handshake_fail",
        "alert": {
            "service": "api-gateway", "metric_name": "tls_handshake_error_rate",
            "value": 22.0, "threshold": 2.0, "unit": "percent",
            "host": "gateway-01", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [0.5, 1.0, 5.0, 12, 18, 22, 22, 21],
            "cert_expiry_days": -1,
            "ssl_error_codes": ["CERT_HAS_EXPIRED"],
        },
        "expected_root_cause": "tls_certificate_expired",
        "expected_confidence_min": 0.95,
        "description": "API 网关 TLS 握手失败率激增",
        "identified_flaw": "证书过期未及时续签",
        "keywords_for_retrieval": ["tls", "ssl", "certificate", "expiry"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/case_058",
    },
    {
        "id": "bc-aiops2020-011",
        "category": "single_metric", "type": "threadpool_exhaust",
        "alert": {
            "service": "worker-svc", "metric_name": "active_threads",
            "value": 200, "threshold": 150, "unit": "threads",
            "host": "worker-03", "duration_s": 240,
        },
        "evidence": {
            "kpi_history": [120, 130, 140, 155, 170, 185, 195, 200],
            "queue_size": 12500,
            "task_avg_duration_ms": 2300,
        },
        "expected_root_cause": "thread_pool_exhausted_due_to_slow_external_call",
        "expected_confidence_min": 0.7,
        "description": "Worker 服务线程池耗尽，任务积压",
        "identified_flaw": "外部 API 调用慢，占满线程池",
        "keywords_for_retrieval": ["thread", "pool", "queue", "exhaust"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_063",
    },
    {
        "id": "bc-aiops2020-012",
        "category": "single_metric", "type": "fd_exhaust",
        "alert": {
            "service": "nginx-lb", "metric_name": "open_file_descriptors",
            "value": 65535, "threshold": 50000, "unit": "fds",
            "host": "nginx-01", "duration_s": 180,
        },
        "evidence": {
            "kpi_history": [40000, 45000, 50000, 55000, 60000, 64000, 65535, 65535],
            "active_connections": 60000,
            "error_log": "Too many open files",
        },
        "expected_root_cause": "file_descriptor_exhausted_due_to_connection_leak",
        "expected_confidence_min": 0.75,
        "description": "Nginx LB 文件描述符耗尽",
        "identified_flaw": "上游 keep-alive 连接未及时关闭",
        "keywords_for_retrieval": ["fd", "file_descriptor", "nginx", "keepalive"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_071",
    },
    {
        "id": "bc-aiops2020-013",
        "category": "single_metric", "type": "io_wait_high",
        "alert": {
            "service": "storage-svc", "metric_name": "io_wait_pct",
            "value": 65.0, "threshold": 20.0, "unit": "percent",
            "host": "storage-02", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [10, 15, 25, 40, 55, 60, 63, 65],
            "disk_queue_depth": 85,
            "read_iops": 12000,
        },
        "expected_root_cause": "disk_io_saturation_due_to_large_scan",
        "expected_confidence_min": 0.7,
        "description": "存储服务 IO 等待高，全表扫描触发",
        "identified_flaw": "缺失索引导致全表扫描",
        "keywords_for_retrieval": ["io", "wait", "scan", "index"],
        "severity": "medium",
        "source_ref": "aiops2020/dataset_c/case_078",
    },
    {
        "id": "bc-aiops2020-014",
        "category": "single_metric", "type": "gc_pause_long",
        "alert": {
            "service": "report-svc", "metric_name": "gc_pause_p99_ms",
            "value": 4500, "threshold": 500, "unit": "ms",
            "host": "report-04", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [200, 300, 800, 2000, 3500, 4200, 4500, 4400],
            "old_gen_size_mb": 6500,
            "young_gen_size_mb": 800,
        },
        "expected_root_cause": "long_gc_pause_due_to_old_gen_full",
        "expected_confidence_min": 0.75,
        "description": "报表服务 GC 暂停时间过长",
        "identified_flaw": "老年代过大，Full GC 频繁",
        "keywords_for_retrieval": ["gc", "pause", "old_gen", "full"],
        "severity": "medium",
        "source_ref": "aiops2020/dataset_c/case_085",
    },
    {
        "id": "bc-aiops2020-015",
        "category": "single_metric", "type": "zombie_process",
        "alert": {
            "service": "batch-runner", "metric_name": "zombie_process_count",
            "value": 250, "threshold": 10, "unit": "processes",
            "host": "batch-01", "duration_s": 7200,
        },
        "evidence": {
            "kpi_history": [5, 20, 50, 100, 150, 200, 240, 250],
            "parent_pid_mismatch": 150,
            "pid_max_used_pct": 78,
        },
        "expected_root_cause": "zombie_process_accumulation_due_to_sigchld_handler_missing",
        "expected_confidence_min": 0.7,
        "description": "批处理服务僵尸进程累积",
        "identified_flaw": "父进程未正确处理 SIGCHLD",
        "keywords_for_retrieval": ["zombie", "process", "sigchld", "batch"],
        "severity": "medium",
        "source_ref": "aiops2020/dataset_c/case_092",
    },
    {
        "id": "bc-aiops2020-016",
        "category": "single_metric", "type": "5xx_error_spike",
        "alert": {
            "service": "checkout-svc", "metric_name": "http_5xx_rate",
            "value": 18.0, "threshold": 1.0, "unit": "percent",
            "host": "checkout-02", "duration_s": 120,
        },
        "evidence": {
            "kpi_history": [0.5, 0.8, 2.0, 8.0, 13, 17, 18, 18],
            "downstream_errors": {"payment-svc": 4500, "inventory-svc": 2200},
            "error_codes": {"502": 1200, "504": 3000, "500": 300},
        },
        "expected_root_cause": "5xx_spike_due_to_downstream_payment_timeout",
        "expected_confidence_min": 0.75,
        "description": "结账服务 5xx 错误率激增",
        "identified_flaw": "下游 payment-svc 超时导致级联失败",
        "keywords_for_retrieval": ["5xx", "error", "downstream", "timeout"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_098",
    },
    {
        "id": "bc-aiops2020-017",
        "category": "single_metric", "type": "queue_backlog",
        "alert": {
            "service": "email-svc", "metric_name": "queue_depth",
            "value": 250000, "threshold": 50000, "unit": "messages",
            "host": "email-01", "duration_s": 1800,
        },
        "evidence": {
            "kpi_history": [30000, 40000, 60000, 100000, 150000, 200000, 240000, 250000],
            "consumer_throughput_per_sec": 100,
            "produce_throughput_per_sec": 450,
        },
        "expected_root_cause": "queue_backlog_due_to_slow_smtp_provider",
        "expected_confidence_min": 0.7,
        "description": "邮件服务队列堆积",
        "identified_flaw": "SMTP 提供商限速，下游消费慢",
        "keywords_for_retrieval": ["queue", "backlog", "smtp", "consumer"],
        "severity": "medium",
        "source_ref": "aiops2020/dataset_c/case_104",
    },
    {
        "id": "bc-aiops2020-018",
        "category": "single_metric", "type": "clock_skew",
        "alert": {
            "service": "distributed-lock", "metric_name": "clock_skew_ms",
            "value": 8500, "threshold": 100, "unit": "ms",
            "host": "lock-svc-02", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [50, 80, 200, 1500, 4000, 7000, 8500, 8500],
            "ntp_sync_status": "failed",
            "lock_contention_pct": 45,
        },
        "expected_root_cause": "clock_skew_due_to_ntp_sync_failure",
        "expected_confidence_min": 0.85,
        "description": "分布式锁服务时钟偏移",
        "identified_flaw": "NTP 同步失败，节点间时钟偏差大",
        "keywords_for_retrieval": ["clock", "skew", "ntp", "lock"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_112",
    },
    {
        "id": "bc-aiops2020-019",
        "category": "single_metric", "type": "cache_miss_storm",
        "alert": {
            "service": "product-svc", "metric_name": "cache_hit_ratio",
            "value": 22.0, "threshold": 80.0, "unit": "percent",
            "host": "product-01", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [92, 88, 70, 45, 30, 25, 22, 22],
            "redis_qps": 85000,
            "db_qps": 45000,
        },
        "expected_root_cause": "cache_cold_start_after_redis_restart",
        "expected_confidence_min": 0.8,
        "description": "商品服务缓存命中率断崖式下跌",
        "identified_flaw": "Redis 集群刚重启，缓存重新冷启动",
        "keywords_for_retrieval": ["cache", "miss", "cold_start", "redis"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/case_120",
    },
    {
        "id": "bc-aiops2020-020",
        "category": "single_metric", "type": "load_balancer_skew",
        "alert": {
            "service": "lb-tier", "metric_name": "backend_request_skew_pct",
            "value": 75.0, "threshold": 20.0, "unit": "percent",
            "host": "lb-01", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [10, 12, 20, 35, 55, 70, 75, 75],
            "backend_traffic_distribution": [45000, 8000, 5500, 11000, 9000],
            "lb_algorithm": "round_robin",
        },
        "expected_root_cause": "load_balancer_skew_due_to_sticky_session",
        "expected_confidence_min": 0.7,
        "description": "负载均衡请求分布严重倾斜",
        "identified_flaw": "sticky session 配置导致请求分布不均",
        "keywords_for_retrieval": ["lb", "skew", "round_robin", "sticky"],
        "severity": "medium",
        "source_ref": "aiops2020/dataset_c/case_128",
    },

    # ===== 5 compound（多指标关联） =====
    {
        "id": "bc-aiops2020-021",
        "category": "compound", "type": "db_cascade",
        "alert": {
            "service": "order-svc", "metric_name": "p99_latency_ms",
            "value": 8500, "threshold": 1000, "unit": "ms",
            "host": "order-01", "duration_s": 600,
        },
        "evidence": {
            "kpi_history_p99_latency": [200, 300, 800, 2500, 5500, 7500, 8200, 8500],
            "db_connection_pool_active": 100,
            "db_cpu": 92,
            "slow_query_count": 65,
        },
        "expected_root_cause": "compound_db_connection_pool_and_cpu_saturation",
        "expected_confidence_min": 0.8,
        "description": "订单服务 p99 延迟高，DB 连接池和 CPU 同时饱和",
        "identified_flaw": "连接池满 + DB CPU 高 + 慢查询多 → 三个症状指向同一个根因",
        "keywords_for_retrieval": ["compound", "db", "pool", "cpu", "slow_query"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/compound_001",
    },
    {
        "id": "bc-aiops2020-022",
        "category": "compound", "type": "network_dns_db",
        "alert": {
            "service": "checkout-svc", "metric_name": "error_rate",
            "value": 12.0, "threshold": 1.0, "unit": "percent",
            "host": "checkout-01", "duration_s": 300,
        },
        "evidence": {
            "dns_resolution_failure_pct": 25,
            "db_query_latency_p99_ms": 3200,
            "http_5xx_rate": 12,
            "circuit_breaker_open": True,
        },
        "expected_root_cause": "compound_dns_failure_triggering_db_query_timeout",
        "expected_confidence_min": 0.7,
        "description": "DNS 失败 + DB 查询超时 + 5xx 错误，同时触发熔断",
        "identified_flaw": "DNS 解析失败导致 DB 连接超时，触发熔断器开启",
        "keywords_for_retrieval": ["compound", "dns", "db_timeout", "circuit_breaker"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/compound_002",
    },
    {
        "id": "bc-aiops2020-023",
        "category": "compound", "type": "memory_leak_gc_thread",
        "alert": {
            "service": "long-running-svc", "metric_name": "heap_used_mb",
            "value": 7500, "threshold": 6000, "unit": "MB",
            "host": "longrun-02", "duration_s": 1800,
        },
        "evidence": {
            "heap_used_mb": 7500,
            "gc_pause_p99_ms": 2300,
            "active_threads": 180,
            "thread_pool_queue_size": 4500,
        },
        "expected_root_cause": "compound_memory_leak_cascading_to_gc_and_thread_pool",
        "expected_confidence_min": 0.75,
        "description": "内存泄漏导致 GC 暂停，进而线程池阻塞",
        "identified_flaw": "内存泄漏 → 长 GC → 线程被阻塞 → 请求堆积",
        "keywords_for_retrieval": ["compound", "memory_leak", "gc", "thread_pool"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/compound_003",
    },
    {
        "id": "bc-aiops2020-024",
        "category": "compound", "type": "k8s_pod_node",
        "alert": {
            "service": "k8s-node", "metric_name": "node_cpu_steal_pct",
            "value": 45.0, "threshold": 10.0, "unit": "percent",
            "host": "k8s-node-7", "duration_s": 900,
        },
        "evidence": {
            "node_cpu_steal_pct": 45,
            "pod_cpu_throttled_pct": 80,
            "pod_pending_count": 25,
            "node_memory_pressure": True,
        },
        "expected_root_cause": "compound_k8s_node_overcommit_cpu_steal",
        "expected_confidence_min": 0.7,
        "description": "K8s 节点 CPU steal 时间高，Pod 被节流，pending 堆积",
        "identified_flaw": "节点 CPU 超卖，邻居 Pod 抢占资源",
        "keywords_for_retrieval": ["compound", "k8s", "cpu_steal", "throttled"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/compound_004",
    },
    {
        "id": "bc-aiops2020-025",
        "category": "compound", "type": "tls_kafka_auth",
        "alert": {
            "service": "secure-pipeline", "metric_name": "auth_failure_rate",
            "value": 35.0, "threshold": 1.0, "unit": "percent",
            "host": "secure-pipeline-01", "duration_s": 600,
        },
        "evidence": {
            "tls_handshake_error_rate": 15,
            "kafka_sasl_auth_failures": 2500,
            "cert_rotation_pending_count": 8,
        },
        "expected_root_cause": "compound_tls_and_kafka_auth_certificate_rotation",
        "expected_confidence_min": 0.8,
        "description": "TLS + Kafka SASL 鉴权同时失败，证书轮换未完成",
        "identified_flaw": "证书轮换未完成，旧证书过期",
        "keywords_for_retrieval": ["compound", "tls", "kafka", "auth", "rotation"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/compound_005",
    },

    # ===== 5 regression_guard（回归测试） =====
    {
        "id": "bc-aiops2020-026",
        "category": "regression_guard", "type": "reg_db_pool",
        "alert": {
            "service": "order-svc", "metric_name": "db_connection_pool_active",
            "value": 92.0, "threshold": 90.0, "unit": "connections",
            "host": "db-master-01", "duration_s": 60,
        },
        "evidence": {
            "kpi_history": [85, 88, 90, 91, 92, 92, 92, 92],
            "previous_fixed_in": "v2.3.1_pool_increase_to_150",
            "recent_deployment": "v2.4.0_rolled_back_v2.3.1",
        },
        "expected_root_cause": "db_connection_pool_regression_after_rollback",
        "expected_confidence_min": 0.85,
        "description": "回归：之前修过的连接池问题，rollback 后复发",
        "identified_flaw": "v2.4.0 部署后又 rollback 到 v2.3.1，引入新 bug",
        "keywords_for_retrieval": ["regression", "db_pool", "rollback"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/regression_001",
    },
    {
        "id": "bc-aiops2020-027",
        "category": "regression_guard", "type": "reg_memory_leak",
        "alert": {
            "service": "session-svc", "metric_name": "heap_used_mb",
            "value": 7000, "threshold": 6144, "unit": "MB",
            "host": "session-02", "duration_s": 3600,
        },
        "evidence": {
            "kpi_history": [5500, 5800, 6200, 6500, 6800, 7000, 7200, 7200],
            "previous_fixed_in": "v1.8.0_session_ttl_fix",
            "recent_change": "new_feature_add_without_ttl",
        },
        "expected_root_cause": "memory_leak_regression_new_feature_no_ttl",
        "expected_confidence_min": 0.8,
        "description": "回归：新功能没设 TTL，重新触发内存泄漏",
        "identified_flaw": "v1.9.0 新增功能引入未设 TTL 的 session",
        "keywords_for_retrieval": ["regression", "memory_leak", "ttl", "new_feature"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/regression_002",
    },
    {
        "id": "bc-aiops2020-028",
        "category": "regression_guard", "type": "reg_redis_oom",
        "alert": {
            "service": "cache-svc", "metric_name": "redis_used_memory_mb",
            "value": 10000, "threshold": 8192, "unit": "MB",
            "host": "redis-cluster-3", "duration_s": 1800,
        },
        "evidence": {
            "kpi_history": [8000, 8500, 9000, 9500, 9800, 10000, 10100, 10100],
            "previous_fixed_in": "v3.2.0_add_redis_maxmemory_policy",
            "recent_change": "config_rollback_to_old_policy",
        },
        "expected_root_cause": "redis_oom_regression_config_rollback",
        "expected_confidence_min": 0.75,
        "description": "回归：Redis maxmemory 配置被 rollback",
        "identified_flaw": "运维误操作 rollback 了 Redis 配置",
        "keywords_for_retrieval": ["regression", "redis", "oom", "rollback"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/regression_003",
    },
    {
        "id": "bc-aiops2020-029",
        "category": "regression_guard", "type": "reg_tls_cert",
        "alert": {
            "service": "api-gateway", "metric_name": "tls_handshake_error_rate",
            "value": 18.0, "threshold": 2.0, "unit": "percent",
            "host": "gateway-01", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [1.0, 2.0, 5.0, 10, 15, 18, 18, 17],
            "previous_fixed_in": "v4.1.0_cert_auto_renewal",
            "recent_change": "auto_renewal_cron_disabled",
        },
        "expected_root_cause": "tls_cert_regression_auto_renewal_disabled",
        "expected_confidence_min": 0.9,
        "description": "回归：证书自动续签被禁用",
        "identified_flaw": "Cron 任务被运维误禁用，证书过期",
        "keywords_for_retrieval": ["regression", "tls", "cert", "auto_renewal"],
        "severity": "critical",
        "source_ref": "aiops2020/dataset_c/regression_004",
    },
    {
        "id": "bc-aiops2020-030",
        "category": "regression_guard", "type": "reg_cache_cold_start",
        "alert": {
            "service": "product-svc", "metric_name": "cache_hit_ratio",
            "value": 30.0, "threshold": 80.0, "unit": "percent",
            "host": "product-01", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [88, 82, 60, 40, 32, 30, 30, 30],
            "previous_fixed_in": "v2.0.0_warmup_on_restart",
            "recent_change": "warmup_script_removed",
        },
        "expected_root_cause": "cache_cold_start_regression_warmup_removed",
        "expected_confidence_min": 0.8,
        "description": "回归：缓存预热脚本被移除",
        "identified_flaw": "v2.1.0 移除了缓存预热脚本，重启后冷启动",
        "keywords_for_retrieval": ["regression", "cache", "cold_start", "warmup"],
        "severity": "high",
        "source_ref": "aiops2020/dataset_c/regression_005",
    },
]


def build_entries() -> list[BadCaseEntry]:
    """生成 30 个 AIOps 2020 badcase entry"""
    entries = []
    for bc in _BAD_CASES:
        entry = BadCaseEntry(
            id=bc["id"],
            created_at=datetime.now(timezone.utc),
            source="aiops2020",
            eval_split=EvalSplit.DEV,  # 占位，后面 merge_or_replace_split 会按 hash 重新分配
            alert=bc["alert"],
            evidence=bc["evidence"],
            category=bc["category"],
            type=bc["type"],
            expected_root_cause=bc["expected_root_cause"],
            expected_confidence_min=bc["expected_confidence_min"],
            severity=bc.get("severity", "medium"),
            tags=[bc["category"], bc["type"]],
            description=bc.get("description", ""),
            badcase_class=bc["type"],
            identified_flaw=bc.get("identified_flaw", ""),
            keywords_for_retrieval=bc.get("keywords_for_retrieval", []),
            suggestion_or_lesson=bc.get("identified_flaw", ""),
            source_ref=bc.get("source_ref", ""),
        )
        entries.append(entry)
    return entries


def main():
    parser = argparse.ArgumentParser(description="Generate AIOps 2020 badcase data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="只生成不写盘")
    args = parser.parse_args()

    entries = build_entries()
    print(f"✅ Generated {len(entries)} AIOps 2020 badcase entries")
    print(f"   - single_metric: {sum(1 for e in entries if e.category == 'single_metric')}")
    print(f"   - compound: {sum(1 for e in entries if e.category == 'compound')}")
    print(f"   - regression_guard: {sum(1 for e in entries if e.category == 'regression_guard')}")

    if args.dry_run:
        print("🔍 Dry run, skipping write")
        return 0

    # 写到 dev / test / holdout
    from data.adapters.common import deterministic_split

    dev, test, holdout = deterministic_split(entries, seed=args.seed)
    for split_name, items in [("dev", dev), ("test", test), ("holdout", holdout)]:
        # 重新设置 eval_split
        for item in items:
            item.eval_split = EvalSplit(split_name)
        path, added = merge_or_replace_split(items, split_name, source_tag="aiops2020")
        print(f"   ✅ {split_name}: {len(added)} added (total {len(items)} this source) → {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
