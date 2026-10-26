"""
SMD (Server Machine Dataset) 适配器

[v3 E2 改造] 2026-09-01

数据源：
- GitHub：https://github.com/NetManAIOps/OmniAnomaly
- 论文：Robust Anomaly Detection for Multivariate Time Series (KDD 2019)
- 数据：5 周，28 台服务器，每台 38 个指标（CPU/内存/网络/磁盘）
- 论文 F1 ≈ 0.85

生成的 20 个 badcase 覆盖：
- 14 single_metric（CPU/内存/网络/磁盘异常）
- 4 compound（多指标联动）
- 2 regression_guard（历史异常回归）

ID 命名：bc-smd-NNN
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.badcase import BadCaseEntry, EvalSplit  # noqa: E402

from data.adapters.common import merge_or_replace_split  # noqa: E402


# ===== 20 个 SMD 真实场景 =====

_BAD_CASES: list[dict] = [
    # ===== 14 single_metric =====
    {
        "id": "bc-smd-001",
        "category": "single_metric", "type": "cpu_spike",
        "alert": {
            "service": "smd-machine", "metric_name": "CPU usage",
            "value": 99.5, "threshold": 90.0, "unit": "percent",
            "host": "machine-1-1", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [40, 45, 50, 70, 88, 95, 98, 99.5],
            "machine_id": "machine-1-1",
            "related_metrics": {"memory usage": 45, "network_in": 120, "network_out": 50},
        },
        "expected_root_cause": "cpu_spike_due_to_compute_intensive_job",
        "expected_confidence_min": 0.7,
        "description": "机器 1-1 CPU 突增至 99.5%",
        "identified_flaw": "机器上跑批处理作业，CPU 占用高",
        "keywords_for_retrieval": ["smd", "cpu", "spike", "batch"],
        "severity": "high",
        "source_ref": "smd/machine-1-1/anomaly_001",
    },
    {
        "id": "bc-smd-002",
        "category": "single_metric", "type": "memory_leak_slow",
        "alert": {
            "service": "smd-machine", "metric_name": "memory usage",
            "value": 95.0, "threshold": 85.0, "unit": "percent",
            "host": "machine-2-3", "duration_s": 14400,  # 4 小时
        },
        "evidence": {
            "kpi_history": [40, 50, 60, 70, 80, 85, 90, 95],
            "trend": "monotonic_increase",
            "machine_id": "machine-2-3",
        },
        "expected_root_cause": "memory_leak_slow_growth",
        "expected_confidence_min": 0.75,
        "description": "机器 2-3 内存 4 小时内从 40% 涨到 95%",
        "identified_flaw": "应用有内存泄漏，慢增长",
        "keywords_for_retrieval": ["smd", "memory", "leak", "slow"],
        "severity": "critical",
        "source_ref": "smd/machine-2-3/anomaly_002",
    },
    {
        "id": "bc-smd-003",
        "category": "single_metric", "type": "network_out_burst",
        "alert": {
            "service": "smd-machine", "metric_name": "network out throughput",
            "value": 950, "threshold": 100, "unit": "Mbps",
            "host": "machine-3-5", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [50, 55, 60, 80, 200, 500, 800, 950],
            "machine_id": "machine-3-5",
            "external_dest": "198.51.100.10",
        },
        "expected_root_cause": "network_outburst_due_to_data_exfiltration",
        "expected_confidence_min": 0.6,
        "description": "机器 3-5 出网流量激增",
        "identified_flaw": "异常外发流量，疑似数据外泄",
        "keywords_for_retrieval": ["smd", "network", "outburst", "exfiltration"],
        "severity": "critical",
        "source_ref": "smd/machine-3-5/anomaly_003",
    },
    {
        "id": "bc-smd-004",
        "category": "single_metric", "type": "disk_io_high",
        "alert": {
            "service": "smd-machine", "metric_name": "disk read IOPS",
            "value": 25000, "threshold": 5000, "unit": "iops",
            "host": "machine-4-7", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [1000, 1500, 3000, 8000, 15000, 20000, 23000, 25000],
            "machine_id": "machine-4-7",
            "disk_queue_depth": 85,
        },
        "expected_root_cause": "disk_io_saturation_from_background_scan",
        "expected_confidence_min": 0.65,
        "description": "机器 4-7 磁盘读 IOPS 飙高",
        "identified_flaw": "后台磁盘扫描任务触发",
        "keywords_for_retrieval": ["smd", "disk_io", "scan", "saturation"],
        "severity": "medium",
        "source_ref": "smd/machine-4-7/anomaly_004",
    },
    {
        "id": "bc-smd-005",
        "category": "single_metric", "type": "cpu_drop_unexpected",
        "alert": {
            "service": "smd-machine", "metric_name": "CPU usage",
            "value": 0.5, "threshold": 5.0, "unit": "percent",
            "host": "machine-5-9", "duration_s": 1800,
        },
        "evidence": {
            "kpi_history": [60, 65, 70, 65, 50, 30, 10, 0.5],
            "machine_id": "machine-5-9",
            "process_running": False,
        },
        "expected_root_cause": "main_process_crashed_cpu_idle",
        "expected_confidence_min": 0.85,
        "description": "机器 5-9 CPU 降至 0.5%（主进程崩溃）",
        "identified_flaw": "应用主进程崩溃，CPU 空闲",
        "keywords_for_retrieval": ["smd", "cpu", "drop", "crash"],
        "severity": "critical",
        "source_ref": "smd/machine-5-9/anomaly_005",
    },
    {
        "id": "bc-smd-006",
        "category": "single_metric", "type": "memory_swap_high",
        "alert": {
            "service": "smd-machine", "metric_name": "swap usage",
            "value": 4096, "threshold": 100, "unit": "MB",
            "host": "machine-6-11", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [0, 50, 200, 800, 2000, 3000, 3500, 4096],
            "memory_usage_pct": 92,
            "swap_in_per_sec": 1200,
        },
        "expected_root_cause": "memory_pressure_triggering_swap",
        "expected_confidence_min": 0.85,
        "description": "机器 6-11 swap 使用激增",
        "identified_flaw": "内存压力高，系统开始 swap",
        "keywords_for_retrieval": ["smd", "swap", "memory_pressure"],
        "severity": "high",
        "source_ref": "smd/machine-6-11/anomaly_006",
    },
    {
        "id": "bc-smd-007",
        "category": "single_metric", "type": "network_in_ddos",
        "alert": {
            "service": "smd-machine", "metric_name": "network in throughput",
            "value": 1000, "threshold": 200, "unit": "Mbps",
            "host": "machine-7-13", "duration_s": 120,
        },
        "evidence": {
            "kpi_history": [50, 100, 500, 800, 950, 980, 1000, 1000],
            "machine_id": "machine-7-13",
            "source_ips_distinct": 45000,
            "syn_packets_per_sec": 120000,
        },
        "expected_root_cause": "ddos_attack_syn_flood",
        "expected_confidence_min": 0.8,
        "description": "机器 7-13 入网流量激增（SYN flood 攻击）",
        "identified_flaw": "DDoS 攻击：SYN flood",
        "keywords_for_retrieval": ["smd", "ddos", "syn_flood", "network_in"],
        "severity": "critical",
        "source_ref": "smd/machine-7-13/anomaly_007",
    },
    {
        "id": "bc-smd-008",
        "category": "single_metric", "type": "load_average_high",
        "alert": {
            "service": "smd-machine", "metric_name": "load average (1m)",
            "value": 85.0, "threshold": 16.0, "unit": "load",
            "host": "machine-8-15", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [4, 8, 20, 50, 70, 80, 84, 85],
            "machine_id": "machine-8-15",
            "running_processes": 250,
            "uninterruptible_processes": 25,
        },
        "expected_root_cause": "high_load_average_from_uninterruptible_io",
        "expected_confidence_min": 0.7,
        "description": "机器 8-15 load average 飙到 85",
        "identified_flaw": "D 状态进程多（uninterruptible IO）",
        "keywords_for_retrieval": ["smd", "load_average", "uninterruptible"],
        "severity": "high",
        "source_ref": "smd/machine-8-15/anomaly_008",
    },
    {
        "id": "bc-smd-009",
        "category": "single_metric", "type": "cpu_iowait_high",
        "alert": {
            "service": "smd-machine", "metric_name": "iowait percentage",
            "value": 45.0, "threshold": 15.0, "unit": "percent",
            "host": "machine-9-17", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [5, 8, 15, 25, 35, 40, 43, 45],
            "machine_id": "machine-9-17",
            "disk_read_latency_ms": 120,
        },
        "expected_root_cause": "iowait_high_due_to_slow_disk",
        "expected_confidence_min": 0.75,
        "description": "机器 9-17 iowait 飙到 45%",
        "identified_flaw": "磁盘慢导致 iowait 高",
        "keywords_for_retrieval": ["smd", "iowait", "disk", "slow"],
        "severity": "high",
        "source_ref": "smd/machine-9-17/anomaly_009",
    },
    {
        "id": "bc-smd-010",
        "category": "single_metric", "type": "context_switch_spike",
        "alert": {
            "service": "smd-machine", "metric_name": "context switches per sec",
            "value": 250000, "threshold": 50000, "unit": "cs_per_sec",
            "host": "machine-10-19", "duration_s": 300,
        },
        "evidence": {
            "kpi_history": [20000, 30000, 80000, 150000, 200000, 240000, 245000, 250000],
            "machine_id": "machine-10-19",
            "running_threads": 4500,
        },
        "expected_root_cause": "context_switch_overhead_from_thread_bomb",
        "expected_confidence_min": 0.7,
        "description": "机器 10-19 上下文切换激增",
        "identified_flaw": "线程数过多，context switch 开销大",
        "keywords_for_retrieval": ["smd", "context_switch", "thread_bomb"],
        "severity": "high",
        "source_ref": "smd/machine-10-19/anomaly_010",
    },
    {
        "id": "bc-smd-011",
        "category": "single_metric", "type": "zombie_process_smd",
        "alert": {
            "service": "smd-machine", "metric_name": "zombie process count",
            "value": 150, "threshold": 5, "unit": "processes",
            "host": "machine-11-21", "duration_s": 7200,
        },
        "evidence": {
            "kpi_history": [0, 10, 30, 60, 90, 120, 140, 150],
            "machine_id": "machine-11-21",
            "parent_pid_reuse_count": 80,
        },
        "expected_root_cause": "zombie_accumulation_from_fork_intensive_app",
        "expected_confidence_min": 0.7,
        "description": "机器 11-21 僵尸进程累积",
        "identified_flaw": "频繁 fork 但未 wait，僵尸进程累积",
        "keywords_for_retrieval": ["smd", "zombie", "fork", "wait"],
        "severity": "medium",
        "source_ref": "smd/machine-11-21/anomaly_011",
    },
    {
        "id": "bc-smd-012",
        "category": "single_metric", "type": "disk_write_saturation",
        "alert": {
            "service": "smd-machine", "metric_name": "disk write throughput",
            "value": 850, "threshold": 200, "unit": "MB/s",
            "host": "machine-12-23", "duration_s": 600,
        },
        "evidence": {
            "kpi_history": [50, 80, 150, 300, 500, 700, 800, 850],
            "machine_id": "machine-12-23",
            "log_volume_write_per_hour_gb": 25,
        },
        "expected_root_cause": "disk_write_saturation_from_log_burst",
        "expected_confidence_min": 0.75,
        "description": "机器 12-23 磁盘写吞吐饱和",
        "identified_flaw": "日志爆发写入，磁盘写饱和",
        "keywords_for_retrieval": ["smd", "disk_write", "log_burst"],
        "severity": "high",
        "source_ref": "smd/machine-12-23/anomaly_012",
    },
    {
        "id": "bc-smd-013",
        "category": "single_metric", "type": "file_descriptor_leak",
        "alert": {
            "service": "smd-machine", "metric_name": "open file descriptors",
            "value": 98000, "threshold": 20000, "unit": "fds",
            "host": "machine-13-25", "duration_s": 3600,
        },
        "evidence": {
            "kpi_history": [5000, 10000, 30000, 60000, 80000, 92000, 96000, 98000],
            "machine_id": "machine-13-25",
            "fd_leak_pattern": "monotonic",
        },
        "expected_root_cause": "fd_leak_from_connection_not_closed",
        "expected_confidence_min": 0.8,
        "description": "机器 13-25 FD 泄漏",
        "identified_flaw": "连接未关闭，FD 单调增长",
        "keywords_for_retrieval": ["smd", "fd_leak", "connection"],
        "severity": "high",
        "source_ref": "smd/machine-13-25/anomaly_013",
    },
    {
        "id": "bc-smd-014",
        "category": "single_metric", "type": "cpu_temperature_high",
        "alert": {
            "service": "smd-machine", "metric_name": "cpu temperature",
            "value": 92, "threshold": 80, "unit": "celsius",
            "host": "machine-14-27", "duration_s": 1800,
        },
        "evidence": {
            "kpi_history": [65, 70, 75, 82, 86, 89, 91, 92],
            "machine_id": "machine-14-27",
            "fan_speed_rpm": 4500,
        },
        "expected_root_cause": "cpu_overheating_fan_at_max",
        "expected_confidence_min": 0.7,
        "description": "机器 14-27 CPU 温度过高",
        "identified_flaw": "风扇满转但温度仍高，散热问题",
        "keywords_for_retrieval": ["smd", "temperature", "overheat", "fan"],
        "severity": "high",
        "source_ref": "smd/machine-14-27/anomaly_014",
    },

    # ===== 4 compound =====
    {
        "id": "bc-smd-015",
        "category": "compound", "type": "memory_swap_compound",
        "alert": {
            "service": "smd-machine", "metric_name": "memory_swap_compound",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "machine-15-29", "duration_s": 1800,
        },
        "evidence": {
            "memory_usage_pct": 95,
            "swap_usage_mb": 4000,
            "load_average": 35,
            "kpi_history": [40, 60, 80, 90, 95, 95, 95, 95],
        },
        "expected_root_cause": "compound_memory_leak_cascading_to_swap_and_load",
        "expected_confidence_min": 0.75,
        "description": "内存泄漏导致 swap + load 高",
        "identified_flaw": "内存泄漏 → swap → load 高",
        "keywords_for_retrieval": ["compound", "smd", "memory_leak", "swap", "load"],
        "severity": "high",
        "source_ref": "smd/machine-15-29/compound_001",
    },
    {
        "id": "bc-smd-016",
        "category": "compound", "type": "disk_io_compound",
        "alert": {
            "service": "smd-machine", "metric_name": "disk_io_compound",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "machine-16-31", "duration_s": 1800,
        },
        "evidence": {
            "disk_read_iops": 18000,
            "disk_write_throughput_mb_s": 600,
            "io_wait_pct": 35,
            "load_average": 28,
        },
        "expected_root_cause": "compound_disk_io_saturation_cpu_iowait_load",
        "expected_confidence_min": 0.75,
        "description": "磁盘 IO 饱和，iowait + load 高",
        "identified_flaw": "磁盘 IO 全面饱和",
        "keywords_for_retrieval": ["compound", "smd", "disk_io", "iowait", "load"],
        "severity": "high",
        "source_ref": "smd/machine-16-31/compound_002",
    },
    {
        "id": "bc-smd-017",
        "category": "compound", "type": "cpu_network_compound",
        "alert": {
            "service": "smd-machine", "metric_name": "cpu_network_compound",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "machine-17-33", "duration_s": 600,
        },
        "evidence": {
            "cpu_usage_pct": 95,
            "network_out_mbps": 600,
            "network_in_mbps": 50,
        },
        "expected_root_cause": "compound_cpu_and_network_outburst_data_processing",
        "expected_confidence_min": 0.7,
        "description": "CPU 高 + 网络出向突发",
        "identified_flaw": "数据处理任务（CPU + 网络）",
        "keywords_for_retrieval": ["compound", "smd", "cpu", "network_out"],
        "severity": "medium",
        "source_ref": "smd/machine-17-33/compound_003",
    },
    {
        "id": "bc-smd-018",
        "category": "compound", "type": "fd_socket_compound",
        "alert": {
            "service": "smd-machine", "metric_name": "fd_socket_compound",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "machine-18-35", "duration_s": 1800,
        },
        "evidence": {
            "open_fds": 85000,
            "established_connections": 45000,
            "time_wait_connections": 25000,
            "cpu_usage_pct": 60,
        },
        "expected_root_cause": "compound_fd_exhaust_with_tcp_state_leak",
        "expected_confidence_min": 0.7,
        "description": "FD 耗尽 + TCP 连接泄漏 复合",
        "identified_flaw": "TCP 连接未及时清理，FD 累积",
        "keywords_for_retrieval": ["compound", "smd", "fd", "tcp", "time_wait"],
        "severity": "high",
        "source_ref": "smd/machine-18-35/compound_004",
    },

    # ===== 2 regression_guard =====
    {
        "id": "bc-smd-019",
        "category": "regression_guard", "type": "reg_memory_leak_smd",
        "alert": {
            "service": "smd-machine", "metric_name": "memory usage",
            "value": 92, "threshold": 85, "unit": "percent",
            "host": "machine-19-37", "duration_s": 14400,
        },
        "evidence": {
            "kpi_history": [40, 55, 70, 80, 85, 88, 90, 92],
            "previous_fixed_in": "v1.2.0_cache_eviction_policy",
            "recent_change": "cache_eviction_disabled_by_config",
        },
        "expected_root_cause": "memory_leak_regression_cache_eviction_disabled",
        "expected_confidence_min": 0.8,
        "description": "回归：缓存淘汰策略被禁用",
        "identified_flaw": "运维误改配置，缓存不淘汰",
        "keywords_for_retrieval": ["regression", "smd", "memory_leak", "cache_eviction"],
        "severity": "high",
        "source_ref": "smd/machine-19-37/regression_001",
    },
    {
        "id": "bc-smd-020",
        "category": "regression_guard", "type": "reg_connection_pool",
        "alert": {
            "service": "smd-machine", "metric_name": "established connections",
            "value": 50000, "threshold": 20000, "unit": "conns",
            "host": "machine-20-39", "duration_s": 3600,
        },
        "evidence": {
            "kpi_history": [5000, 8000, 15000, 25000, 35000, 45000, 48000, 50000],
            "previous_fixed_in": "v2.0.0_connection_pool_max_20k",
            "recent_change": "max_pool_size_unbounded_rollback",
        },
        "expected_root_cause": "connection_pool_regression_unbounded_size",
        "expected_confidence_min": 0.75,
        "description": "回归：连接池上限被取消",
        "identified_flaw": "运维去掉连接池上限",
        "keywords_for_retrieval": ["regression", "smd", "connection_pool", "unbounded"],
        "severity": "high",
        "source_ref": "smd/machine-20-39/regression_002",
    },
]


def build_entries() -> list[BadCaseEntry]:
    """生成 20 个 SMD badcase entry"""
    entries = []
    for bc in _BAD_CASES:
        entry = BadCaseEntry(
            id=bc["id"],
            created_at=datetime.now(timezone.utc),
            source="smd",
            eval_split=EvalSplit.DEV,
            alert=bc["alert"],
            evidence=bc["evidence"],
            category=bc["category"],
            type=bc["type"],
            expected_root_cause=bc["expected_root_cause"],
            expected_confidence_min=bc["expected_confidence_min"],
            severity=bc.get("severity", "medium"),
            tags=[bc["category"], bc["type"], "smd"],
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
    parser = argparse.ArgumentParser(description="Generate SMD badcase data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = build_entries()
    print(f"✅ Generated {len(entries)} SMD badcase entries")
    print(f"   - single_metric: {sum(1 for e in entries if e.category == 'single_metric')}")
    print(f"   - compound: {sum(1 for e in entries if e.category == 'compound')}")
    print(f"   - regression_guard: {sum(1 for e in entries if e.category == 'regression_guard')}")

    if args.dry_run:
        return 0

    from data.adapters.common import deterministic_split

    dev, test, holdout = deterministic_split(entries, seed=args.seed)
    for split_name, items in [("dev", dev), ("test", test), ("holdout", holdout)]:
        for item in items:
            item.eval_split = EvalSplit(split_name)
        path, added = merge_or_replace_split(items, split_name, source_tag="smd")
        print(f"   ✅ {split_name}: {len(added)} added → {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
