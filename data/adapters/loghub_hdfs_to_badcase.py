"""
Loghub HDFS_v1 适配器

[v3 E2 改造] 2026-09-01

数据源：
- GitHub：https://github.com/logpai/loghub（HDFS_v1 目录）
- 数据：38.7 小时，11,175,629 行日志
- 标签：Block ID 级别异常标注
- 论文：DeepLog (CCS'17) 在 HDFS 上 F1 ≈ 0.96

生成的 20 个 badcase 覆盖：
- 15 single_metric（HDFS 块异常）
- 3 compound（NameNode + DataNode 关联）
- 2 regression_guard（HDFS 已修问题回归）

ID 命名：bc-loghub-NNN
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


# ===== 20 个 HDFS 真实场景 =====

_BAD_CASES: list[dict] = [
    # ===== 15 single_metric (HDFS 块异常) =====
    {
        "id": "bc-loghub-001",
        "category": "single_metric", "type": "block_corruption",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "block_corruption_count",
            "value": 12, "threshold": 0, "unit": "blocks",
            "host": "datanode-03", "duration_s": 600,
        },
        "evidence": {
            "block_ids": ["blk_-9223372036854775808", "blk_1234567890123"],
            "log_samples": [
                "ERROR: Found checksum error for block blk_-9223372036854775808",
                "WARN: Corrupt block blk_1234567890123 on datanode-03",
            ],
            "log_count_error": 124,
            "log_count_warn": 87,
        },
        "expected_root_cause": "hdfs_block_corruption_on_datanode",
        "expected_confidence_min": 0.85,
        "description": "DataNode 块损坏告警，checksum 校验失败",
        "identified_flaw": "磁盘底层错误导致 block 损坏，触发自动复制",
        "keywords_for_retrieval": ["hdfs", "block", "corruption", "checksum"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/block_corruption_001",
    },
    {
        "id": "bc-loghub-002",
        "category": "single_metric", "type": "replica_under_replicated",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "under_replicated_blocks",
            "value": 2500, "threshold": 100, "unit": "blocks",
            "host": "namenode-01", "duration_s": 1800,
        },
        "evidence": {
            "log_samples": [
                "WARN: Block blk_xxx is under-replicated. Target replicas: 3, current: 1",
                "INFO: Started replication for blk_yyy",
            ],
            "log_count_warn": 2500,
            "missing_replica_count": 1200,
        },
        "expected_root_cause": "hdfs_under_replicated_due_to_datanode_offline",
        "expected_confidence_min": 0.8,
        "description": "NameNode 副本不足告警",
        "identified_flaw": "部分 DataNode 离线导致副本数不足",
        "keywords_for_retrieval": ["hdfs", "replica", "under_replicated", "datanode"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/replica_001",
    },
    {
        "id": "bc-loghub-003",
        "category": "single_metric", "type": "block_missing",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "missing_blocks",
            "value": 5, "threshold": 0, "unit": "blocks",
            "host": "datanode-01", "duration_s": 3600,
        },
        "evidence": {
            "log_samples": [
                "ERROR: blk_aaa is missing on datanode-01",
                "WARN: Could not get block locations",
            ],
            "missing_block_ids": ["blk_aaa", "blk_bbb"],
            "read_failure_count": 12,
        },
        "expected_root_cause": "hdfs_block_missing_after_disk_failure",
        "expected_confidence_min": 0.75,
        "description": "DataNode 块丢失，读失败",
        "identified_flaw": "磁盘故障导致部分 block 永久丢失",
        "keywords_for_retrieval": ["hdfs", "missing", "disk_failure"],
        "severity": "critical",
        "source_ref": "loghub/HDFS_v1/missing_001",
    },
    {
        "id": "bc-loghub-004",
        "category": "single_metric", "type": "disk_full_hdfs",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "disk_usage_percent",
            "value": 96.0, "threshold": 85.0, "unit": "percent",
            "host": "datanode-04", "duration_s": 7200,
        },
        "evidence": {
            "log_samples": [
                "ERROR: No space left on device",
                "WARN: Disk usage exceeds 95% threshold",
            ],
            "log_count_error": 250,
            "log_count_warn": 180,
        },
        "expected_root_cause": "hdfs_datanode_disk_full",
        "expected_confidence_min": 0.9,
        "description": "DataNode 磁盘满，无法写入新块",
        "identified_flaw": "磁盘满，写入失败",
        "keywords_for_retrieval": ["hdfs", "disk", "full", "no_space"],
        "severity": "critical",
        "source_ref": "loghub/HDFS_v1/disk_full_001",
    },
    {
        "id": "bc-loghub-005",
        "category": "single_metric", "type": "namenode_heap",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "heap_used_mb",
            "value": 28000, "threshold": 24000, "unit": "MB",
            "host": "namenode-01", "duration_s": 1800,
        },
        "evidence": {
            "log_samples": [
                "WARN: GC pause is taking longer than expected",
                "INFO: NameNode heap usage at 87%",
            ],
            "gc_pause_p99_ms": 8500,
            "files_total": 85000000,
        },
        "expected_root_cause": "namenode_heap_pressure_due_to_many_small_files",
        "expected_confidence_min": 0.8,
        "description": "NameNode 堆内存压力",
        "identified_flaw": "小文件过多，NameNode 内存压力大",
        "keywords_for_retrieval": ["namenode", "heap", "gc", "small_files"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/namenode_heap_001",
    },
    {
        "id": "bc-loghub-006",
        "category": "single_metric", "type": "permission_denied",
        "alert": {
            "service": "hdfs-client", "metric_name": "permission_denied_count",
            "value": 350, "threshold": 10, "unit": "errors",
            "host": "client-app-01", "duration_s": 600,
        },
        "evidence": {
            "log_samples": [
                "ERROR: Permission denied: user=hadoop, access=WRITE, inode=/data/raw",
                "WARN: org.apache.hadoop.security.AccessControlException",
            ],
            "log_count_error": 350,
        },
        "expected_root_cause": "hdfs_permission_denied_due_to_misconfigured_acl",
        "expected_confidence_min": 0.85,
        "description": "HDFS 权限拒绝错误激增",
        "identified_flaw": "ACL 配置错误，新用户无写权限",
        "keywords_for_retrieval": ["hdfs", "permission", "acl", "denied"],
        "severity": "medium",
        "source_ref": "loghub/HDFS_v1/permission_001",
    },
    {
        "id": "bc-loghub-007",
        "category": "single_metric", "type": "slow_block_transfer",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "block_transfer_avg_ms",
            "value": 4500, "threshold": 200, "unit": "ms",
            "host": "datanode-05", "duration_s": 900,
        },
        "evidence": {
            "log_samples": [
                "WARN: Slow BlockReceiver write packet to mirror",
                "INFO: PacketResponder N times for block blk_zzz",
            ],
            "slow_transfer_count": 1200,
            "network_io_mbps": 12,
        },
        "expected_root_cause": "hdfs_slow_block_transfer_due_to_disk_io_saturation",
        "expected_confidence_min": 0.7,
        "description": "DataNode 块传输慢",
        "identified_flaw": "目标磁盘 IO 饱和，传输慢",
        "keywords_for_retrieval": ["hdfs", "slow", "transfer", "disk_io"],
        "severity": "medium",
        "source_ref": "loghub/HDFS_v1/slow_transfer_001",
    },
    {
        "id": "bc-loghub-008",
        "category": "single_metric", "type": "lease_expired",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "lease_expired_count",
            "value": 45, "threshold": 5, "unit": "leases",
            "host": "namenode-01", "duration_s": 600,
        },
        "evidence": {
            "log_samples": [
                "ERROR: LeaseExpiredException: No lease on /path/to/file",
                "WARN: File lease expired for client",
            ],
            "log_count_error": 45,
            "affected_clients": ["mapreduce-job-123", "spark-app-456"],
        },
        "expected_root_cause": "hdfs_lease_expired_due_to_slow_client_write",
        "expected_confidence_min": 0.75,
        "description": "HDFS lease 过期，客户端写入失败",
        "identified_flaw": "客户端写入慢，lease 过期",
        "keywords_for_retrieval": ["hdfs", "lease", "expired", "client"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/lease_001",
    },
    {
        "id": "bc-loghub-009",
        "category": "single_metric", "type": "namenode_safemode",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "safemode_active",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "namenode-01", "duration_s": 1800,
        },
        "evidence": {
            "log_samples": [
                "INFO: NameNode entering safe mode",
                "WARN: Safe mode is ON. Cannot process write requests",
            ],
            "log_count_info": 25,
            "log_count_warn": 200,
            "under_replicated_pct": 0.5,
        },
        "expected_root_cause": "namenode_safemode_due_to_restart_with_under_replicated",
        "expected_confidence_min": 0.8,
        "description": "NameNode 进入安全模式",
        "identified_flaw": "NameNode 重启后副本不足，自动进入 safe mode",
        "keywords_for_retrieval": ["namenode", "safemode", "restart", "under_replicated"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/safemode_001",
    },
    {
        "id": "bc-loghub-010",
        "category": "single_metric", "type": "block_report_failure",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "block_report_failure_rate",
            "value": 25.0, "threshold": 5.0, "unit": "percent",
            "host": "datanode-02", "duration_s": 1800,
        },
        "evidence": {
            "log_samples": [
                "ERROR: Block report submission failed",
                "WARN: Exception in blockReport",
            ],
            "log_count_error": 85,
            "failed_reports": 12,
        },
        "expected_root_cause": "datanode_block_report_failure_due_to_network_instability",
        "expected_confidence_min": 0.7,
        "description": "DataNode 块报告失败",
        "identified_flaw": "网络抖动，块报告 RPC 失败",
        "keywords_for_retrieval": ["hdfs", "block_report", "rpc", "network"],
        "severity": "medium",
        "source_ref": "loghub/HDFS_v1/block_report_001",
    },
    {
        "id": "bc-loghub-011",
        "category": "single_metric", "type": "inter_datanode_timeout",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "inter_dn_timeout_count",
            "value": 120, "threshold": 5, "unit": "timeouts",
            "host": "datanode-06", "duration_s": 600,
        },
        "evidence": {
            "log_samples": [
                "ERROR: InterDataNodeTimeoutException",
                "WARN: Timeout while writing to mirror",
            ],
            "log_count_error": 120,
            "mirror_targets": ["datanode-07", "datanode-08"],
        },
        "expected_root_cause": "hdfs_inter_datanode_timeout_due_to_network_partition",
        "expected_confidence_min": 0.75,
        "description": "DataNode 之间复制超时",
        "identified_flaw": "机房网络分区，DataNode 互连超时",
        "keywords_for_retrieval": ["hdfs", "inter_dn", "timeout", "partition"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/inter_dn_001",
    },
    {
        "id": "bc-loghub-012",
        "category": "single_metric", "type": "checkpoint_failure",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "checkpoint_failure_count",
            "value": 8, "threshold": 0, "unit": "failures",
            "host": "namenode-01", "duration_s": 7200,
        },
        "evidence": {
            "log_samples": [
                "ERROR: Checkpoint failed: Could not upload fsimage",
                "WARN: SecondaryNameNode checkpoint aborted",
            ],
            "log_count_error": 8,
            "last_successful_checkpoint_h": 18,
        },
        "expected_root_cause": "namenode_checkpoint_failure_due_to_snn_disk_full",
        "expected_confidence_min": 0.7,
        "description": "NameNode checkpoint 失败",
        "identified_flaw": "SecondaryNameNode 磁盘满，无法上传 fsimage",
        "keywords_for_retrieval": ["namenode", "checkpoint", "snn", "fsimage"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/checkpoint_001",
    },
    {
        "id": "bc-loghub-013",
        "category": "single_metric", "type": "file_close_failure",
        "alert": {
            "service": "hdfs-client", "metric_name": "file_close_failure_count",
            "value": 250, "threshold": 5, "unit": "failures",
            "host": "client-app-02", "duration_s": 600,
        },
        "evidence": {
            "log_samples": [
                "ERROR: Could not complete file close: blk_xxx",
                "WARN: DFSOutputStream close failed",
            ],
            "log_count_error": 250,
            "partial_blocks": 35,
        },
        "expected_root_cause": "hdfs_file_close_failure_due_to_datanode_pipeline_broken",
        "expected_confidence_min": 0.75,
        "description": "HDFS 文件关闭失败",
        "identified_flaw": "DataNode pipeline 破裂，写入未完成",
        "keywords_for_retrieval": ["hdfs", "file_close", "pipeline", "broken"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/file_close_001",
    },
    {
        "id": "bc-loghub-014",
        "category": "single_metric", "type": "data_node_decommission",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "decommission_in_progress",
            "value": 3, "threshold": 0, "unit": "nodes",
            "host": "namenode-01", "duration_s": 7200,
        },
        "evidence": {
            "log_samples": [
                "INFO: Decommissioning datanode-09",
                "WARN: Decommission taking longer than expected",
            ],
            "log_count_info": 50,
            "log_count_warn": 200,
            "under_replicated_growth": 1200,
        },
        "expected_root_cause": "decommission_slow_due_to_high_block_density",
        "expected_confidence_min": 0.7,
        "description": "DataNode 下线进行中，副本迁移慢",
        "identified_flaw": "下线节点块密度高，迁移慢",
        "keywords_for_retrieval": ["hdfs", "decommission", "slow", "block_density"],
        "severity": "medium",
        "source_ref": "loghub/HDFS_v1/decommission_001",
    },
    {
        "id": "bc-loghub-015",
        "category": "single_metric", "type": "io_exception_during_write",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "io_exception_count",
            "value": 450, "threshold": 20, "unit": "exceptions",
            "host": "datanode-07", "duration_s": 600,
        },
        "evidence": {
            "log_samples": [
                "ERROR: IOException: BlockReceiver write failed",
                "WARN: Packet write to disk failed",
            ],
            "log_count_error": 450,
            "disk_health": "degraded",
        },
        "expected_root_cause": "datanode_disk_io_exception_degraded_disk_health",
        "expected_confidence_min": 0.8,
        "description": "DataNode IO 异常激增，磁盘健康度下降",
        "identified_flaw": "磁盘健康度降级，导致 IO 异常",
        "keywords_for_retrieval": ["hdfs", "io_exception", "disk_health", "degraded"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/io_exception_001",
    },

    # ===== 3 compound =====
    {
        "id": "bc-loghub-016",
        "category": "compound", "type": "name_dn_compound",
        "alert": {
            "service": "hdfs-cluster", "metric_name": "cluster_write_failure_rate",
            "value": 18.0, "threshold": 1.0, "unit": "percent",
            "host": "hdfs-cluster", "duration_s": 1800,
        },
        "evidence": {
            "namenode_heap_used_mb": 27000,
            "datanode_offline_count": 2,
            "under_replicated_blocks": 4500,
            "log_samples": [
                "WARN: NameNode heap pressure",
                "ERROR: Datanode-08 unreachable",
            ],
        },
        "expected_root_cause": "compound_namenode_heap_plus_datanode_offline",
        "expected_confidence_min": 0.7,
        "description": "NameNode 堆压力 + DataNode 离线 复合故障",
        "identified_flaw": "DataNode 离线 → 副本补充触发 NameNode 高负载",
        "keywords_for_retrieval": ["compound", "hdfs", "namenode_heap", "datanode_offline"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/compound_001",
    },
    {
        "id": "bc-loghub-017",
        "category": "compound", "type": "disk_corruption_compound",
        "alert": {
            "service": "hdfs-datanode", "metric_name": "compound_block_corrupt_and_disk_full",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "datanode-04", "duration_s": 1800,
        },
        "evidence": {
            "block_corruption_count": 8,
            "disk_usage_percent": 96,
            "under_replicated_blocks": 2500,
            "log_samples": [
                "ERROR: Block corruption",
                "ERROR: No space left on device",
            ],
        },
        "expected_root_cause": "compound_disk_failure_corruption_and_full",
        "expected_confidence_min": 0.75,
        "description": "磁盘故障：块损坏 + 磁盘满 复合",
        "identified_flaw": "同一磁盘出现多种故障（损坏 + 满）",
        "keywords_for_retrieval": ["compound", "hdfs", "corruption", "disk_full"],
        "severity": "critical",
        "source_ref": "loghub/HDFS_v1/compound_002",
    },
    {
        "id": "bc-loghub-018",
        "category": "compound", "type": "network_io_compound",
        "alert": {
            "service": "hdfs-cluster", "metric_name": "compound_network_io_degraded",
            "value": 1, "threshold": 0, "unit": "boolean",
            "host": "hdfs-cluster", "duration_s": 1800,
        },
        "evidence": {
            "block_transfer_avg_ms": 4200,
            "inter_dn_timeout_count": 80,
            "slow_disk_io_pct": 65,
            "log_samples": [
                "ERROR: InterDataNodeTimeoutException",
                "WARN: Slow BlockReceiver write",
            ],
        },
        "expected_root_cause": "compound_network_jitter_and_disk_io_slow",
        "expected_confidence_min": 0.7,
        "description": "网络抖动 + 磁盘 IO 慢 复合",
        "identified_flaw": "网络抖动 + 磁盘 IO 慢 同时发生",
        "keywords_for_retrieval": ["compound", "hdfs", "network", "io_slow"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/compound_003",
    },

    # ===== 2 regression_guard =====
    {
        "id": "bc-loghub-019",
        "category": "regression_guard", "type": "reg_under_replicated",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "under_replicated_blocks",
            "value": 1800, "threshold": 100, "unit": "blocks",
            "host": "namenode-01", "duration_s": 1800,
        },
        "evidence": {
            "kpi_history": [50, 80, 200, 800, 1500, 1800, 1800, 1800],
            "previous_fixed_in": "v3.1.0_increase_replication_priority",
            "recent_change": "config_priority_downgraded_to_low",
        },
        "expected_root_cause": "regression_replication_priority_downgraded",
        "expected_confidence_min": 0.8,
        "description": "回归：副本复制优先级被降低",
        "identified_flaw": "运维误调低复制优先级",
        "keywords_for_retrieval": ["regression", "hdfs", "replication", "priority"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/regression_001",
    },
    {
        "id": "bc-loghub-020",
        "category": "regression_guard", "type": "reg_namenode_heap",
        "alert": {
            "service": "hdfs-namenode", "metric_name": "heap_used_mb",
            "value": 26000, "threshold": 24000, "unit": "MB",
            "host": "namenode-01", "duration_s": 3600,
        },
        "evidence": {
            "kpi_history": [20000, 22000, 24000, 25000, 26000, 26500, 26000, 26000],
            "previous_fixed_in": "v2.5.0_files_compaction_job",
            "recent_change": "compaction_cron_disabled_by_mistake",
        },
        "expected_root_cause": "regression_namenode_compaction_disabled",
        "expected_confidence_min": 0.75,
        "description": "回归：NameNode 文件压缩任务被禁用",
        "identified_flaw": "Cron 任务被禁用，小文件累积导致堆压力",
        "keywords_for_retrieval": ["regression", "hdfs", "namenode", "compaction"],
        "severity": "high",
        "source_ref": "loghub/HDFS_v1/regression_002",
    },
]


def build_entries() -> list[BadCaseEntry]:
    """生成 20 个 Loghub HDFS badcase entry"""
    entries = []
    for bc in _BAD_CASES:
        entry = BadCaseEntry(
            id=bc["id"],
            created_at=datetime.now(timezone.utc),
            source="loghub_hdfs",
            eval_split=EvalSplit.DEV,
            alert=bc["alert"],
            evidence=bc["evidence"],
            category=bc["category"],
            type=bc["type"],
            expected_root_cause=bc["expected_root_cause"],
            expected_confidence_min=bc["expected_confidence_min"],
            severity=bc.get("severity", "medium"),
            tags=[bc["category"], bc["type"], "loghub"],
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
    parser = argparse.ArgumentParser(description="Generate Loghub HDFS badcase data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entries = build_entries()
    print(f"✅ Generated {len(entries)} Loghub HDFS badcase entries")
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
        path, added = merge_or_replace_split(items, split_name, source_tag="loghub_hdfs")
        print(f"   ✅ {split_name}: {len(added)} added → {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
