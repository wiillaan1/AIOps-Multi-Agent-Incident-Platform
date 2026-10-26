"""
AIOps Agent Platform - Layered Evaluation Datasets

把 `data/datasets.FAULT_SCENARIOS` 和 `evaluation/scenarios.SCENARIO_REGISTRY`
派生为 4 层评测数据集：

- UnitDatasets        — 单 skill 单调用测试（≥6 条）
- IntegrationDatasets — 多 skill 协同测试（≥4 条）
- E2EDatasets         — 端到端场景测试（15 条 = FAULT_SCENARIOS 全部）
- GoldenDatasets      — 回归基线（3-8 条代表场景，每类至少一条）

替代原先 `data/datasets.py:EvaluationDatasets` 死代码。
"""

from __future__ import annotations

from typing import Any, ClassVar


class UnitDatasets:
    """单 skill 单调用测试样本"""

    SAMPLES: ClassVar[list[dict[str, Any]]] = [
        # ============== E2E unit (time score 边界) ==============
        {
            "id": "unit_e2e_time_fast",
            "dimension": "e2e",
            "skill": "end_to_end",
            "sample": {
                "id": "u-e2e-fast",
                "actual_result": {
                    "resolved": True,
                    "automated": True,
                    "root_cause": "traffic_spike",
                    "action": "scale_up",
                    "time_to_resolve_seconds": 60,
                },
                "expected_result": {
                    "resolved": True,
                    "root_cause": "traffic_spike",
                    "action": "scale_up",
                },
            },
            "expected_time_score": 1.0,
        },
        {
            "id": "unit_e2e_time_mid",
            "dimension": "e2e",
            "skill": "end_to_end",
            "sample": {
                "id": "u-e2e-mid",
                "actual_result": {
                    "resolved": True,
                    "automated": True,
                    "root_cause": "memory_leak",
                    "action": "restart_pod",
                    "time_to_resolve_seconds": 240,
                },
                "expected_result": {
                    "resolved": True,
                    "root_cause": "memory_leak",
                    "action": "restart_pod",
                },
            },
            "expected_time_score": 0.8,
        },
        {
            "id": "unit_e2e_time_slow",
            "dimension": "e2e",
            "skill": "end_to_end",
            "sample": {
                "id": "u-e2e-slow",
                "actual_result": {
                    "resolved": False,
                    "automated": False,
                    "escalated": True,
                    "root_cause": "unknown",
                    "action": "escalate",
                    "time_to_resolve_seconds": 900,
                },
                "expected_result": {
                    "resolved": False,
                    "root_cause": "unknown",
                    "action": "escalate",
                },
            },
            "expected_time_score_below": 0.3,
        },
        # ============== Reasoning unit (perfect match) ==============
        {
            "id": "unit_reasoning_perfect",
            "dimension": "reasoning",
            "skill": "rca_agent",
            "sample": {
                "id": "u-rsn-001",
                "predicted": {
                    "root_cause": "memory_leak",
                    "confidence": 0.9,
                    "evidence": {
                        "metrics": ["memory_usage"],
                        "logs": ["oom_warning"],
                    },
                    "impact_chain": ["memory_leak", "oom"],
                    "reasoning_steps": [
                        {"description": "observe memory growth", "order": 1},
                        {"description": "identify leak", "order": 2},
                    ],
                },
                "ground_truth": {
                    "root_cause": "memory_leak",
                    "confidence": 0.9,
                    "evidence": {
                        "metrics": ["memory_usage"],
                        "logs": ["oom_warning"],
                    },
                    "impact_chain": ["memory_leak", "oom"],
                    "reasoning_steps": [
                        {"description": "observe memory growth", "order": 1},
                        {"description": "identify leak", "order": 2},
                    ],
                },
            },
            "expected_root_cause_accuracy": 1.0,
        },
        # ============== Tool call unit (selection correctness) ==============
        {
            "id": "unit_tool_selection_correct",
            "dimension": "tool_call",
            "skill": "tool_router",
            "sample": {
                "tool_calls": [
                    {
                        "tool_name": "query_metrics",
                        "parameters": {
                            "service": "order-service",
                            "metric": "cpu_usage_percent",
                        },
                        "expected_tool": "query_metrics",
                        "expected_parameters": {
                            "service": "order-service",
                            "metric": "cpu_usage_percent",
                        },
                        "execution_result": {"success": True},
                    },
                ],
            },
            "expected_selection_accuracy": 1.0,
        },
        # ============== RAG unit (retrieval precision) ==============
        {
            "id": "unit_rag_perfect_retrieval",
            "dimension": "rag",
            "skill": "knowledge_base",
            "sample": {
                "id": "u-rag-001",
                "query": "How to fix high CPU on K8s pod?",
                "retrieved_docs": [
                    {
                        "id": "kb_001",
                        "content": "High CPU after deployment - rollback",
                        "score": 0.9,
                    },
                ],
                "relevant_docs": ["kb_001"],
                "generated_answer": "Rollback recent deployment to fix high CPU.",
            },
            "expected_precision": 1.0,
            "expected_recall": 1.0,
        },
    ]

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        return [s.copy() for s in cls.SAMPLES]


class IntegrationDatasets:
    """多 skill 协同测试样本"""

    SAMPLES: ClassVar[list[dict[str, Any]]] = [
        # ============== RCA 阶段：metrics + logs + kb 三件套 ==============
        {
            "id": "integ_rca_trio",
            "dimension": "reasoning",
            "skill_chain": ["query_metrics", "query_logs", "query_knowledge_base"],
            "scenario_id": "fs_001",
            "sample": {
                "tool_calls": [
                    {
                        "tool_name": "query_metrics",
                        "parameters": {
                            "service": "order-service",
                            "metric": "cpu_usage_percent",
                        },
                        "expected_tool": "query_metrics",
                        "expected_parameters": {
                            "service": "order-service",
                            "metric": "cpu_usage_percent",
                        },
                        "execution_result": {"success": True},
                    },
                    {
                        "tool_name": "query_logs",
                        "parameters": {
                            "service": "order-service",
                            "level": "error",
                            "lookback_minutes": 30,
                        },
                        "expected_tool": "query_logs",
                        "expected_parameters": {
                            "service": "order-service",
                            "level": "error",
                        },
                        "execution_result": {"success": True},
                    },
                    {
                        "tool_name": "query_knowledge_base",
                        "parameters": {"query": "deployment rollback", "top_k": 3},
                        "expected_tool": "query_knowledge_base",
                        "expected_parameters": {"query": "deployment rollback"},
                        "execution_result": {"success": True},
                    },
                ],
                "predicted": {
                    "root_cause": "recent_deployment",
                    "confidence": 0.85,
                    "evidence": {
                        "metrics": ["cpu_usage_percent"],
                        "logs": ["deployment_marker"],
                    },
                },
                "ground_truth": {
                    "root_cause": "recent_deployment",
                    "confidence": 0.85,
                    "evidence": {
                        "metrics": ["cpu_usage_percent"],
                        "logs": ["deployment_marker"],
                    },
                },
            },
            "expected": {
                "tool_chain_order_correct": True,
                "root_cause_correct": True,
            },
        },
        # ============== Heal 阶段：playbook + dry_run + verify ==============
        {
            "id": "integ_heal_trio",
            "dimension": "tool_call",
            "skill_chain": ["get_playbook", "execute_playbook_step", "verify_health"],
            "scenario_id": "fs_002",
            "sample": {
                "tool_calls": [
                    {
                        "tool_name": "get_playbook",
                        "parameters": {"fault_type": "memory_leak"},
                        "expected_tool": "get_playbook",
                        "expected_parameters": {"fault_type": "memory_leak"},
                        "execution_result": {"success": True},
                    },
                    {
                        "tool_name": "execute_playbook_step",
                        "parameters": {
                            "playbook_id": "pb_memory_leak",
                            "step_index": 0,
                            "parameters": {"service": "payment-service"},
                            "dry_run": True,
                        },
                        "expected_tool": "execute_playbook_step",
                        "expected_parameters": {
                            "playbook_id": "pb_memory_leak",
                            "step_index": 0,
                        },
                        "execution_result": {"success": True},
                    },
                    {
                        "tool_name": "verify_health",
                        "parameters": {"service": "payment-service", "timeout": 30},
                        "expected_tool": "verify_health",
                        "expected_parameters": {"service": "payment-service"},
                        "execution_result": {"success": True},
                    },
                ],
            },
            "expected": {
                "tool_chain_order_correct": True,
                "all_tools_succeed": True,
            },
        },
        # ============== Change 阶段：risk + approval + audit ==============
        {
            "id": "integ_change_trio",
            "dimension": "e2e",
            "skill_chain": ["risk_calculate", "approval_decide", "audit_log"],
            "scenario_id": "fs_005",
            "sample": {
                "id": "integ-change-fs005",
                "actual_result": {
                    "resolved": True,
                    "automated": False,
                    "escalated": False,
                    "root_cause": "configuration_error",
                    "action": "rollback_config",
                    "time_to_resolve_seconds": 300,
                    "approval_status": "pending",
                    "approvers": ["oncall", "team_lead"],
                    "risk_level": "high",
                },
                "expected_result": {
                    "resolved": True,
                    "root_cause": "configuration_error",
                    "action": "rollback_config",
                },
            },
            "expected": {
                "approval_decision_consistent": True,
                "risk_level_match": True,
            },
        },
        # ============== Tool selection cascade：metrics → logs → traces ==============
        {
            "id": "integ_tool_cascade",
            "dimension": "tool_call",
            "skill_chain": ["query_metrics", "query_logs", "query_traces"],
            "scenario_id": "fs_003",
            "sample": {
                "tool_calls": [
                    {
                        "tool_name": "query_metrics",
                        "parameters": {
                            "service": "user-service",
                            "metric": "p99_latency_ms",
                        },
                        "expected_tool": "query_metrics",
                        "expected_parameters": {
                            "service": "user-service",
                            "metric": "p99_latency_ms",
                        },
                        "execution_result": {"success": True},
                    },
                    {
                        "tool_name": "query_logs",
                        "parameters": {
                            "service": "user-service",
                            "level": "warn",
                        },
                        "expected_tool": "query_logs",
                        "expected_parameters": {
                            "service": "user-service",
                        },
                        "execution_result": {"success": True},
                    },
                    {
                        "tool_name": "get_dependencies",
                        "parameters": {"service": "user-service"},
                        "expected_tool": "get_dependencies",
                        "expected_parameters": {"service": "user-service"},
                        "execution_result": {"success": True},
                    },
                ],
            },
            "expected": {
                "tool_chain_order_correct": True,
            },
        },
    ]

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        return [s.copy() for s in cls.SAMPLES]


class E2EDatasets:
    """端到端场景样本，源数据集：SCENARIO_REGISTRY × FAULT_SCENARIOS"""

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        from app.data.datasets import FAULT_SCENARIOS
        from app.evaluation.scenarios import get_scenario

        samples: list[dict[str, Any]] = []
        for fs in FAULT_SCENARIOS:
            spec = get_scenario(fs["id"])
            if spec is None:
                continue
            samples.append({
                "scenario_id": fs["id"],
                "name": fs.get("name", ""),
                "description": fs.get("description", ""),
                "service": fs.get("service", ""),
                "category": fs.get("category", ""),
                "severity": fs.get("severity", ""),
                # ground-truth 字段（评测对照）
                "root_cause": fs.get("root_cause", ""),
                "expected_action": fs.get("expected_action", {}).get("type", ""),
                "expected_level": fs.get("expected_action", {}).get("level", ""),
                "expected_approval": fs.get("expected_approval", ""),
                "blast_radius": fs.get("blast_radius", 0.0),
                "business_rule": (
                    fs.get("business_rules", [""])[0]
                    if fs.get("business_rules")
                    else ""
                ),
                # 评测维度（spec 决定）
                "dimensions": list(spec.dimensions),
                "relevant_docs": list(spec.relevant_docs),
                # 注入的完整 ground_truth 字典供 ScenarioDrivenEvaluator 使用
                "ground_truth": dict(spec.ground_truth),
            })
        return samples


class GoldenDatasets:
    """回归基线样本：每个 category 选代表场景，确保覆盖三类"""

    # 由 `SCENARIO_REGISTRY` 中精心挑选的 6 个代表场景
    _GOLDEN_IDS: ClassVar[list[str]] = [
        "fs_001",       # infrastructure: deployment
        "fs_002",       # infrastructure: memory leak
        "fs_003",       # infrastructure: db pool
        "fs_biz_003",   # business: token cache
        "fs_biz_007",   # business_logic: duplicate charge
        "fs_biz_010",   # business_logic: callback loss
    ]

    @classmethod
    def all(cls) -> list[dict[str, Any]]:
        e2e_samples = {s["scenario_id"]: s for s in E2EDatasets.all()}
        golden: list[dict[str, Any]] = []
        for sid in cls._GOLDEN_IDS:
            if sid in e2e_samples:
                golden.append(e2e_samples[sid])
        return golden

    @classmethod
    def ids(cls) -> list[str]:
        return list(cls._GOLDEN_IDS)
