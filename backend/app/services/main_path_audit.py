"""W8.1 — MainPathAuditEnvelope.

低风险地把主链路 `incident.context["audit_trail"]` 升级为 W8 风格的
`entity_snapshot` 字段，按 EvidenceBlock 5 字段（object_ref / time /
observation / mechanism / confidence）输出。

设计原则：
- 不修改 audit_trail 字段（已有 BadCase 触发器依赖它）
- 仅新增 entity_snapshot 字段（向后兼容）
- 通过 feature flag `AIOPS_USE_W8_1_AUDIT` 控制（关闭 = 退回到 W7 行为）
- 失败时不影响主链路（异常吞掉，仅警告日志）
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ===== Feature Flag =====

def is_w8_1_audit_enabled() -> bool:
    """判断 W8.1 升级是否启用。默认关闭（保守）。"""
    val = os.getenv("AIOPS_USE_W8_1_AUDIT", "false").lower()
    return val in ("true", "1", "yes", "on")


# ===== 5 路证据识别（从 audit_trail 推断）=====

# 哪些 step 名称对应 5 路证据
_EVIDENCE_STEP_MAPPING: dict[str, tuple[str, str, float]] = {
    # step_name -> (evidence_kind, mechanism, default_confidence)
    "triage": ("monitor", "3sigma_ewma", 0.95),
    "rca_bayesian": ("bayesian", "statistical_inference", 0.85),
    "rca_bfs": ("topology", "bfs_traversal", 0.80),
    "rca_rag": ("rag", "vector_search", 0.78),
    "rca_history": ("history", "chroma_similarity", 0.82),
    "rca_log": ("log", "loki_query", 0.88),
    "rca_reverify_rerun": ("rca_rerun", "reflection_context", 0.80),
    "heal_dry_run": ("heal", "playbook_match", 0.75),
    "heal_plan_b_retry": ("plan_b", "fallback_playbook", 0.70),
    "verify_reflection": ("reflection", "w7_reflection", 0.85),
    "monitor_feedback": ("monitor_feedback", "verification_score", 0.70),
}


# ===== 核心服务 =====


class MainPathAuditEnvelope:
    """Wrap existing audit_trail entries with EvidenceBlock 5 fields.

    Backward-compat: 如果 incident.context["audit_trail"] 已有 entries，
    仅在末尾追加 entity_snapshot 字段，不动 audit_trail。
    """

    def __init__(self) -> None:
        self._enabled = is_w8_1_audit_enabled()

    def upgrade(self, incident_context: dict[str, Any]) -> dict[str, Any] | None:
        """升级 incident.context，返回 entity_snapshot dict。失败时返回 None。

        Args:
            incident_context: incident.context 字典

        Returns:
            entity_snapshot dict；调用失败返回 None
        """
        if not self._enabled:
            return None
        try:
            trail = incident_context.get("audit_trail", [])
            if not trail:
                return None

            # 1. 提取 services / root_cause / steps
            services = self._extract_services(incident_context, trail)
            root_cause = self._extract_root_cause(incident_context)
            steps = [entry.get("step", "") for entry in trail if entry.get("step")]

            # 2. 构造 EvidenceBlock 5 字段
            evidence_blocks = self._build_evidence_blocks(trail, incident_context)

            # 3. 提取 verification 字段（如果存在）
            verification = incident_context.get("verification", {})
            confidence = self._estimate_confidence(verification, root_cause)

            snapshot = {
                "schema_version": "1.0",
                "incident_id": incident_context.get("incident_id"),
                "services": sorted(services),
                "root_cause": root_cause,
                "confidence": confidence,
                "steps": list(dict.fromkeys(steps)),  # 去重但保序
                "evidence_blocks": evidence_blocks,
                "audit_trail_entries": len(trail),
                "verification_status": verification.get("rca_match_status", "unknown"),
                "plan_b_triggered": bool(verification.get("plan_b_triggered", False)),
                "badcase_captured_id": verification.get("badcase_captured_id"),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            incident_context["entity_snapshot"] = snapshot
            return snapshot

        except Exception as exc:
            logger.warning(
                "W8.1 audit upgrade failed (non-fatal)",
                error=str(exc),
            )
            return None

    # ----- 内部方法 -----

    def _extract_services(
        self,
        incident_context: dict[str, Any],
        trail: list[dict],
    ) -> set[str]:
        """从 incident_context 和 trail 提取涉及的 services。"""
        services: set[str] = set()
        # 1. 顶层 incident_context 上的 service 字段
        if "service" in incident_context:
            services.add(incident_context["service"])
        # 2. alert_event.service
        # （routes.py 把 alert 字段直接平铺到 incident.context）
        # 3. rca_event.impact_chain
        rca_event = incident_context.get("rca_event")
        if isinstance(rca_event, dict):
            for s in rca_event.get("impact_chain", []):
                if isinstance(s, str):
                    services.add(s)
        # 4. 从 rca.result WorkingMemory 读
        rca_result = incident_context.get("rca_result")
        if isinstance(rca_result, dict):
            impact = rca_result.get("impact_services", [])
            if isinstance(impact, list):
                services.update(s for s in impact if isinstance(s, str))
        return services

    def _extract_root_cause(self, incident_context: dict[str, Any]) -> str | None:
        """从多种渠道提取 root_cause。"""
        # 1. rca_event.root_cause
        rca_event = incident_context.get("rca_event")
        if isinstance(rca_event, dict) and rca_event.get("root_cause"):
            return rca_event["root_cause"]
        # 2. rca_result.root_cause
        rca_result = incident_context.get("rca_result")
        if isinstance(rca_result, dict) and rca_result.get("root_cause"):
            return rca_result["root_cause"]
        # 3. incident.context["root_cause"]
        if "root_cause" in incident_context:
            return incident_context["root_cause"]
        return None

    def _build_evidence_blocks(
        self,
        trail: list[dict],
        incident_context: dict[str, Any],
    ) -> list[dict]:
        """从 audit_trail 推断 EvidenceBlock 5 字段。"""
        blocks: list[dict] = []
        for idx, entry in enumerate(trail):
            step = entry.get("step")
            if not step:
                continue
            mapping = _EVIDENCE_STEP_MAPPING.get(step)
            if mapping is None:
                # 未知 step → 兜底设为通用 observation
                kind, mechanism, conf = "generic", "audit_trail", 0.5
            else:
                kind, mechanism, conf = mapping

            block = {
                "block_id": f"eb-legacy-{idx:04d}",
                "object_ref": self._resolve_object_ref(incident_context, kind),
                "time": entry.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                "observation": self._extract_observation(entry),
                "mechanism": mechanism,
                "confidence": conf,
                "is_evidence_against_hypothesis": [],
                "source_step": step,
                "source_kind": kind,
            }
            blocks.append(block)
        return blocks

    def _resolve_object_ref(
        self,
        incident_context: dict[str, Any],
        kind: str,
    ) -> str:
        """根据 kind 推断 object_ref（UModel 风格 `type:id`）。"""
        if kind in ("monitor", "monitor_feedback"):
            service = incident_context.get("service", "unknown")
            return f"service:{service}"
        if kind in ("bayesian", "rag", "rca_rerun", "reflection"):
            return f"incident:{incident_context.get('incident_id', 'unknown')}"
        if kind == "topology":
            return f"topology:{incident_context.get('service', 'unknown')}"
        if kind in ("history", "log"):
            return f"memory:{kind}"
        if kind in ("heal", "plan_b"):
            return f"action:{incident_context.get('service', 'unknown')}"
        return f"generic:{kind}"

    def _extract_observation(self, entry: dict) -> str:
        """从 audit_trail entry 提取 observation 文本。"""
        trigger = entry.get("trigger", "")
        outcome = entry.get("outcome", "")
        actions = entry.get("actions_taken", [])
        actions_str = ", ".join(str(a) for a in actions) if actions else ""

        parts = []
        if trigger:
            parts.append(f"trigger: {trigger}")
        if actions_str:
            parts.append(f"actions: {actions_str}")
        if outcome:
            parts.append(f"outcome: {outcome}")
        return " | ".join(parts) or "(no detail)"

    def _estimate_confidence(
        self,
        verification: dict,
        root_cause: str | None,
    ) -> float:
        """基于 verify_phase 11 字段估算综合 confidence。"""
        if not root_cause:
            return 0.0
        # 1. prefer_final_confidence（如果 verification 已有）
        if "rca_final_confidence" in verification and verification["rca_final_confidence"]:
            return float(verification["rca_final_confidence"])
        # 2. healed = True → 高 confidence
        if verification.get("heal_recovered"):
            return 0.85
        # 3. mismatch / plan_b 触发 → 中 confidence
        if verification.get("rca_match_status") == "mismatch":
            return 0.55
        if verification.get("plan_b_triggered"):
            return 0.65
        # 4. unknown
        return 0.5


# ===== 工厂函数 =====


def get_main_path_audit_envelope() -> MainPathAuditEnvelope:
    """获取 MainPathAuditEnvelope 单例。"""
    return MainPathAuditEnvelope()
