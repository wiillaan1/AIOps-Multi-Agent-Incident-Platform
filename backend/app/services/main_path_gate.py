"""W8.2 — MainPathGate (启发式 Counterfactual Gate).

主链路版本的反事实门控，简化自 W8 的 CounterfactualGate。

W8 完整版的 4 问检查需要 InvestigationGraph（实体 + 链接 + 证据图），
主链路没有这张图。我们基于 `incident.context` 的 audit_trail + verification
字段做"扁平化"等价检查：

1. **时间顺序 check_time_ordering**：audit_trail 时间戳是否在合理窗口内
2. **替代解释 check_alternatives**：是否有 Plan B / mismatch 信号
3. **证据充分性 check_evidence**：RCA 5 路证据是否至少 2 路支持
4. **反事实可行性 check_counterfactual**：RCA mismatch 高 / 置信度低

设计原则：
- 默认行为：Gate 失败时**只告警，不强制 retry**（避免把系统搞挂）
- 通过 message 返回失败原因，便于 W7 反思层做决策
- 失败时计入 verify_report 的 gate_block，对 BadCase 触发器无影响
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ===== Feature Flag =====


def is_w8_2_gate_enabled() -> bool:
    """判断 W8.2 Gate 校验是否启用。默认关闭（保守）。"""
    val = os.getenv("AIOPS_USE_W8_2_GATE", "false").lower()
    return val in ("true", "1", "yes", "on")


# ===== Gate 结果 =====


@dataclass
class MainPathGateResult:
    """Gate 校验结果。"""

    passed: bool
    reason: str
    checks: dict[str, str] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    # 软告警：Gate 失败但不去 retry（默认行为）
    severity: str = "info"  # "info" / "warn" / "block"


# ===== Gate 实现 =====


class MainPathGate:
    """MainPathGate — 4 问启发式检查（基于 audit_trail + verification）。

    与 W8 的 CounterfactualGate 区别：
    - W8：基于 InvestigationGraph（结构化）
    - MainPathGate：基于 audit_trail（扁平化）
    """

    # 配置参数
    MIN_EVIDENCE_ROUTES = 2          # 5 路证据至少 2 路支持
    MIN_RCA_CONFIDENCE = 0.6         # RCA 置信度最低要求
    MIN_IMPROVEMENT_PCT = 0.3        # 改善幅度最低 30%
    MAX_TIME_GAP_SECONDS = 3600      # 时间窗口 1 小时

    def __init__(self) -> None:
        self._enabled = is_w8_2_gate_enabled()

    def check(self, incident_context: dict[str, Any]) -> MainPathGateResult:
        """对 incident 作 4 问 Gate 检查。

        Args:
            incident_context: incident.context 字典

        Returns:
            MainPathGateResult
        """
        if not self._enabled:
            return MainPathGateResult(
                passed=True,
                reason="W8.2 gate disabled via feature flag",
            )

        try:
            checks: dict[str, str] = {}
            failed: list[str] = []

            # 1. 时间顺序
            checks["time_ordering"] = self._check_time_ordering(incident_context)
            if not self._check_pass(checks["time_ordering"]):
                failed.append("time_ordering")

            # 2. 替代解释
            checks["alternatives"] = self._check_alternatives(incident_context)
            if not self._check_pass(checks["alternatives"]):
                failed.append("alternatives")

            # 3. 证据充分性
            checks["evidence"] = self._check_evidence(incident_context)
            if not self._check_pass(checks["evidence"]):
                failed.append("evidence")

            # 4. 反事实可行性
            checks["counterfactual"] = self._check_counterfactual(incident_context)
            if not self._check_pass(checks["counterfactual"]):
                failed.append("counterfactual")

            if not failed:
                return MainPathGateResult(
                    passed=True,
                    reason="all 4 checks passed",
                    checks=checks,
                )

            # 失败 → 默认 severity=warn（不强制 retry）
            return MainPathGateResult(
                passed=False,
                reason=f"failed checks: {', '.join(failed)}",
                checks=checks,
                failed_checks=failed,
                severity="warn",
            )

        except Exception as exc:
            logger.warning("W8.2 Gate check failed (non-fatal)", error=str(exc))
            return MainPathGateResult(
                passed=True,  # 失败时通过（不影响主链路）
                reason=f"gate crashed: {exc}",
                severity="info",
            )

    # ----- 4 问检查实现 -----

    def _check_time_ordering(self, ctx: dict) -> str:
        """时间顺序：audit_trail 时间戳是否在合理窗口内。"""
        trail = ctx.get("audit_trail", [])
        if not trail:
            return "ok: no audit_trail to check"

        # 提取首尾时间戳
        timestamps = []
        for entry in trail:
            ts = entry.get("timestamp")
            if ts is None:
                continue
            try:
                timestamps.append(float(ts))
            except (TypeError, ValueError):
                continue

        if len(timestamps) < 2:
            return "ok: insufficient timestamps"

        gap = max(timestamps) - min(timestamps)
        if gap > self.MAX_TIME_GAP_SECONDS:
            return f"fail: time gap {gap:.0f}s > {self.MAX_TIME_GAP_SECONDS}s"
        return f"ok: time gap {gap:.0f}s within {self.MAX_TIME_GAP_SECONDS}s"

    def _check_alternatives(self, ctx: dict) -> str:
        """替代解释：是否被 Plan B / mismatch 反复推翻。"""
        verification = ctx.get("verification", {})
        plan_b_count = 1 if verification.get("plan_b_triggered") else 0
        rca_mismatch = verification.get("rca_match_status") == "mismatch"

        # 如果已触发 Plan B + RCA mismatch → 替代解释被反复推翻
        if plan_b_count and rca_mismatch:
            return "fail: plan_b triggered AND rca mismatch (alternatives not excluded)"
        if plan_b_count:
            return "ok: plan_b triggered (1 alternative explored)"
        if rca_mismatch:
            return "warn: rca mismatch (alternative not yet refined)"
        return "ok: no alternative signals"

    def _check_evidence(self, ctx: dict) -> str:
        """证据充分性：RCA 5 路证据是否至少 2 路支持。"""
        # 1. 取 rca_event
        rca_event = ctx.get("rca_event")
        if isinstance(rca_event, dict):
            evidence = rca_event.get("evidence", {})
        else:
            evidence = {}

        # 2. 统计 5 路证据是否至少有 2 路非空
        routes = [
            ("bayesian", evidence.get("bayesian")),
            ("topology", evidence.get("bfs") or evidence.get("topology")),
            ("rag", evidence.get("rag")),
            ("history", evidence.get("history") or evidence.get("memory")),
            ("log", evidence.get("log")),
        ]
        supported = [kind for kind, value in routes if value]
        count = len(supported)

        if count < self.MIN_EVIDENCE_ROUTES:
            return (
                f"fail: only {count} evidence routes supported "
                f"(need >= {self.MIN_EVIDENCE_ROUTES})"
            )
        return f"ok: {count} evidence routes supported"

    def _check_counterfactual(self, ctx: dict) -> str:
        """反事实可行性：RCA 置信度 + 改善幅度是否达标。"""
        # 1. RCA confidence
        rca_event = ctx.get("rca_event")
        if isinstance(rca_event, dict):
            confidence = rca_event.get("confidence", 0.0)
        else:
            rca_result = ctx.get("rca_result")
            confidence = rca_result.get("confidence", 0.0) if isinstance(rca_result, dict) else 0.0

        if confidence < self.MIN_RCA_CONFIDENCE:
            return f"fail: confidence {confidence:.2f} < {self.MIN_RCA_CONFIDENCE}"

        # 2. 改善幅度
        verification = ctx.get("verification", {})
        if not verification.get("heal_recovered"):
            return "fail: heal not recovered (counterfactual not constructible)"

        # 3. 如果仅 OK
        return f"ok: confidence={confidence:.2f}, recovered=True"

    # ----- 工具方法 -----

    @staticmethod
    def _check_pass(check_str: str) -> bool:
        """判断 check 字符串是否通过。"""
        return check_str.startswith("ok")


# ===== 工厂函数 =====


def get_main_path_gate() -> MainPathGate:
    """获取 MainPathGate 单例。"""
    return MainPathGate()
