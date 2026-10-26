"""W8.3 — DecisionContext.

主链路 4 stage (Monitor → RCA → Heal → Change) 之间共享的决策上下文。

设计原则：
- 不破坏 Agent 接口（每个 Agent 仍接收自己的 input + context）
- DecisionContext 通过 `incident.context["decision_ctx"]` 字段共享
- 各 stage 写入 / 读取时记录 refinement_history（"谁、何时、改了什么"）
- 默认关闭（feature flag），关闭 = 退回到 W7 行为
- 不影响 W7 反思 / W7.5 BadCase 触发器

字段分类：
- feature_flags: 4 个 stage 都可以读取的 feature 开关
- resource_limits: 资源限制（CPU / 内存 / 调外部 API 配额）
- version_info: 各 Agent 版本（用于 A/B 测试 / Harness 评测）
- cross_stage_decisions: stage A 留给 stage B 的决策（例：RCA 留 hint 给 Heal）
- refinement_history: 每次 set 都记录（用于审计）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


# ===== Feature Flag =====


def is_w8_3_context_enabled() -> bool:
    """判断 W8.3 DecisionContext 注入是否启用。默认关闭。"""
    val = os.getenv("AIOPS_USE_W8_3_CONTEXT", "false").lower()
    return val in ("true", "1", "yes", "on")


# ===== Refinement History 项 =====


@dataclass
class RefinementEntry:
    """一次 DecisionContext 修改的审计记录。"""

    key: str
    value: Any
    stage: str
    timestamp: str
    reason: str = ""


# ===== Decision Context =====


@dataclass
class DecisionContext:
    """跨 stage 共享的决策上下文。

    字段说明：
    - feature_flags: 4 个 stage 都可读的 feature 开关
    - resource_limits: 资源限制（防过度调外部 API）
    - version_info: Agent 版本（用于 A/B）
    - cross_stage_decisions: A 留给 B 的决策 hint
    - refinement_history: 修改审计
    """

    feature_flags: dict[str, bool] = field(default_factory=dict)
    resource_limits: dict[str, float] = field(default_factory=dict)
    version_info: dict[str, str] = field(default_factory=dict)
    cross_stage_decisions: dict[str, Any] = field(default_factory=dict)
    refinement_history: list[dict] = field(default_factory=list)

    def set(
        self,
        category: str,
        key: str,
        value: Any,
        stage: str,
        reason: str = "",
        timestamp: str = "",
    ) -> None:
        """设置字段值（带审计）。

        Args:
            category: feature_flags / resource_limits / version_info / cross_stage_decisions
            key: 字段名
            value: 字段值
            stage: 哪个 stage 写的（monitor / rca / heal / change）
            reason: 修改原因
            timestamp: 时间戳（默认当前时间）
        """
        if not timestamp:
            from datetime import datetime, timezone
            timestamp = datetime.now(timezone.utc).isoformat()

        if category not in {
            "feature_flags", "resource_limits", "version_info", "cross_stage_decisions",
        }:
            raise ValueError(f"unknown category: {category}")

        target = getattr(self, category)
        target[key] = value

        self.refinement_history.append({
            "category": category,
            "key": key,
            "value": value,
            "stage": stage,
            "reason": reason,
            "timestamp": timestamp,
        })

    def get(self, category: str, key: str, default: Any = None) -> Any:
        """读取字段值。"""
        target = getattr(self, category, None)
        if not isinstance(target, dict):
            return default
        return target.get(key, default)

    def get_refinements(self, key: str | None = None) -> list[dict]:
        """获取 refinement 历史（可选按 key 过滤）。"""
        if key is None:
            return list(self.refinement_history)
        return [r for r in self.refinement_history if r["key"] == key]

    def to_dict(self) -> dict:
        """序列化为 dict（用于 incident.context 存储）。"""
        return {
            "feature_flags": dict(self.feature_flags),
            "resource_limits": dict(self.resource_limits),
            "version_info": dict(self.version_info),
            "cross_stage_decisions": dict(self.cross_stage_decisions),
            "refinement_history": list(self.refinement_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionContext":
        """从 dict 还原（用于 incident.context 反序列化）。"""
        return cls(
            feature_flags=data.get("feature_flags", {}),
            resource_limits=data.get("resource_limits", {}),
            version_info=data.get("version_info", {}),
            cross_stage_decisions=data.get("cross_stage_decisions", {}),
            refinement_history=data.get("refinement_history", []),
        )


# ===== Stage 适配器 =====


class StageAdapter:
    """4 stage 共用：每个 stage 写入 / 读取 DecisionContext。

    用法：
        adapter = StageAdapter(incident.context, stage="rca")
        adapter.set_feature_flag("use_w8_log", True, reason="log rich incident")
        adapter.set_cross_stage("heal_hint", "database_issue", reason="rca said DB")
    """

    def __init__(self, incident_context: dict, stage: str) -> None:
        self._ctx = incident_context
        self._stage = stage
        self._enabled = is_w8_3_context_enabled()
        # 懒加载 DecisionContext
        self._dc = self._get_or_create()

    def _get_or_create(self) -> DecisionContext:
        """从 incident.context 还原 DecisionContext，不存在则新建。"""
        existing = self._ctx.get("decision_ctx")
        if isinstance(existing, dict):
            return DecisionContext.from_dict(existing)
        return DecisionContext()

    def set_feature_flag(
        self, key: str, value: bool, reason: str = ""
    ) -> None:
        """设置 feature_flag。"""
        if not self._enabled:
            return
        self._dc.set("feature_flags", key, value, self._stage, reason=reason)
        self._persist()

    def set_resource_limit(
        self, key: str, value: float, reason: str = ""
    ) -> None:
        """设置 resource_limit。"""
        if not self._enabled:
            return
        self._dc.set("resource_limits", key, value, self._stage, reason=reason)
        self._persist()

    def set_version(self, agent: str, version: str, reason: str = "") -> None:
        """设置 agent 版本。"""
        if not self._enabled:
            return
        self._dc.set("version_info", agent, version, self._stage, reason=reason)
        self._persist()

    def set_cross_stage(
        self, key: str, value: Any, reason: str = ""
    ) -> None:
        """A stage 留给 B stage 的决策 hint。"""
        if not self._enabled:
            return
        self._dc.set("cross_stage_decisions", key, value, self._stage, reason=reason)
        self._persist()

    def get_feature_flag(self, key: str, default: bool = False) -> bool:
        """读 feature_flag。"""
        return self._dc.get("feature_flags", key, default)

    def get_cross_stage(self, key: str, default: Any = None) -> Any:
        """读 cross_stage_decisions。"""
        return self._dc.get("cross_stage_decisions", key, default)

    def get_refinements(self, key: str | None = None) -> list[dict]:
        """读 refinement_history。"""
        return self._dc.get_refinements(key)

    def _persist(self) -> None:
        """把 DecisionContext 写回 incident.context。"""
        if not self._enabled:
            return
        self._ctx["decision_ctx"] = self._dc.to_dict()


# ===== 工厂函数 =====


def create_stage_adapter(incident_context: dict, stage: str) -> StageAdapter:
    """为指定 stage 创建一个 StageAdapter。

    Args:
        incident_context: incident.context 字典
        stage: 哪个 stage（monitor / rca / heal / change）

    Returns:
        StageAdapter 实例（即使 feature flag 关闭也能安全使用，只是不写）
    """
    return StageAdapter(incident_context, stage)
