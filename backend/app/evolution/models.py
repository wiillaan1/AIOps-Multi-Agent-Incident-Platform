"""
Evolution 数据模型
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EvolutionStage(str, Enum):
    """自进化 5 阶段"""
    BADCASE_COLLECTED = "badcase_collected"
    PATTERN_ANALYZED = "pattern_analyzed"
    PROMPT_GENERATED = "prompt_generated"
    AB_TESTED = "ab_tested"
    REVIEWED = "reviewed"


class PromotionDecision(str, Enum):
    """promote 决策"""
    PENDING = "pending"           # 待人工决定
    PROMOTED = "promoted"         # 已采纳
    REJECTED = "rejected"         # 已拒绝
    AUTO_ACCEPTED = "auto_accepted"  # 自动采纳（pass_rate 提升 ≥ 阈值）


@dataclass
class PromptVersion:
    """Prompt 版本（每次 prompt 改动是一行记录）"""
    id: int | None = None
    version: str = ""                    # "v1.0.0", "v1.0.1"...
    parent_version: str = ""             # 基于哪个版本改的
    prompt_segment: str = ""             # prompt 段内容
    target_dimension: str = ""           # 修复哪个评估维度（accuracy/citation/hallucination）
    badcase_ids: list[str] = field(default_factory=list)
    cycle_id: int | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ab_test_results: dict[str, Any] = field(default_factory=dict)
    # AB 测试结果：{old_pass_rate, new_pass_rate, delta, regression_check}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "parent_version": self.parent_version,
            "prompt_segment": self.prompt_segment,
            "target_dimension": self.target_dimension,
            "badcase_ids": self.badcase_ids,
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp,
            "ab_test_results": self.ab_test_results,
        }


@dataclass
class EvolutionCycle:
    """一次完整的自进化循环"""
    id: int | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None

    # 阶段结果
    badcase_ids: list[str] = field(default_factory=list)
    pattern_analysis: dict[str, Any] = field(default_factory=dict)
    generated_prompt: str = ""
    ab_test_summary: dict[str, Any] = field(default_factory=dict)
    promotion_decision: str = PromotionDecision.PENDING.value
    reviewer: str = ""
    review_notes: str = ""

    current_stage: str = EvolutionStage.BADCASE_COLLECTED.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "badcase_ids": self.badcase_ids,
            "pattern_analysis": self.pattern_analysis,
            "generated_prompt": self.generated_prompt,
            "ab_test_summary": self.ab_test_summary,
            "promotion_decision": self.promotion_decision,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "current_stage": self.current_stage,
        }
