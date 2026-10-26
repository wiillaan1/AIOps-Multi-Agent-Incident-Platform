"""
Review 数据模型
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ReviewDecision(str, Enum):
    """人评决策（3 类）"""
    ACCEPT = "accept"                      # 接受
    REJECT = "reject"                      # 拒绝（记入 badcase）
    NEEDS_IMPROVEMENT = "needs_improvement"  # 接受但需改进


class ReviewDimension(str, Enum):
    """评分维度"""
    ACCURACY = "accuracy"            # 根因准确度
    EXPLAINABILITY = "explainability"  # 推理可解释性
    ACTIONABILITY = "actionability"  # 修复建议有效性
    OVERALL = "overall"              # 总评


# 默认 4 维度（1-5 分）
DEFAULT_DIMENSIONS = [
    ReviewDimension.ACCURACY,
    ReviewDimension.EXPLAINABILITY,
    ReviewDimension.ACTIONABILITY,
    ReviewDimension.OVERALL,
]


@dataclass
class Review:
    """单条人评记录"""
    id: int | None = None
    incident_id: str = ""          # 关联事故 ID（与 badcase 对应）
    badcase_id: str = ""           # 关联 badcase ID（可选）
    reviewer: str = ""             # oncall 姓名
    decision: str = ""             # ReviewDecision.value
    scores: dict[str, int] = field(default_factory=dict)  # {dim: score}
    comments: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "incident_id": self.incident_id,
            "badcase_id": self.badcase_id,
            "reviewer": self.reviewer,
            "decision": self.decision,
            "scores": self.scores,
            "comments": self.comments,
            "timestamp": self.timestamp,
        }

    def avg_score(self) -> float:
        """平均分（0-5）"""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def is_passing(self, threshold: float = 3.0) -> bool:
        """是否通过（overall ≥ threshold 且 decision != reject）"""
        overall = self.scores.get(ReviewDimension.OVERALL.value, 0)
        return self.decision != ReviewDecision.REJECT.value and overall >= threshold
