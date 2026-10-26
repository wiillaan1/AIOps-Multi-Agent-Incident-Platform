"""
Review 模块 - 人评机制（oncall 评分）

设计目标：
- oncall 收到 badcase 通知后能快速评分
- 评分数据进入 ground truth 反哺评估集
- 评分行为写审计（AuditAction.REVIEW_SUBMITTED）

入口：
- CLI: python -m app.review {list|show|submit|stats}
- 编程式: from app.review import ReviewStore, Review, submit_review

评分维度（4 项，每项 1-5 分）：
1. accuracy: 根因准确度
2. explainability: 推理可解释性
3. actionability: 修复建议有效性
4. overall: 总评

决策（3 类）：
- accept: 接受，作为 ground truth
- reject: 拒绝，记入 badcase 库
- needs_improvement: 接受但需要改进
"""
from app.review.models import (
    Review,
    ReviewDecision,
    ReviewDimension,
    DEFAULT_DIMENSIONS,
)
from app.review.store import (
    ReviewStore,
    get_review_store,
    reset_review_store,
)
from app.review.cli import main

__all__ = [
    "Review",
    "ReviewDecision",
    "ReviewDimension",
    "DEFAULT_DIMENSIONS",
    "ReviewStore",
    "get_review_store",
    "reset_review_store",
    "main",
]
