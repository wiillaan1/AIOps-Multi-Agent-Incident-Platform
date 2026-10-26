"""BadCaseEntry model — W7.5 data flywheel + Level 3 unified eval schema.

[v3 统一改造 E1] 2026-09-01
合并了原 eval/dataset.py 的 Badcase 字段到本模型：
- alert / evidence / expected_root_cause / expected_confidence_min / category / type / tags

这样：
- badcase_capture 写 candidates.jsonl 用 BadCaseEntry
- 评估集 v1_dev.json / v1_test.json / v1_holdout.json 也用 BadCaseEntry
- RCAEvaluator 接收 BadCaseEntry
- 闭环：捕获 → 评估 → 改进 → 回归，一套 schema 走到底

向后兼容：
- 原 BadCaseEntry caller（registry / capture / harness / routes）继续 work
- 原 Badcase 别名指向 BadCaseEntry（eval/evaluator.py 继续 work）
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import json

from pydantic import BaseModel, Field, field_validator


class BadCaseSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvalSplit(str, Enum):
    DEV = "dev"
    TEST = "test"
    HOLDOUT = "holdout"  # [v3 新增] 月度复测，防过拟合


# 数据来源（哪个适配器生成的）
BadcaseSource = Literal[
    "manual",           # 手工编写
    "auto_capture",     # 生产 badcase_capture 自动捕获
    "aiops2020",        # 清华 AIOps Challenge 2020
    "loghub_hdfs",      # Loghub HDFS_v1
    "smd",              # Server Machine Dataset (OmniAnomaly)
    "aiopslab",         # Microsoft AIOpsLab
    "regression_guard", # 历史已修问题回归测试
]


class BadCaseEntry(BaseModel):
    """统一的 badcase schema —— 既是评估集，也是捕获草稿，也是 dev/test 存储。"""

    # ===== 标识 =====
    id: str = Field(pattern=r"^bc-[a-zA-Z0-9_\-]{2,80}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: BadcaseSource = "manual"
    eval_split: EvalSplit = EvalSplit.DEV

    # ===== 告警与证据 =====
    alert: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)

    # ===== 评估期望（机器可验） =====
    category: Literal["single_metric", "compound", "regression_guard"] = "single_metric"
    type: str = "unknown"  # cpu_high / db_timeout / network_drop ...
    # 捕获草稿可能尚未完成真实根因标注；promote 前必须补齐人工标注。
    expected_root_cause: str = Field(default="unknown", min_length=1)
    expected_confidence_min: float = Field(default=0.5, ge=0.0, le=1.0)

    # ===== 元数据 =====
    severity: BadCaseSeverity = BadCaseSeverity.MEDIUM
    tags: list[str] = Field(default_factory=list, max_length=20)
    description: str = ""

    # ===== 缺陷描述（捕获/适配器生成） =====
    badcase_class: str = Field(default="unknown", min_length=3, max_length=64)
    identified_flaw: str = Field(default="", max_length=2000)
    keywords_for_retrieval: list[str] = Field(default_factory=list, max_length=20)
    suggestion_or_lesson: str = Field(default="", max_length=2000)
    audit_trail_excerpt: list[dict] = Field(default_factory=list, max_length=20)

    # ===== 修复追踪 =====
    fixed_in: str | None = None
    source_ref: str | None = None  # e.g. "aiops2020/case_001", "loghub/HDFS_v1/block_123"

    @field_validator("keywords_for_retrieval")
    @classmethod
    def _lower_keywords(cls, v: list[str]) -> list[str]:
        return [k.lower().strip() for k in v if k.strip()]

    @field_validator("id")
    @classmethod
    def _validate_id_prefix(cls, v: str) -> str:
        if not v.startswith("bc-"):
            raise ValueError(f"id must start with 'bc-', got: {v}")
        return v

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BadCaseEntry":
        """从 dict 创建（向后兼容原 eval/dataset.Badcase.from_dict）。

        处理几种特殊情形：
        - id 缺失 → 自动生成
        - 字段缺失 → 用默认值
        - 时间戳格式不一致 → 容错
        """
        d = dict(data)
        if "id" not in d:
            # 自动生成稳定 id（用 alert + expected_root_cause 哈希）
            seed = json.dumps(d.get("alert", {}), sort_keys=True) + d.get("expected_root_cause", "")
            import hashlib as _hl
            d["id"] = "bc-" + _hl.sha256(seed.encode()).hexdigest()[:12]
        # 缺失字段用默认值
        d.setdefault("alert", {})
        d.setdefault("evidence", {})
        d.setdefault("expected_root_cause", "unknown")
        d.setdefault("expected_confidence_min", 0.5)
        d.setdefault("category", "single_metric")
        d.setdefault("type", "unknown")
        return cls(**d)


# ===== 向后兼容别名 =====
# 原 eval/dataset.py 用的 Badcase —— 现在统一指向 BadCaseEntry
Badcase = BadCaseEntry
