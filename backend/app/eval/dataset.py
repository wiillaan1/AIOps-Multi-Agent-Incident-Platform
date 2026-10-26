"""
BadcaseDataset - 测试集加载

[v3 统一改造 E1 + E3] 2026-09-01
- 单一 schema: BadCaseEntry（与 badcase_capture / harness_engine 共用）
- 三分离: v1_dev.json / v1_test.json / v1_holdout.json
- 默认路径: ./data/badcases/v1.json (向后兼容) → 自动按 eval_split 分流

设计原则：
- v1.json (兼容旧路径) 仍可加载
- 推荐使用 v1_dev.json / v1_test.json / v1_holdout.json
- get_default_dataset(split="dev") 显式指定 split
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from app.models.badcase import BadCaseEntry, EvalSplit


logger = logging.getLogger(__name__)


# 保持向后兼容：Badcase = BadCaseEntry（已移到 models/badcase.py）
Badcase = BadCaseEntry


@dataclass
class BadcaseDataset:
    """完整测试集（按 split 组织）"""

    version: str
    description: str
    badcases: list[BadCaseEntry]
    metadata: dict[str, Any] = field(default_factory=dict)
    eval_split: Literal["dev", "test", "holdout"] | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        eval_split: Literal["dev", "test", "holdout"] | None = None,
    ) -> "BadcaseDataset":
        """加载数据集。

        路径策略：
        1. 显式传入 path → 用 path
        2. 路径含 `{split}` → 用 eval_split 替换（默认 dev）
        """
        path = Path(path)
        if "{split}" in str(path) and eval_split:
            path = Path(str(path).replace("{split}", eval_split))
        elif "{split}" in str(path):
            path = Path(str(path).replace("{split}", "dev"))

        if not path.exists():
            raise FileNotFoundError(f"Badcase 数据集不存在: {path}")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("_meta", {})
        # 兼容两种格式：顶层 badcases list / dict 按 split 分
        if isinstance(data.get("badcases"), dict):
            # 多 split 文件
            split_data = data["badcases"]
            if eval_split:
                raw = split_data.get(eval_split, [])
            else:
                # 合并所有 split
                raw = []
                for s, items in split_data.items():
                    raw.extend(items)
                eval_split = "all"
        else:
            raw = data.get("badcases", [])

        badcases = []
        for b in raw:
            # 兼容两种 id 字段
            if "id" not in b and "entry_id" in b:
                b["id"] = b.pop("entry_id")
            # 兼容旧的 created_at 缺失
            if "created_at" not in b:
                b["created_at"] = datetime.utcnow().isoformat()
            # 兼容字段名
            if "audit_trail_excerpt" not in b:
                b["audit_trail_excerpt"] = []
            try:
                badcases.append(BadCaseEntry(**b))
            except Exception as e:
                logger.warning("跳过非法 badcase entry %s: %s", b.get("id"), e)

        return cls(
            version=meta.get("version", "v1"),
            description=meta.get("description", ""),
            badcases=badcases,
            metadata=meta,
            eval_split=eval_split,
        )

    def by_category(self, category: str) -> list[BadCaseEntry]:
        return [b for b in self.badcases if b.category == category]

    def by_source(self, source: str) -> list[BadCaseEntry]:
        return [b for b in self.badcases if b.source == source]

    def by_severity(self, severity: str) -> list[BadCaseEntry]:
        return [b for b in self.badcases if b.severity == severity]

    def stats(self) -> dict[str, Any]:
        """数据集统计"""
        from collections import Counter

        cats = Counter(b.category for b in self.badcases)
        types = Counter(b.type for b in self.badcases)
        sources = Counter(b.source for b in self.badcases)
        severities = Counter(b.severity for b in self.badcases)
        return {
            "total": len(self.badcases),
            "by_category": dict(cats),
            "by_type": dict(types),
            "by_source": dict(sources),
            "by_severity": dict(severities),
            "version": self.version,
            "eval_split": self.eval_split,
        }


# ===== 默认路径 =====

# 项目根: backend/app/eval/dataset.py → parents[3] = aiops-agent-platform/
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BASE_DIR = _PROJECT_ROOT / "data" / "badcases"

# 单文件兼容路径（旧）
_DEFAULT_PATH = _BASE_DIR / "v1.json"

# 三分离路径（新）
_DEV_PATH = _BASE_DIR / "v1_dev.json"
_TEST_PATH = _BASE_DIR / "v1_test.json"
_HOLDOUT_PATH = _BASE_DIR / "v1_holdout.json"


def get_default_dataset(
    eval_split: Literal["dev", "test", "holdout"] | None = None,
) -> BadcaseDataset:
    """加载默认数据集。

    策略：
    1. 如果指定 eval_split，优先用 v1_{split}.json
    2. 否则 fallback 到 v1.json
    3. 都没有 → FileNotFoundError（让调用方决定怎么处理）
    """
    if eval_split:
        split_path = _BASE_DIR / f"v1_{eval_split}.json"
        if split_path.exists():
            return BadcaseDataset.load(split_path, eval_split=eval_split)

    if _DEFAULT_PATH.exists():
        return BadcaseDataset.load(_DEFAULT_PATH, eval_split=eval_split)

    raise FileNotFoundError(
        f"Badcase 数据集不存在。请运行数据适配器生成：\n"
        f"  python -m data.adapters.generate_v1\n"
        f"或参考：\n"
        f"  data/adapters/aiops2020_to_badcase.py\n"
        f"  data/adapters/loghub_hdfs_to_badcase.py\n"
        f"  data/adapters/smd_to_badcase.py"
    )


def get_split_path(eval_split: Literal["dev", "test", "holdout"]) -> Path:
    """获取指定 split 的文件路径（用于适配器生成）"""
    return _BASE_DIR / f"v1_{eval_split}.json"
