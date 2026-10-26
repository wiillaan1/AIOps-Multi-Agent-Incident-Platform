"""
RuntimePatternStore — Harness promote 后沉淀的运行时 pattern

为什么需要：
- `data/knowledge_base.py` 的 KNOWLEDGE_BASE 是静态种子数据，
  不接受运行时追加（Harness champion-challenger 升级后的 pattern 没有去处）。
- `HarnessLoopEngine.promote()` 只写 `champion.json`，没把经验沉淀到可检索的知识层
  （审查报告七、长期演 #14：飞轮"知识层"断链）。
- LTM（LongTermMemory）已经存在，但 Champion 是"候选修复的胜出者"，
  是结构化的"what fixed what"，跟 LTM 的 lesson 维度不同，需要单独的 store。

设计：
- append-only JSONL（与 `data/audit.db` 同样风格：线程安全 + 不暴露 UPDATE/DELETE）
- RuntimePattern：{id, badcase_class, source_run_id, pattern_type, fix_summary,
                     trigger_signature, metrics, promoted_at}
- query_by_class(badcase_class)：召回某类故障的 pattern（用于未来 RCA 召回）
- query_recent(top_k)：最近 N 条

这是新增模块，不修改现有 `data/knowledge_base.py`，避免破坏静态种子数据
（1131 个测试依赖 KNOWLEDGE_BASE 静态内容）。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger(__name__)


DEFAULT_PATTERN_STORE_PATH = os.getenv(
    "RUNTIME_PATTERN_STORE_PATH",
    str(Path(__file__).resolve().parents[3] / "data" / "runtime_patterns.jsonl"),
)


@dataclass(frozen=True)
class RuntimePattern:
    """一条从 Harness promote 抽取出来的运行时 pattern"""

    id: str = field(default_factory=lambda: f"pat-{uuid.uuid4().hex[:12]}")
    badcase_class: str = ""                       # e.g. "memory_oom_high"
    source_run_id: str = ""                       # HarnessRunRecord.run_id
    source_champion_version: str = ""             # Champion.version
    pattern_type: str = ""                        # e.g. "configuration_change" / "logic_fix"
    fix_summary: str = ""                         # 人/规则可读摘要
    trigger_signature: str = ""                   # 触发该 pattern 的特征（如 metric 名 + 阈值）
    evidence_refs: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # champion 胜出时的指标
    promoted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    promoted_by: str = "harness:promote"           # 来源标注

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimePatternStore:
    """运行时 pattern 持久化（append-only JSONL）"""

    def __init__(self, store_path: str = DEFAULT_PATTERN_STORE_PATH):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # 不需要 schema 初始化（JSONL 自描述），但要确保文件存在
        if not self.store_path.exists():
            self.store_path.touch(mode=0o644)

    def _read_all(self) -> list[dict[str, Any]]:
        """读取所有 pattern（按行解析，跳过坏行）"""
        if not self.store_path.exists():
            return []
        items: list[dict[str, Any]] = []
        with self.store_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"[RuntimePatternStore] 第 {lineno} 行 JSON 损坏，跳过: {e}")
        return items

    def add(self, pattern: RuntimePattern) -> str:
        """追加一条 pattern（fire-and-forget：失败不抛异常）

        返回写入的 pattern id（写入失败返回空字符串）
        """
        try:
            with self._lock:
                with self.store_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(pattern.to_dict(), ensure_ascii=False, default=str) + "\n")
            return pattern.id
        except Exception as e:
            logger.warning(f"[RuntimePatternStore] 追加失败（不影响主链路）: {e}")
            return ""

    def query_by_class(
        self,
        badcase_class: str,
        top_k: int = 5,
    ) -> list[RuntimePattern]:
        """按 badcase_class 召回 pattern（按 promoted_at DESC 排序）

        注意：当前没有向量检索，是精确匹配；后续如果 pattern_type 多样化，
        可以叠加 chromadb 向量召回。
        """
        all_items = self._read_all()
        matched = [p for p in all_items if p.get("badcase_class") == badcase_class]
        matched.sort(key=lambda p: p.get("promoted_at", ""), reverse=True)
        return [self._from_dict(p) for p in matched[:top_k]]

    def query_recent(self, top_k: int = 10) -> list[RuntimePattern]:
        """最近 N 条 pattern（按 promoted_at DESC）"""
        all_items = self._read_all()
        all_items.sort(key=lambda p: p.get("promoted_at", ""), reverse=True)
        return [self._from_dict(p) for p in all_items[:top_k]]

    def count(self) -> int:
        """总条数（用于监控/调试）"""
        return len(self._read_all())

    @staticmethod
    def _from_dict(d: dict[str, Any]) -> RuntimePattern:
        """从 dict 还原 RuntimePattern（跳过不识别的字段）"""
        valid_fields = {f for f in RuntimePattern.__dataclass_fields__.keys()}
        return RuntimePattern(**{k: v for k, v in d.items() if k in valid_fields})


# ---------- 全局单例 ----------
_store: RuntimePatternStore | None = None


def get_runtime_pattern_store(store_path: str | None = None) -> RuntimePatternStore:
    """获取全局 RuntimePatternStore（懒加载）"""
    global _store
    if _store is None:
        _store = RuntimePatternStore(store_path or DEFAULT_PATTERN_STORE_PATH)
    return _store


def reset_runtime_pattern_store() -> None:
    """重置（用于测试或热更新）"""
    global _store
    _store = None
