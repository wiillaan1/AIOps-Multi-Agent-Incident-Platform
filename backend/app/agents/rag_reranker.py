"""
AIOps Agent Platform - RAG Re-ranker (Phase 2c)

对 ChromaDB 召回的 top-K 候选做二次重排，提升召回质量。

工作流：
  ChromaDB 召回 (top 20)
     ↓
  LLMReranker.rerank()   ← LLM 按 query 相关性打分
     ↓
  保留 top_k (默认 3)
     ↓
  写 audit_trail（rerank 前后变化）

设计原则：
1. **不调 LLM 也可用**：RuleOnlyProvider 退回到原顺序（ChromaDB distance 排序）
2. **超时降级**：3 秒超时或异常时返回原顺序
3. **可审计**：每次重排记录 query / 候选数 / 排序前后变化
4. **可选启用**：可配置 `enable=False` 走原顺序
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RankedItem:
    """重排后的单条结果。"""

    doc_id: str
    original_rank: int      # ChromaDB 原始排名（0-indexed）
    new_rank: int           # 重排后排名（0-indexed）
    original_score: float   # 原始距离（越小越相关）
    rerank_score: float     # LLM 给的相关性分数（0~1）
    content_snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "original_rank": self.original_rank,
            "new_rank": self.new_rank,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "content_snippet": self.content_snippet,
            "metadata": self.metadata,
        }


@dataclass
class RerankResult:
    """重排结果。"""

    query: str
    top_k: int
    total_candidates: int
    items: list[RankedItem]
    rerank_changed: bool        # 排序是否发生变化
    llm_used: bool
    latency_ms: int
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "total_candidates": self.total_candidates,
            "items": [i.to_dict() for i in self.items],
            "rerank_changed": self.rerank_changed,
            "llm_used": self.llm_used,
            "latency_ms": self.latency_ms,
            "fallback_reason": self.fallback_reason,
        }


class LLMReranker:
    """基于 LLM 的 RAG 二次重排。"""

    def __init__(
        self,
        llm_provider=None,
        timeout_seconds: float = 3.0,
        enable: bool = True,
    ) -> None:
        from app.nlu.intent_llm import LLMProvider, RuleOnlyProvider
        self.llm_provider = llm_provider or RuleOnlyProvider()
        self.timeout_seconds = timeout_seconds
        self.enable = enable

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
    ) -> RerankResult:
        """
        Args:
            query: 用户原始 query
            candidates: ChromaDB 召回的候选 list，每个含 {id, content, metadata, distance}
            top_k: 最终保留条数
        """
        started = time.time()
        result = RerankResult(
            query=query,
            top_k=top_k,
            total_candidates=len(candidates),
            items=[],
            rerank_changed=False,
            llm_used=False,
            latency_ms=0,
        )

        if not candidates:
            result.latency_ms = int((time.time() - started) * 1000)
            return result

        # 1. 先按 ChromaDB 原顺序排好（保底）
        original_order = self._build_ranked_items_from_candidates(candidates)

        if not self.enable or self.llm_provider.name == "rule_only":
            # 不调 LLM → 直接返回原顺序
            result.items = original_order[:top_k]
            result.latency_ms = int((time.time() - started) * 1000)
            return result

        # 2. 调 LLM 重排
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            llm_resp = loop.run_until_complete(
                asyncio.wait_for(
                    self.llm_provider.classify(
                        query=query,
                        rule_intent="rag_rerank",
                        rule_confidence=1.0,
                        rule_clarification_questions=[],
                    ),
                    timeout=self.timeout_seconds,
                )
            )
            loop.close()
        except Exception as exc:
            logger.warning("RAG rerank LLM failed", error=str(exc))
            result.items = original_order[:top_k]
            result.fallback_reason = f"llm_failed: {exc}"
            result.latency_ms = int((time.time() - started) * 1000)
            return result

        # 3. 解析 LLM 输出（约定：reasoning 是 JSON list of doc_ids）
        new_order_ids = self._parse_llm_output(llm_resp.reasoning if llm_resp else "")
        if not new_order_ids:
            result.items = original_order[:top_k]
            result.fallback_reason = "llm_output_unparseable"
            result.latency_ms = int((time.time() - started) * 1000)
            return result

        # 4. 按 LLM 给的顺序重排，保留原顺序作为 fallback
        result.llm_used = True
        items_by_id = {item.doc_id: item for item in original_order}
        new_items: list[RankedItem] = []
        for new_idx, doc_id in enumerate(new_order_ids[:top_k]):
            if doc_id in items_by_id:
                original_item = items_by_id[doc_id]
                # 更新 new_rank 和 rerank_score（1.0 是 top-1，0.0 是末尾）
                original_item.new_rank = new_idx
                original_item.rerank_score = max(0.0, 1.0 - new_idx * 0.1)
                new_items.append(original_item)

        # 5. 填充 LLM 没返回的剩余候选（按原顺序）
        used_ids = {i.doc_id for i in new_items}
        for item in original_order:
            if item.doc_id not in used_ids:
                item.new_rank = len(new_items)
                item.rerank_score = max(0.0, 1.0 - len(new_items) * 0.1)
                new_items.append(item)
            if len(new_items) >= top_k:
                break

        result.items = new_items[:top_k]
        result.rerank_changed = self._check_rerank_changed(original_order[:top_k], result.items)
        result.latency_ms = int((time.time() - started) * 1000)
        return result

    # ==================== 内部方法 ====================

    def _build_ranked_items_from_candidates(
        self, candidates: list[dict[str, Any]],
    ) -> list[RankedItem]:
        items = []
        for i, c in enumerate(candidates):
            content = c.get("content", "") or c.get("document", "")
            items.append(RankedItem(
                doc_id=c.get("id", c.get("doc_id", f"doc-{i}")),
                original_rank=i,
                new_rank=i,
                original_score=float(c.get("distance", c.get("score", 1.0))),
                rerank_score=0.0,  # LLM 还没打分
                content_snippet=content[:280] if content else "",
                metadata=c.get("metadata", {}),
            ))
        return items

    def _parse_llm_output(self, reasoning: str) -> list[str]:
        """
        解析 LLM 输出。约定：reasoning 字段是 JSON list of doc_id strings
        （MockLLMProvider 透传这个格式）。
        """
        if not reasoning:
            return []
        try:
            data = json.loads(reasoning)
        except (json.JSONDecodeError, TypeError):
            return []
        if isinstance(data, list):
            return [str(x) for x in data if x]
        return []

    def _check_rerank_changed(
        self,
        before: list[RankedItem],
        after: list[RankedItem],
    ) -> bool:
        if len(before) != len(after):
            return True
        return any(b.doc_id != a.doc_id for b, a in zip(before, after))
