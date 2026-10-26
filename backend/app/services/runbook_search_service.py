"""
AIOps Agent Platform - Runbook Search Service

把已发布的 Runbook 版本索引到独立的 ChromaDB 集合（runbook_index），
提供基于语义的"找历史 Runbook"能力。

与 MemorySystem 的区别：
- MemorySystem 索引短期记忆条目（detection / RCA / heal），粒度细
- RunbookSearchService 索引完整 Runbook 快照，粒度粗但适合直接复用

索引字段：
- document: markdown 全文
- metadata: incident_id / version_number / status / service / root_cause /
            confidence / change_note / published_by / created_at
"""

from __future__ import annotations

import os
from typing import Any

from app.services.runbook_version_store import RunbookVersion
from app.utils.logging import get_logger

logger = get_logger(__name__)


COLLECTION_NAME = "runbook_index"


def _doc_id_for(version: RunbookVersion) -> str:
    return f"{version.incident_id}::v{version.version_number}"


def _build_search_document(version: RunbookVersion) -> str:
    """把 Runbook 摘要拼成一段可索引文本，避免整篇 markdown 过重。"""
    return "\n".join(
        filter(
            None,
            [
                version.title,
                f"服务: {version.service}",
                f"根因: {version.root_cause}",
                f"置信度: {version.confidence:.2%}",
                version.change_note,
                version.markdown[:2000],  # 截断长文档
            ],
        )
    )


class RunbookSearchService:
    """Runbook 版本检索服务，封装独立的 Chroma 集合。"""

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.persist_directory = (
            persist_directory
            or os.getenv("CHROMA_PERSIST_DIR", "./data/chromadb")
        )
        self.collection_name = collection_name
        self._client: Any = None
        self._collection: Any = None

    def _ensure_initialized(self) -> bool:
        """惰性初始化 Chroma client 与 collection。失败时返回 False 让上层降级。"""
        if self._collection is not None:
            return True
        try:
            import chromadb
            from chromadb.config import Settings

            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Indexed published Runbooks"},
            )
            return True
        except Exception as exc:
            logger.warning(
                "ChromaDB init failed, runbook search will degrade to empty",
                error=str(exc),
            )
            self._client = None
            self._collection = None
            return False

    # ==================== Index ====================

    def index_version(self, version: RunbookVersion) -> bool:
        """把单个版本索引到 Chroma。幂等：重复索引覆盖同一 doc_id。"""
        if not self._ensure_initialized():
            return False
        try:
            doc_id = _doc_id_for(version)
            self._collection.upsert(
                ids=[doc_id],
                documents=[_build_search_document(version)],
                metadatas=[{
                    "incident_id": version.incident_id,
                    "version_number": version.version_number,
                    "status": version.status,
                    "service": version.service,
                    "root_cause": version.root_cause,
                    "confidence": float(version.confidence),
                    "change_note": version.change_note,
                    "published_by": version.published_by or "",
                    "created_at": version.created_at,
                    "title": version.title,
                }],
            )
            logger.info(
                "Runbook indexed",
                doc_id=doc_id,
                service=version.service,
                root_cause=version.root_cause,
            )
            return True
        except Exception as exc:
            logger.error(
                "Failed to index runbook version",
                incident_id=version.incident_id,
                version=version.version_number,
                error=str(exc),
            )
            return False

    def remove_version(self, incident_id: str, version_number: int) -> bool:
        """从索引中删除一个版本（归档时可选调用）。"""
        if not self._ensure_initialized():
            return False
        try:
            self._collection.delete(ids=[f"{incident_id}::v{version_number}"])
            return True
        except Exception as exc:
            logger.warning(
                "Failed to remove runbook from index",
                incident_id=incident_id,
                version=version_number,
                error=str(exc),
            )
            return False

    # ==================== Search ====================

    def search(
        self,
        query: str,
        top_k: int = 5,
        service: str | None = None,
        status: str | None = "published",
        min_confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        语义检索 Runbook。

        Args:
            query: 自然语言查询（如 "order-service 高 CPU 怎么自愈"）
            top_k: 返回条数
            service: 可选按 service 过滤
            status: 可选按 status 过滤（默认仅搜 published）
            min_confidence: 最低置信度过滤

        Returns:
            list[dict]: 命中条目，每条含 distance + 元数据 + 摘要
        """
        if not query.strip():
            return []
        if not self._ensure_initialized():
            return []

        where: dict[str, Any] = {}
        if service:
            where["service"] = service
        if status:
            where["status"] = status
        if min_confidence is not None:
            where["confidence"] = {"$gte": float(min_confidence)}

        try:
            result = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where or None,
            )
        except Exception as exc:
            logger.warning("Runbook search failed", error=str(exc))
            return []

        hits: list[dict[str, Any]] = []
        ids_list = result.get("ids", [[]])[0] or []
        docs_list = result.get("documents", [[]])[0] or []
        metas_list = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        for i, doc_id in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) else {}
            doc = docs_list[i] if i < len(docs_list) else ""
            distance = distances[i] if i < len(distances) else None
            similarity = (1.0 - distance) if distance is not None else None
            hits.append({
                "doc_id": doc_id,
                "incident_id": meta.get("incident_id"),
                "version_number": meta.get("version_number"),
                "status": meta.get("status"),
                "service": meta.get("service"),
                "root_cause": meta.get("root_cause"),
                "confidence": meta.get("confidence"),
                "title": meta.get("title"),
                "change_note": meta.get("change_note"),
                "published_by": meta.get("published_by"),
                "created_at": meta.get("created_at"),
                "score": round(similarity, 4) if similarity is not None else None,
                "snippet": doc[:280] + ("..." if len(doc) > 280 else ""),
            })

        logger.info(
            "Runbook search executed",
            query=query[:80],
            hits=len(hits),
            service_filter=service,
        )
        return hits

    def stats(self) -> dict[str, Any]:
        """返回索引统计。"""
        if not self._ensure_initialized():
            return {"available": False, "count": 0}
        try:
            count = self._collection.count()
            return {"available": True, "count": count}
        except Exception:
            return {"available": False, "count": 0}


# 单例
runbook_search_service = RunbookSearchService()
