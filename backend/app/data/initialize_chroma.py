"""
backend/app/data/initialize_chroma.py

初始化 ChromaDB 知识库索引：把 app.data.knowledge_base.KNOWLEDGE_BASE
中的运维故障知识条目写入一个独立的 Chroma collection（用于后续 RAG 检索/评测）。

运行方式（在 backend/ 目录下）：
    python -m app.data.initialize_chroma
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from app.data.knowledge_base import KNOWLEDGE_BASE

PERSIST_DIRECTORY = "./data/chromadb"
COLLECTION_NAME = "knowledge_base"

_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _entry_to_document(entry: dict) -> str:
    """把结构化知识条目拼成一段可供 embedding 的自然语言文本"""
    symptoms = "、".join(entry.get("symptoms", []))
    root_causes = "、".join(entry.get("root_causes", []))
    solutions = "、".join(entry.get("solutions", []))
    return (
        f"类别: {entry.get('category', '')}. "
        f"症状: {symptoms}. "
        f"根因: {root_causes}. "
        f"解决方案: {solutions}."
    )


def _entry_to_metadata(entry: dict) -> dict:
    """Chroma metadata 只接受 str/int/float/bool，列表字段需要拍平成字符串"""
    return {
        "category": entry.get("category", ""),
        "symptoms": ",".join(entry.get("symptoms", [])),
        "root_causes": ",".join(entry.get("root_causes", [])),
        "solutions": ",".join(entry.get("solutions", [])),
        "confidence_boost": float(entry.get("confidence_boost", 0.0)),
    }


def initialize_chroma() -> None:
    client = chromadb.PersistentClient(
        path=PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_embedding_function,
        metadata={"description": "AIOps knowledge base for RAG"},
    )

    # 幂等：已经写过就不重复插入
    if collection.count() >= len(KNOWLEDGE_BASE):
        print(f"ChromaDB initialized with {collection.count()} knowledge base entries")
        return

    ids = [entry["id"] for entry in KNOWLEDGE_BASE]
    documents = [_entry_to_document(entry) for entry in KNOWLEDGE_BASE]
    metadatas = [_entry_to_metadata(entry) for entry in KNOWLEDGE_BASE]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"ChromaDB initialized with {collection.count()} knowledge base entries")


if __name__ == "__main__":
    initialize_chroma()