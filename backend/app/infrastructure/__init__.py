"""
AIOps Agent Platform - Infrastructure Registry

统一管理所有外部依赖（ChromaDB / Prometheus / MemorySystem）的单例。
提供 initialize() 用于启动初始化、close() 用于关闭清理。
"""
from __future__ import annotations

import os
from typing import Any

from app.config import AppConfig
from app.utils.logging import get_logger

logger = get_logger(__name__)


class InfrastructureRegistry:
    """基础设施注册中心 — 单例管理外部 client"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.chroma_client: Any = None
        self.memory_system: Any = None
        self.prometheus_client: Any = None
        self.loki_client: Any = None
        self._initialized = False

    async def initialize(self) -> dict[str, bool]:
        """
        初始化所有外部依赖

        Returns:
            dict[str, bool]: 每个依赖的就绪状态
        """
        status: dict[str, bool] = {}

        # 1. ChromaDB
        try:
            from app.memory.storage import ChromaDBStorage
            self.chroma_client = ChromaDBStorage(
                collection_name="aiops_memory",
                persist_directory="./data/chromadb",
            )
            await self.chroma_client._ensure_initialized()
            status["chromadb"] = self.chroma_client._initialized
            logger.info("ChromaDB connected", initialized=self.chroma_client._initialized)
        except Exception as e:
            logger.warning("ChromaDB unavailable", error=str(e))
            status["chromadb"] = False

        # 2. MemorySystem
        try:
            from app.memory.core import MemorySystem
            self.memory_system = await MemorySystem.get_instance()
            status["memory"] = True
            logger.info("MemorySystem ready")
        except Exception as e:
            logger.warning("MemorySystem unavailable", error=str(e))
            status["memory"] = False

        # 3. LokiClient（按需创建；由 registry 统一托管关闭）
        try:
            from app.infrastructure.log_client import LokiClient
            self.loki_client = LokiClient(
                base_url=os.getenv("LOKI_URL", "http://localhost:3100")
            )
            status["loki"] = True
        except Exception as e:
            logger.warning("Loki client init failed", error=str(e))
            status["loki"] = False

        # 4. PrometheusClient
        try:
            from app.infrastructure.prometheus_client import PrometheusClient
            self.prometheus_client = PrometheusClient(
                base_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090")
            )
            health = await self.prometheus_client.health_check()
            status["prometheus"] = health.get("reachable", False)
            logger.info("Prometheus health", **health)
        except Exception as e:
            logger.warning("Prometheus client init failed", error=str(e))
            status["prometheus"] = False

        self._initialized = True
        return status

    async def close(self) -> None:
        """关闭所有 client"""
        for name in ("prometheus_client", "loki_client"):
            client = getattr(self, name, None)
            if client:
                try:
                    await client.close()
                except Exception as e:
                    logger.debug("Infrastructure client close error", client=name, error=str(e))
        logger.info("Infrastructure closed")
