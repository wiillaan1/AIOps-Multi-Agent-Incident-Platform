"""
AIOps Agent Platform — Loki HTTP Client

通过 Loki HTTP API 查询日志数据。
为 RCA / Heal Agent 提供真实日志数据源接入能力。

API 文档: https://grafana.com/docs/loki/latest/reference/api/

Loki 查询使用 LogQL：标签选择器 `{service="order"} |= "ERROR"`，与
PromQL 的 label selector 风格保持一致。
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp

from app.utils.logging import get_logger

logger = get_logger(__name__)


class LokiClient:
    """
    Loki HTTP API 客户端。

    支持：
    - query_range（时间范围查询日志流）
    - query（即时查询当前日志）
    - labels / series（标签发现）
    - health_check（连通性检查）
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("LOKI_URL", "http://localhost:3100")
        ).rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ==================== Query Range ====================

    async def query_range(
        self,
        query: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
        direction: str = "backward",
    ) -> list[dict[str, Any]]:
        """
        查询时间范围内的日志流。

        Args:
            query: LogQL 表达式，例如 `{service="order-service"} |= "ERROR"`
            start: 开始时间（默认 1 小时前）
            end: 结束时间（默认现在）
            limit: 返回最大条数
            direction: forward / backward

        Returns:
            list[dict]: 标准化的日志条目列表，结构：
                [
                    {
                        "timestamp": "2026-07-17T10:00:00Z",
                        "labels": {"service": "order-service", ...},
                        "line": "ERROR 2026-07-17 ... connection timeout",
                    },
                    ...
                ]
        """
        if end is None:
            end = datetime.now(timezone.utc)
        if start is None:
            start = end - timedelta(hours=1)

        url = f"{self.base_url}/loki/api/v1/query_range"
        params: dict[str, Any] = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "limit": limit,
            "direction": direction,
        }

        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Loki query_range non-200",
                        query=query,
                        status=resp.status,
                    )
                    return []
                payload = await resp.json()

            return self._flatten_streams(payload)
        except Exception as exc:
            logger.error(
                "Loki query_range error",
                query=query,
                error=str(exc),
            )
            return []

    # ==================== Instant Query ====================

    async def query_instant(
        self,
        query: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        即时查询（最近一个时间窗的日志）。

        Args:
            query: LogQL 表达式
            limit: 最大返回条数
        """
        url = f"{self.base_url}/loki/api/v1/query"
        params: dict[str, Any] = {"query": query, "limit": limit}
        try:
            session = await self._get_session()
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                payload = await resp.json()
            return self._flatten_streams(payload)
        except Exception as exc:
            logger.error("Loki query_instant error", error=str(exc))
            return []

    # ==================== 标签发现 ====================

    async def list_label_values(self, label: str) -> list[str]:
        """列出指定 label 的所有取值（如 service / namespace）。"""
        url = f"{self.base_url}/loki/api/v1/label/{label}/values"
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                payload = await resp.json()
            return list(payload.get("data", []))
        except Exception as exc:
            logger.error(
                "Loki list_label_values error", label=label, error=str(exc)
            )
            return []

    # ==================== 预置便捷方法 ====================

    async def get_service_errors(
        self,
        service: str,
        lookback_minutes: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取指定服务 ERROR 级别日志。"""
        query = f'{{service="{service}"}} |= "ERROR"'
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        return await self.query_range(query, start=start, end=end, limit=limit)

    async def get_service_logs(
        self,
        service: str,
        keyword: str | None = None,
        lookback_minutes: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """获取指定服务的近期日志，可选关键词过滤。"""
        if keyword:
            query = f'{{service="{service}"}} |= "{keyword}"'
        else:
            query = f'{{service="{service}"}}'
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        return await self.query_range(query, start=start, end=end, limit=limit)

    # ==================== 健康检查 ====================

    async def health_check(self) -> dict[str, Any]:
        """检查 Loki 服务是否可达（使用 /ready 端点）。"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/ready") as resp:
                return {
                    "reachable": resp.status == 200,
                    "status_code": resp.status,
                    "url": self.base_url,
                }
        except Exception as exc:
            return {
                "reachable": False,
                "error": str(exc),
                "url": self.base_url,
            }

    # ==================== 内部辅助 ====================

    @staticmethod
    def _flatten_streams(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """
        把 Loki 的流式响应（按 stream 分组）扁平化为单条记录列表。

        Loki 原生结构：
            {
              "status": "success",
              "data": {
                "resultType": "streams",
                "result": [
                  {
                    "stream": {"service": "order-service"},
                    "values": [
                      ["<ts_ns>", "<log line>"],
                      ...
                    ]
                  }
                ]
              }
            }
        """
        entries: list[dict[str, Any]] = []
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        for stream in data.get("result", []):
            stream_labels = stream.get("stream", {}) or {}
            for ts_ns, line in stream.get("values", []) or []:
                try:
                    timestamp = (
                        datetime.fromtimestamp(
                            int(ts_ns) / 1e9, tz=timezone.utc
                        ).isoformat()
                    )
                except (TypeError, ValueError):
                    timestamp = str(ts_ns)
                entries.append(
                    {
                        "timestamp": timestamp,
                        "labels": dict(stream_labels),
                        "line": line,
                    }
                )
        return entries