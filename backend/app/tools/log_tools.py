"""
AIOps Agent Platform - Log Tools

日志查询工具，基于 Loki HTTP API 提供真实日志检索能力。
提供基础查询工具（LogQL）+ 高级便捷方法（按服务 / 错误级别）。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.infrastructure.log_client import LokiClient
from app.tools.base import BaseTool, ToolParameter, ToolResult
from app.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# 基础日志查询工具（保留用于 LLM function calling）
# ============================================================


class QueryLogsTool(BaseTool):
    """查询日志工具（LogQL）"""

    @property
    def name(self) -> str:
        return "query_logs"

    @property
    def description(self) -> str:
        return "从日志系统（Loki）查询日志，支持 LogQL 表达式"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                description='LogQL 查询(如 {service="order"} |= "ERROR")',
                type="string",
                required=True,
            ),
            ToolParameter(
                name="start_time",
                description="查询起始时间(ISO 8601 格式)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="end_time",
                description="查询结束时间(ISO 8601 格式)",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="limit",
                description="返回最大条数",
                type="integer",
                required=False,
                default=100,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        limit = int(kwargs.get("limit", 100))

        logger.info("Querying logs", query=query, limit=limit)

        client = LokiClient()
        try:
            end = (
                _parse_iso(end_time)
                if end_time
                else datetime.now(timezone.utc)
            )
            start = (
                _parse_iso(start_time)
                if start_time
                else end - timedelta(hours=1)
            )
            entries = await client.query_range(
                query=query, start=start, end=end, limit=limit
            )
            return ToolResult.ok(
                tool_name=self.name,
                data={
                    "query": query,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": limit,
                    "entries": entries,
                    "count": len(entries),
                    "source": "loki",
                },
            )
        except Exception as exc:
            logger.error("Loki query failed", query=query, error=str(exc))
            return ToolResult.error(
                tool_name=self.name,
                error_message=f"Loki query failed: {exc}",
            )
        finally:
            await client.close()


class GetServiceLogsTool(BaseTool):
    """按服务名查询日志的工具（便捷封装）"""

    @property
    def name(self) -> str:
        return "get_service_logs"

    @property
    def description(self) -> str:
        return "按服务名查询近期日志，可选关键词过滤"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="service",
                description="服务名，如 order-service",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="keyword",
                description="可选关键词（如 ERROR / timeout）",
                type="string",
                required=False,
            ),
            ToolParameter(
                name="lookback_minutes",
                description="回溯分钟数",
                type="integer",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="limit",
                description="返回最大条数",
                type="integer",
                required=False,
                default=100,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = kwargs.get("service", "")
        keyword = kwargs.get("keyword")
        lookback = int(kwargs.get("lookback_minutes", 30))
        limit = int(kwargs.get("limit", 100))

        logger.info("Get service logs", service=service, keyword=keyword)

        if not service:
            return ToolResult.error(
                tool_name=self.name,
                error_message="service is required",
            )

        client = LokiClient()
        try:
            entries = await client.get_service_logs(
                service=service,
                keyword=keyword,
                lookback_minutes=lookback,
                limit=limit,
            )
            return ToolResult.ok(
                tool_name=self.name,
                data={
                    "service": service,
                    "keyword": keyword,
                    "lookback_minutes": lookback,
                    "entries": entries,
                    "count": len(entries),
                    "source": "loki",
                },
            )
        except Exception as exc:
            logger.error(
                "Loki get_service_logs failed",
                service=service,
                error=str(exc),
            )
            return ToolResult.error(
                tool_name=self.name,
                error_message=f"Loki get_service_logs failed: {exc}",
            )
        finally:
            await client.close()


class GetServiceErrorsTool(BaseTool):
    """按服务名查询 ERROR 级别日志的工具"""

    @property
    def name(self) -> str:
        return "get_service_errors"

    @property
    def description(self) -> str:
        return "按服务名查询 ERROR 级别日志"

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="service",
                description="服务名，如 order-service",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="lookback_minutes",
                description="回溯分钟数",
                type="integer",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="limit",
                description="返回最大条数",
                type="integer",
                required=False,
                default=100,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        service = kwargs.get("service", "")
        lookback = int(kwargs.get("lookback_minutes", 30))
        limit = int(kwargs.get("limit", 100))

        logger.info("Get service errors", service=service)

        if not service:
            return ToolResult.error(
                tool_name=self.name,
                error_message="service is required",
            )

        client = LokiClient()
        try:
            entries = await client.get_service_errors(
                service=service,
                lookback_minutes=lookback,
                limit=limit,
            )
            return ToolResult.ok(
                tool_name=self.name,
                data={
                    "service": service,
                    "lookback_minutes": lookback,
                    "entries": entries,
                    "count": len(entries),
                    "source": "loki",
                },
            )
        except Exception as exc:
            logger.error(
                "Loki get_service_errors failed",
                service=service,
                error=str(exc),
            )
            return ToolResult.error(
                tool_name=self.name,
                error_message=f"Loki get_service_errors failed: {exc}",
            )
        finally:
            await client.close()


# ============================================================
# 注册函数
# ============================================================


def register_log_tools() -> list[BaseTool]:
    """
    注册所有日志工具

    Returns:
        list[BaseTool]: 日志工具列表
    """
    return [
        QueryLogsTool(),
        GetServiceLogsTool(),
        GetServiceErrorsTool(),
    ]


# ============================================================
# 内部辅助
# ============================================================


def _parse_iso(value: str) -> datetime:
    """解析 ISO 8601 时间字符串为 timezone-aware datetime。"""
    # Python 3.10 fromisoformat 不支持 'Z' 后缀，3.11+ 才支持
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt