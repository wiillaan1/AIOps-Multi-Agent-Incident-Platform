"""
Tests for the Loki integration.

We don't require a running Loki — these tests use a fake aiohttp session
so they stay hermetic and fast.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.log_client import LokiClient
from app.tools.log_tools import (
    GetServiceErrorsTool,
    GetServiceLogsTool,
    QueryLogsTool,
    _parse_iso,
    register_log_tools,
)


# ============================ LokiClient ============================


def _make_aiohttp_response(payload: dict[str, Any], status: int = 200):
    """Build a minimal aiohttp response context manager."""

    class _Resp:
        def __init__(self) -> None:
            self.status = status

        async def __aenter__(self) -> "_Resp":
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def json(self) -> dict[str, Any]:
            return payload

    return _Resp()


def _patched_session(client: LokiClient, response_payload: dict[str, Any], status: int = 200) -> AsyncMock:
    session = MagicMock()
    session.closed = False
    session.close = AsyncMock()

    def _get(url: str, params: dict[str, Any] | None = None) -> Any:
        return _make_aiohttp_response(response_payload, status)

    session.get = _get
    return session


def test_flatten_streams_basic():
    """_flatten_streams 把 Loki 流式响应正确扁平化。"""
    payload = {
        "status": "success",
        "data": {
            "resultType": "streams",
            "result": [
                {
                    "stream": {"service": "order-service"},
                    "values": [
                        ["1700000000000000000", "ERROR connection timeout"],
                        ["1700000060000000000", "ERROR retry exhausted"],
                    ],
                }
            ],
        },
    }
    entries = LokiClient._flatten_streams(payload)
    assert len(entries) == 2
    assert entries[0]["line"] == "ERROR connection timeout"
    assert entries[0]["labels"] == {"service": "order-service"}
    assert entries[0]["timestamp"].startswith("2023-")


def test_flatten_streams_empty():
    assert LokiClient._flatten_streams({"data": {"result": []}}) == []
    assert LokiClient._flatten_streams({}) == []


def test_flatten_streams_multiple_streams():
    payload = {
        "data": {
            "result": [
                {"stream": {"service": "a"}, "values": [["1", "line-a1"]]},
                {"stream": {"service": "b"}, "values": [["2", "line-b1"], ["3", "line-b2"]]},
            ]
        }
    }
    entries = LokiClient._flatten_streams(payload)
    assert len(entries) == 3
    services = {e["labels"]["service"] for e in entries}
    assert services == {"a", "b"}


def test_parse_iso_accepts_z_suffix():
    dt = _parse_iso("2026-07-17T10:00:00Z")
    assert dt.tzinfo is not None
    assert dt.year == 2026 and dt.hour == 10


def test_parse_iso_accepts_offset():
    dt = _parse_iso("2026-07-17T10:00:00+08:00")
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 8 * 3600


def test_parse_iso_naive_assumes_utc():
    dt = _parse_iso("2026-07-17T10:00:00")
    assert dt.tzinfo == timezone.utc


def test_parse_iso_invalid_falls_back_to_now():
    dt = _parse_iso("not-a-date")
    assert dt.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - dt).total_seconds()) < 5


@pytest.mark.asyncio
async def test_query_range_returns_entries():
    payload = {
        "data": {
            "result": [
                {
                    "stream": {"service": "order-service"},
                    "values": [["1700000000000000000", "ERROR oops"]],
                }
            ]
        }
    }
    client = LokiClient(base_url="http://test-loki:3100")
    try:
        session = _patched_session(client, payload)
        client._session = session

        entries = await client.query_range(
            query='{service="order-service"} |= "ERROR"',
            start=datetime(2026, 7, 17, tzinfo=timezone.utc),
            end=datetime(2026, 7, 17, 0, 30, tzinfo=timezone.utc),
            limit=50,
        )
        assert len(entries) == 1
        assert entries[0]["line"] == "ERROR oops"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_query_range_handles_non_200():
    client = LokiClient(base_url="http://test-loki:3100")
    try:
        client._session = _patched_session(client, {}, status=500)
        entries = await client.query_range(query='{service="x"}')
        assert entries == []
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_check_reachable():
    client = LokiClient(base_url="http://test-loki:3100")
    try:
        client._session = _patched_session(client, {}, status=200)
        result = await client.health_check()
        assert result["reachable"] is True
        assert result["url"] == "http://test-loki:3100"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_health_check_unreachable():
    client = LokiClient(base_url="http://test-loki:3100")
    try:
        session = MagicMock()
        session.closed = False
        session.close = AsyncMock()

        def _get(url: str, params: dict[str, Any] | None = None):
            raise ConnectionError("nope")

        session.get = _get
        client._session = session

        result = await client.health_check()
        assert result["reachable"] is False
        assert "nope" in result["error"]
    finally:
        await client.close()


# ============================ Tools ============================


@pytest.mark.asyncio
async def test_query_logs_tool_closes_loki_client():
    with patch("app.tools.log_tools.LokiClient") as client_cls:
        client = client_cls.return_value
        client.query_range = AsyncMock(return_value=[])
        client.close = AsyncMock()
        result = await QueryLogsTool().execute(query='{service="x"}', limit=10)
        assert result.success is True
        client.close.assert_awaited_once()


async def test_query_logs_tool_returns_loki_entries():
    payload = {
        "data": {
            "result": [
                {"stream": {"service": "x"}, "values": [["1", "hello"]]}
            ]
        }
    }
    with patch(
        "app.infrastructure.log_client.LokiClient._get_session",
        AsyncMock(
            return_value=_patched_session(SimpleNamespace(), payload)
        ),
    ):
        tool = QueryLogsTool()
        result = await tool.execute(query='{service="x"}', limit=10)
        assert result.success is True
        assert result.data["source"] == "loki"
        assert result.data["count"] == 1


@pytest.mark.asyncio
async def test_query_logs_tool_handles_loki_failure():
    with patch(
        "app.infrastructure.log_client.LokiClient.query_range",
        AsyncMock(side_effect=RuntimeError("connection refused")),
    ):
        tool = QueryLogsTool()
        result = await tool.execute(query='{service="x"}')
        assert result.success is False
        assert "connection refused" in result.error_message


@pytest.mark.asyncio
async def test_get_service_logs_tool_requires_service():
    tool = GetServiceLogsTool()
    result = await tool.execute()
    assert result.success is False
    assert "service" in result.error_message


@pytest.mark.asyncio
async def test_get_service_errors_tool_requires_service():
    tool = GetServiceErrorsTool()
    result = await tool.execute()
    assert result.success is False


def test_register_log_tools_returns_three_tools():
    tools = register_log_tools()
    names = {t.name for t in tools}
    assert names == {"query_logs", "get_service_logs", "get_service_errors"}
    # 所有工具都应能在主注册表中查到
    from app.tools.base import tool_registry
    for name in names:
        assert tool_registry.get(name) is not None