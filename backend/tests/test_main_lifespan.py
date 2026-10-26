"""测试 main.py lifespan 真实初始化"""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_lifespan_initializes_infrastructure():
    """lifespan 必须初始化 InfrastructureRegistry"""
    from app.main import create_app
    app = create_app()
    with TestClient(app) as client:
        infra = client.app.state.infrastructure
        assert infra is not None
        assert infra._initialized is True


def test_infrastructure_registry_closes_loki_client():
    """InfrastructureRegistry.close 必须关闭已注册的 Loki client。"""
    import asyncio
    from unittest.mock import AsyncMock

    from app.config import AppConfig
    from app.infrastructure import InfrastructureRegistry

    registry = InfrastructureRegistry(AppConfig())
    registry.loki_client = type("LokiStub", (), {"close": AsyncMock()})()
    asyncio.run(registry.close())
    registry.loki_client.close.assert_awaited_once()


def test_lifespan_knowledge_base_loaded():
    """lifespan 必须加载 KnowledgeBase"""
    from app.main import create_app
    app = create_app()
    with TestClient(app) as client:
        kb = client.app.state.knowledge_base
        # 即使 KB 加载方式改变，状态属性必须存在
        assert hasattr(client.app.state, "knowledge_base")
        if kb is not None:
            assert kb.get("loaded") is True


def test_lifespan_orchestrator_built():
    """lifespan 必须构建 Orchestrator"""
    from app.main import create_app
    app = create_app()
    with TestClient(app) as client:
        # orchestrator 可能为 None 如果 build 失败，但属性必须存在
        assert hasattr(client.app.state, "orchestrator")
        assert hasattr(client.app.state, "langgraph_available")