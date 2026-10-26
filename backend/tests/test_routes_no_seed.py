"""测试 routes.py 不再在模块加载时 seed incidents"""
from __future__ import annotations


def test_import_routes_no_seed():
    """导入 routes 后 _incident_service 不应有 seed 数据（模块顶层无调用）"""
    # 通过行为验证：导入 routes 后 incidents 字典应为空
    import importlib
    import sys
    from app.services.incident_service import IncidentService

    baseline = len(IncidentService()._incidents)
    if "app.api.routes" in sys.modules:
        del sys.modules["app.api.routes"]
    import app.api.routes as routes_module
    # 行为验证：重新导入不应新增 seed 数据；共享服务可能已有其他测试数据。
    assert len(routes_module._incident_service._incidents) == baseline, (
        "Importing routes must not add seed incidents"
    )


def test_seed_demo_endpoint_exists():
    """/incidents/seed-demo 端点必须存在"""
    from app.main import create_app
    app = create_app()
    routes = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/api/v1/incidents/seed-demo" in routes, (
        "seed-demo endpoint should be registered"
    )