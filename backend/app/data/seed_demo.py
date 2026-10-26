"""
backend/app/data/seed_demo.py

生成/验证演示用的模拟告警数据。

复用 backend/app/api/routes.py 里已有的 _seed_incidents()，避免和
真实运行时用的种子数据脱节 —— 教学文档写"50 条"，但当前代码硬编码
的种子数据实际只有 7 条，这里如实复用现有实现，不臆造数字。

⚠️ 重要限制：IncidentService 目前是纯内存单例（见 incident_service.py
"# TODO: 使用真实数据库"），每个 Python 进程各自持有一份数据。本脚本
单独运行一次，插入的数据只存在于本脚本自己的进程里，进程退出后即
丢失 —— 不会被随后单独启动的 `python -m app.main` 后端进程看到。
如果你需要让"跑起来的后端"里真的带着这些种子数据，应改为在后端
已经启动之后，调用它自带的 dev-only 端点：
    POST /api/v1/incidents/seed-demo

运行方式（在 backend/ 目录下，仅用于验证种子数据本身是否正确）：
    python -m app.data.seed_demo
"""

from __future__ import annotations

from app.api.routes import _incident_service, _seed_incidents


def seed_demo() -> None:
    before = len(_incident_service._incidents)
    _seed_incidents()
    after = len(_incident_service._incidents)

    print(f"插入 {after - before} 条模拟告警（用于跑 demo）")
    print("注意：这些数据只存在于本次脚本进程的内存里，不会自动出现在随后启动的后端服务中。")


if __name__ == "__main__":
    seed_demo()