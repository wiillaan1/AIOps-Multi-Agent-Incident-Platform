"""
Audit 模块 - 持久化审计日志

设计原则：
1. append-only（不允许 UPDATE/DELETE）
2. 覆盖 6 类关键操作：LLM / RCA / 变更 / 自愈 / Badcase / 自进化
3. 每个 entry 含 input/output hash（用于事后验证）
4. 支持 CSV/JSON 导出（合规审计）
5. hash chain 是 SOX 升级路径，当前 TODO 注释

用法：
    from app.audit import AuditLogger, AuditAction, audit

    logger = AuditLogger()
    logger.record(
        actor="system:rca_agent",
        action=AuditAction.RCA_COMPLETED,
        target_type="incident",
        target_id="inc-123",
        input_data={"alert": {...}},
        output_data={"root_cause": "...", "confidence": 0.85},
    )

    # 或者装饰器
    @audit(action=AuditAction.LLM_CALLED, target_type="llm")
    async def my_llm_call(prompt):
        ...
"""
from app.audit.models import AuditAction, AuditEntry
from app.audit.logger import AuditLogger, get_audit_logger, reset_audit_logger
from app.audit.decorator import audit

__all__ = [
    "AuditAction",
    "AuditEntry",
    "AuditLogger",
    "get_audit_logger",
    "reset_audit_logger",
    "audit",
]
