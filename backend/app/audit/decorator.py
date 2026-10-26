"""
@audit 装饰器 - 自动记录函数调用

用法：
    @audit(
        action=AuditAction.LLM_CALLED,
        target_type="llm",
        target_id_extractor=lambda args, kwargs: kwargs.get("model", "unknown"),
    )
    async def chat(messages, model="gpt-4o-mini"):
        ...

参数：
    action: 必填，AuditAction 值
    actor: 默认 "system:decorator"，可覆盖
    target_type: 默认 "function"
    target_id_extractor: 函数，从 args/kwargs 提取 target_id（默认用函数名）
    include_input: 是否记录入参（默认 True，可能含敏感信息时设为 False）
    include_output: 是否记录返回值（默认 False，输出可能很大）
    on_exception: 异常时是否记录（默认 True，action 用 <原 action>.failed）

注意：
- 装饰器本身不捕获异常，调用方仍需处理
- 异常时记录 action=<原 action> + status=failed（不修改 enum）
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable

from app.audit.logger import get_audit_logger
from app.audit.models import compute_hash


logger = logging.getLogger(__name__)


def audit(
    action: str,
    actor: str = "system:decorator",
    target_type: str = "function",
    target_id_extractor: Callable[..., str] | None = None,
    include_input: bool = True,
    include_output: bool = False,
) -> Callable:
    """
    装饰器：自动审计函数调用。

    target_id_extractor 签名: (args: tuple, kwargs: dict) -> str
    默认用 func.__name__。
    """
    def decorator(func: Callable) -> Callable:
        func_name = func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            target_id = (
                target_id_extractor(args, kwargs)
                if target_id_extractor
                else func_name
            )
            audit_logger = get_audit_logger()
            start = time.time()

            input_data = None
            if include_input:
                try:
                    input_data = {"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)}
                except Exception:
                    input_data = {"_repr_error": True}

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = int((time.time() - start) * 1000)

                output_data = None
                if include_output:
                    try:
                        output_data = {"result": _safe_repr(result), "elapsed_ms": elapsed_ms}
                    except Exception:
                        output_data = {"_repr_error": True}

                audit_logger.record(
                    actor=actor,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    input_data=input_data,
                    output_data=output_data,
                    details={"status": "success", "elapsed_ms": elapsed_ms},
                )
                return result
            except Exception as e:
                elapsed_ms = int((time.time() - start) * 1000)
                audit_logger.record(
                    actor=actor,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    input_data=input_data,
                    output_data=None,
                    details={
                        "status": "failed",
                        "elapsed_ms": elapsed_ms,
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e)[:500],  # 截断防止过大
                    },
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            target_id = (
                target_id_extractor(args, kwargs)
                if target_id_extractor
                else func_name
            )
            audit_logger = get_audit_logger()
            start = time.time()

            input_data = None
            if include_input:
                try:
                    input_data = {"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)}
                except Exception:
                    input_data = {"_repr_error": True}

            try:
                result = func(*args, **kwargs)
                elapsed_ms = int((time.time() - start) * 1000)

                output_data = None
                if include_output:
                    try:
                        output_data = {"result": _safe_repr(result), "elapsed_ms": elapsed_ms}
                    except Exception:
                        output_data = {"_repr_error": True}

                audit_logger.record(
                    actor=actor,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    input_data=input_data,
                    output_data=output_data,
                    details={"status": "success", "elapsed_ms": elapsed_ms},
                )
                return result
            except Exception as e:
                elapsed_ms = int((time.time() - start) * 1000)
                audit_logger.record(
                    actor=actor,
                    action=action,
                    target_type=target_type,
                    target_id=target_id,
                    input_data=input_data,
                    output_data=None,
                    details={
                        "status": "failed",
                        "elapsed_ms": elapsed_ms,
                        "exception_type": type(e).__name__,
                        "exception_msg": str(e)[:500],
                    },
                )
                raise

        # 根据原函数类型返回对应 wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _safe_repr(obj: Any, max_len: int = 2000) -> Any:
    """安全 repr（截断防止审计日志过大）"""
    try:
        s = repr(obj)
        if len(s) > max_len:
            return s[:max_len] + f"...<truncated {len(s) - max_len} chars>"
        return obj
    except Exception:
        return f"<unreprable: {type(obj).__name__}>"
