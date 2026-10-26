"""
LangfuseTracer - Langfuse trace 封装

提供：
- 每次 LLM chat 自动 trace（含 input/output/model/latency/token）
- score() 给 trace 打分（eval 用）
- silent fallback —— Langfuse 不可用时不阻断主链路

依赖：
- langfuse>=4.0
- LangfuseConfig（来自 app.config）

设计决策：
- 不强制 Langfuse —— enabled=false 或配置缺失时降级到 no-op
- 失败全部 logger.warning —— 不抛异常
- trace_id 返回 None 表示未 trace
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from app.config import LangfuseConfig


logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Langfuse trace 封装（线程安全）"""

    def __init__(self, config: LangfuseConfig):
        self.config = config
        self._client: Any | None = None
        self._init_lock = threading.Lock()
        self._init_client()

    def _init_client(self) -> None:
        """延迟初始化：只在 enabled + 有 key 时连"""
        if not self.config.enabled:
            logger.info("[Langfuse] disabled（config.langfuse.enabled=false）")
            return
        if not self.config.public_key or not self.config.secret_key:
            logger.info("[Langfuse] 配置缺失 public_key/secret_key，禁用")
            return
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=self.config.public_key,
                secret_key=self.config.secret_key,
                host=self.config.host,
                release=self.config.release,
                environment=self.config.environment,
            )
            logger.info(f"[Langfuse] initialized host={self.config.host} env={self.config.environment}")
        except Exception as e:
            logger.warning(f"[Langfuse] 初始化失败，降级到 no-op: {e}")
            self._client = None

    # ---------- 状态 ----------
    @property
    def is_enabled(self) -> bool:
        return self._client is not None

    def status(self) -> dict:
        """返回可观测性状态（用于健康检查/dashboard）"""
        return {
            "enabled": self.config.enabled,
            "host": self.config.host if self.config.enabled else None,
            "environment": self.config.environment if self.config.enabled else None,
            "client_active": self.is_enabled,
        }

    # ---------- trace ----------
    def trace_chat(
        self,
        messages: list[Any],
        response: str | None = None,
        model: str = "",
        provider: str = "",
        latency_ms: int = 0,
        token_in: int | None = None,
        token_out: int | None = None,
        error: str | None = None,
        incident_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        记录一次 LLM chat 到 Langfuse。

        返回 trace_id（用于后续 score）；失败/未启用返回 None。
        """
        if not self.is_enabled:
            return None

        try:
            md = {
                "incident_id": incident_id,
                "provider": provider,
                **(metadata or {}),
            }
            usage: dict[str, Any] = {}
            if token_in is not None:
                usage["input"] = token_in
            if token_out is not None:
                usage["output"] = token_out

            trace = self._client.trace(
                name="llm_chat",
                metadata=md,
            )
            gen_kwargs: dict[str, Any] = {
                "name": "chat_completion",
                "model": model,
                "input": messages,
            }
            if response is not None:
                gen_kwargs["output"] = response
            if usage:
                gen_kwargs["usage"] = usage
            if metadata:
                gen_kwargs["metadata"] = metadata

            generation = trace.generation(**gen_kwargs)
            if error:
                generation.update(level="ERROR", status_message=error)

            # 显式 flush（Langfuse 4.x 是异步批量上报）
            try:
                self._client.flush()
            except Exception:
                pass

            return trace.id
        except Exception as e:
            logger.warning(f"[Langfuse] trace 失败: {e}")
            return None

    # ---------- score（eval 用）----------
    def score(
        self,
        trace_id: str | None,
        name: str,
        value: float | int | str,
        comment: str = "",
    ) -> bool:
        """
        给指定 trace 打分（用于 badcase 评估、人评）。

        返回是否成功。
        """
        if not self.is_enabled or not trace_id:
            return False
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
            return True
        except Exception as e:
            logger.warning(f"[Langfuse] score 失败: {e}")
            return False

    # ---------- flush ----------
    def flush(self) -> None:
        """强制 flush（用于测试/优雅关闭）"""
        if self.is_enabled:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"[Langfuse] flush 失败: {e}")

    def shutdown(self) -> None:
        """关闭客户端（用于测试清理）"""
        if self.is_enabled:
            try:
                self._client.shutdown()
            except Exception:
                pass


# ---------- 全局单例 ----------
_tracer: LangfuseTracer | None = None
_tracer_lock = threading.Lock()


def get_langfuse_tracer(config: LangfuseConfig | None = None) -> LangfuseTracer:
    """获取全局 Langfuse tracer"""
    global _tracer
    if _tracer is None:
        with _tracer_lock:
            if _tracer is None:
                from app.config import get_config
                cfg = config or get_config().langfuse
                _tracer = LangfuseTracer(cfg)
    return _tracer


def reset_langfuse_tracer() -> None:
    """重置（用于测试）"""
    global _tracer
    if _tracer is not None:
        _tracer.shutdown()
    _tracer = None
