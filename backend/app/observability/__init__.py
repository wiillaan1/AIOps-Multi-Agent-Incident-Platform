"""
Observability 模块 - LLM 可观测性

当前实现：
- LangfuseTracer：封装 Langfuse SDK（trace + score）

设计原则（来自调研）：
1. trace 是"调用全链路记录"，不是 audit 替代 —— 两者互补
2. Langfuse 失败不能影响主链路 —— silent fallback
3. token 成本、延迟、错误率是关键指标
4. score() 给 trace 打分，用于 eval（badcase 评估）
"""
from app.observability.langfuse_tracer import LangfuseTracer, get_langfuse_tracer

__all__ = ["LangfuseTracer", "get_langfuse_tracer"]
