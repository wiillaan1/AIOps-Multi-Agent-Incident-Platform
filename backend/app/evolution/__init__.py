"""
Evolution 模块 - 自进化 5 阶段闭环

设计原则（Meta/Microsoft/Zalando 共识 + 去粗取精）：
1. **不自动热加载**——必须人工 review（避免 LLM 改坏）
2. **A/B 测试量化**——pass_rate 提升 ≥5% 才进入 promote 候选
3. **写审计**——每次 prompt 改动写 AuditAction.PROMPT_*
4. **不依赖真实 LLM**——用 mock 跑通流程（生产可换真 LLM）

5 阶段：
1. Badcase 收集（Phase B2 数据集）
2. LLM 分析模式（Phase A2 LLMClient + Phase B1 Langfuse）
3. LLM 生成 prompt 修复（生成新 prompt 段）
4. A/B 测试（Phase B2 RCAEvaluator）
5. 人工 review（Phase B3 CLI）

入口：
- 编程式：from app.evolution import SelfEvolutionOrchestrator, run_cycle
- CLI: python -m app.evolution {run|status|promote|history}
"""
from app.evolution.models import (
    PromptVersion,
    EvolutionCycle,
    EvolutionStage,
    PromotionDecision,
)
from app.evolution.store import PromptStore, get_prompt_store, reset_prompt_store
from app.evolution.self_evolution import SelfEvolutionOrchestrator, run_cycle

__all__ = [
    "PromptVersion",
    "EvolutionCycle",
    "EvolutionStage",
    "PromotionDecision",
    "PromptStore",
    "get_prompt_store",
    "reset_prompt_store",
    "SelfEvolutionOrchestrator",
    "run_cycle",
]
