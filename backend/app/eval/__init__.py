"""Eval 模块 - Badcase 评测

[v3 统一改造 E1+E3+E5] 2026-09-01
- 单一 schema: BadCaseEntry（与 badcase_capture / harness_engine 共用）
- 三分离: v1_dev.json / v1_test.json / v1_holdout.json
- 历史 + baseline: SQLite + JSON 文件

设计目标：
- 跑 badcase 测试集，量化 RCA Agent 效果
- 4 个核心指标：pass_rate / confidence / citation / hallucination
- Phase C 自进化用 A/B 测试对比 prompt 改动效果
- 当前实现：BadcaseDataset / RCAEvaluator / run_evaluation / history / baselines
"""
from app.eval.dataset import (
    BadcaseDataset,
    EvalSplit,
    get_default_dataset,
    get_split_path,
)
from app.eval.evaluator import (
    EvalResult,
    EvalSummary,
    RCAEvaluator,
    RCAOutput,
    run_evaluation,
)
from app.eval.history import (
    check_against_baseline,
    get_baselines_dir,
    get_history_db_path,
    get_latest_run,
    get_trend,
    list_baselines,
    list_runs,
    load_baseline,
    record_eval_run,
    save_baseline,
)
from app.models.badcase import BadCaseEntry

# 向后兼容：Badcase 别名
Badcase = BadCaseEntry

__all__ = [
    "Badcase",
    "BadCaseEntry",
    "BadcaseDataset",
    "EvalSplit",
    "RCAEvaluator",
    "RCAOutput",
    "EvalResult",
    "EvalSummary",
    "run_evaluation",
    "get_default_dataset",
    "get_split_path",
    # v3 E5
    "record_eval_run",
    "list_runs",
    "get_trend",
    "get_latest_run",
    "save_baseline",
    "load_baseline",
    "list_baselines",
    "check_against_baseline",
    "get_history_db_path",
    "get_baselines_dir",
]