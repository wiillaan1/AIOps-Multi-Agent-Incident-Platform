"""
审计日志数据模型

涵盖 6 类关键操作：
1. LLM 调用（核心可观测）
2. RCA 决策
3. 变更决策 + 执行
4. 自愈执行
5. Badcase 捕获
6. 自进化 prompt 改动
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class AuditAction(str, Enum):
    """审计动作枚举（白名单）"""

    # --- LLM 类 ---
    LLM_CALLED = "llm_called"
    LLM_FALLBACK = "llm_fallback"
    LLM_CIRCUIT_OPEN = "llm_circuit_open"
    LLM_CIRCUIT_HALF_OPEN = "llm_circuit_half_open"

    # --- RCA 类 ---
    RCA_COMPLETED = "rca_completed"
    RCA_FALLBACK = "rca_fallback"

    # --- 变更类 ---
    CHANGE_DECIDED = "change_decided"
    CHANGE_APPROVED = "change_approved"
    CHANGE_REJECTED = "change_rejected"
    CHANGE_EXECUTED = "change_executed"
    CHANGE_FAILED = "change_failed"

    # --- 自愈类 ---
    HEAL_EXECUTED = "heal_executed"
    HEAL_FAILED = "heal_failed"
    HEAL_DRY_RUN = "heal_dry_run"

    # --- Badcase & 评估 ---
    BADCASE_CAPTURED = "badcase_captured"
    REVIEW_REQUESTED = "review_requested"
    REVIEW_SUBMITTED = "review_submitted"

    # --- 自进化 ---
    PATTERN_ANALYZED = "pattern_analyzed"
    PROMPT_FIX_GENERATED = "prompt_fix_generated"
    PROMPT_AB_TESTED = "prompt_ab_tested"
    PROMPT_PROMOTED = "prompt_promoted"
    PROMPT_REJECTED = "prompt_rejected"


@dataclass
class AuditEntry:
    """审计日志条目（不可变）"""

    id: int | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    actor: str = ""           # "system:rca_agent" / "user:zhangsan" / "tool:knowledge_search"
    action: str = ""          # AuditAction.value
    target_type: str = ""     # "incident" / "change" / "prompt" / "agent" / "llm"
    target_id: str = ""       # 具体 ID
    input_hash: str = ""      # sha256 of input_data
    output_hash: str = ""     # sha256 of output_data
    details: dict[str, Any] = field(default_factory=dict)
    # TODO: prev_hash for SOX hash chain compliance
    prev_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compute_hash(data: Any) -> str:
    """计算任意对象的 sha256（用于 input/output 验证）"""
    import hashlib
    if data is None:
        payload = b""
    elif isinstance(data, (str, bytes)):
        payload = data.encode("utf-8") if isinstance(data, str) else data
    else:
        payload = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
