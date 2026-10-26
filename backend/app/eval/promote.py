"""v3 E10: badcase 草稿 → 评估集自动 promote

工作流：
1. maybe_capture_badcase() 把 incident 中的失败信号写到 candidates.jsonl
2. 人工 review candidates.jsonl
3. 人工调用 promote_candidate(entry_id, expected_root_cause, dest_split='dev')
4. 草稿被转成完整 BadCaseEntry，写到 v1_{dest_split}.json
5. 下次 run_evaluation 就能看到这个新 badcase

安全：promote 必须显式给 expected_root_cause（不能从草稿照搬）
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.badcase import BadCaseEntry

# 复用 data/adapters/common.py 的路径解析（避免重复定义）
def _get_badcases_dir() -> Path:
    """解析 badcases 目录路径（兼容多种 cwd）"""
    # 优先从环境变量读（测试覆盖用）
    env = __import__("os").environ.get("BADCASES_DIR_OVERRIDE")
    if env:
        return Path(env)
    # 默认：项目根/data/badcases
    candidates = [
        Path(__file__).resolve().parents[3] / "data" / "badcases",  # backend/app/eval/promote.py → ../../../
        Path.cwd() / "data" / "badcases",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # 默认返回第一个

# 默认 candidates 路径（与 badcase_capture.py 保持一致）
def _candidates_path() -> Path:
    return _get_badcases_dir() / "candidates.jsonl"


def _reject_log_path() -> Path:
    return _get_badcases_dir() / "rejected.jsonl"


def list_candidates(candidates_path: Path | None = None) -> list[dict]:
    """列出 candidates.jsonl 里所有候选 badcase"""
    if candidates_path is None:
        candidates_path = _candidates_path()
    if not candidates_path.exists():
        return []
    out = []
    for line in candidates_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def get_candidate(entry_id: str, candidates_path: Path | None = None) -> dict | None:
    """根据 entry_id 拿候选 badcase"""
    for c in list_candidates(candidates_path):
        if c.get("id") == entry_id:
            return c
    return None


def promote_candidate(
    entry_id: str,
    *,
    expected_root_cause: str,
    dest_split: str = "dev",
    expected_confidence_min: float = 0.8,
    severity: str = "medium",
    candidates_path: Path | None = None,
) -> dict[str, Any]:
    """把候选 badcase 提升到评估集 v1_{dest_split}.json

    Args:
        entry_id: candidates.jsonl 里的 badcase id
        expected_root_cause: 必须人工标注（不能照搬草稿）
        dest_split: 'dev' | 'test' | 'holdout'
        expected_confidence_min: 期望置信度阈值

    Returns:
        {"dest": "...", "expected_root_cause": "...", "id": "..."}

    Raises:
        FileNotFoundError: 候选不存在
        ValueError: 评估集已有同 id
    """
    if dest_split not in ("dev", "test", "holdout"):
        raise ValueError(f"dest_split must be dev/test/holdout, got {dest_split}")

    candidate = get_candidate(entry_id, candidates_path)
    if candidate is None:
        raise FileNotFoundError(f"candidate {entry_id} not found in candidates.jsonl")

    # 把草稿升级为完整 BadCaseEntry
    new_badcase = BadCaseEntry(
        id=candidate["id"],  # 沿用原 id
        category=candidate.get("category") or candidate.get("badcase_class") or "single_metric",
        type=candidate.get("type", "unknown"),
        alert=candidate.get("alert", {}),
        evidence=candidate.get("evidence", {}),
        expected_root_cause=expected_root_cause,
        expected_confidence_min=expected_confidence_min,
        tags=candidate.get("tags", []),
        badcase_class=candidate.get("badcase_class", "auto_capture"),
        identified_flaw=candidate.get("identified_flaw", ""),
        keywords_for_retrieval=candidate.get("keywords_for_retrieval", []),
        severity=severity,
        eval_split=dest_split,
        source="auto_capture",
        description=candidate.get("description", f"promoted from candidate {entry_id}"),
    )

    # 加载 v1_{dest_split}.json
    target_path = _get_badcases_dir() / f"v1_{dest_split}.json"
    if target_path.exists():
        payload = json.loads(target_path.read_text(encoding="utf-8"))
    else:
        payload = {"_meta": {}, "badcases": []}

    # 检查重复
    existing_ids = {b.get("id") for b in payload.get("badcases", [])}
    if new_badcase.id in existing_ids:
        raise ValueError(f"badcase {new_badcase.id} already exists in {dest_split}")

    # 追加
    payload.setdefault("badcases", []).append(new_badcase.model_dump(mode="json"))
    payload["_meta"]["version"] = "v1"
    payload["_meta"]["eval_split"] = dest_split
    payload["_meta"]["last_added"] = f"promoted from candidate {entry_id}"
    payload["_meta"]["count"] = len(payload["badcases"])

    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "dest": str(target_path),
        "expected_root_cause": expected_root_cause,
        "id": new_badcase.id,
        "dest_split": dest_split,
    }


def reject_candidate(
    entry_id: str,
    *,
    candidates_path: Path | None = None,
    reject_log_path: Path | None = None,
) -> bool:
    """标记候选为 rejected（移动到 rejected.jsonl）"""
    if reject_log_path is None:
        reject_log_path = _reject_log_path()

    candidate = get_candidate(entry_id, candidates_path)
    if candidate is None:
        return False

    reject_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reject_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({**candidate, "_rejected_at": _now_iso()}, ensure_ascii=False) + "\n")
    return True


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()