"""
公共工具：适配器基类 + 数据集输出

[v3 E2 改造] 2026-09-01
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 项目路径：data/adapters/ → ../  → 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BADCASES_DIR = PROJECT_ROOT / "data" / "badcases"


def deterministic_split(
    items: list[Any],
    seed: int = 42,
    dev_ratio: float = 0.7,
    test_ratio: float = 0.2,
) -> tuple[list, list, list]:
    """确定性切分：dev / test / holdout。

    用 id 哈希取模保证同一份数据永远切分到同一 split。
    """
    rng = random.Random(seed)
    indexed = list(enumerate(items))
    rng.shuffle(indexed)

    n = len(items)
    dev_end = int(n * dev_ratio)
    test_end = dev_end + int(n * test_ratio)

    dev_items = [items[i] for i, _ in indexed[:dev_end]]
    test_items = [items[i] for i, _ in indexed[dev_end:test_end]]
    holdout_items = [items[i] for i, _ in indexed[test_end:]]
    return dev_items, test_items, holdout_items


def write_split(
    items: list[Any],
    eval_split: str,
    output_path: Path | None = None,
    meta: dict | None = None,
) -> Path:
    """写一个 split 的 JSON 文件。"""
    if output_path is None:
        output_path = BADCASES_DIR / f"v1_{eval_split}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": {
            "version": "v1",
            "eval_split": eval_split,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "count": len(items),
            **(meta or {}),
        },
        "badcases": [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in items],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def merge_or_replace_split(items: list[Any], eval_split: str, source_tag: str) -> Path:
    """追加一个 source 的 items 到 split 文件（不覆盖其他 source 的内容）。"""
    path = BADCASES_DIR / f"v1_{eval_split}.json"

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_badcases = existing.get("badcases", [])
    else:
        existing = {"_meta": {}, "badcases": []}
        existing_badcases = []

    new_dicts = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in items
    ]
    # 去重（同 id 不重复添加）
    existing_ids = {b.get("id") for b in existing_badcases}
    added = []
    for d in new_dicts:
        if d["id"] not in existing_ids:
            existing_badcases.append(d)
            existing_ids.add(d["id"])
            added.append(d["id"])

    existing["_meta"]["version"] = "v1"
    existing["_meta"]["eval_split"] = eval_split
    existing["_meta"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    existing["_meta"]["count"] = len(existing_badcases)
    existing["_meta"]["last_added"] = source_tag
    existing["badcases"] = existing_badcases

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path, added
