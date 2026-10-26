"""v3 E8: AIOpsLab leaderboard 对照 baseline

不真接入 AIOpsLab 框架（沙箱跑不动 docker/k8s），但提供：
1. 静态 leaderboard 参考分数
2. 按 task_type 对比的 utility 函数
3. 把 Badcase 中的 AIOpsLab problem 按 task_type 分组对比

学生本地有 docker 时可：
```bash
git clone --recurse-submodules https://github.com/microsoft/AIOpsLab
cd AIOpsLab
poetry install
# 按 README 启动 kind 集群
python -m app.eval.aiopslab_runner --real-agent my_agent.py
```
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

# 从 adapter 导入 AIOpsLab 数据
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "data"))
from adapters.aiopslab_to_badcase import AIOPSLAB_LEADERBOARD, AIOPSLAB_PROBLEMS


def get_leaderboard_score(agent_name: str, task_type: str) -> float:
    """获取某 agent 在某 task_type 的分数"""
    if agent_name not in AIOPSLAB_LEADERBOARD:
        raise KeyError(f"Unknown agent: {agent_name}. Available: {list(AIOPSLAB_LEADERBOARD.keys())}")
    scores = AIOPSLAB_LEADERBOARD[agent_name]
    if task_type not in scores:
        # "average" 不在四类里
        raise KeyError(f"Unknown task_type: {task_type}. Available: detection/localization/diagnosis/mitigation/average")
    return scores[task_type]


def get_all_leaderboard() -> dict[str, dict[str, float]]:
    """返回完整 leaderboard 字典"""
    return AIOPSLAB_LEADERBOARD


def count_aiopslab_problems_by_task() -> dict[str, int]:
    """统计 AIOpsLab problem 按 task_type 分布"""
    counts: dict[str, int] = defaultdict(int)
    for p in AIOPSLAB_PROBLEMS:
        # p = (id, application, fault_type, task_type, difficulty)
        counts[p[3]] += 1
    return dict(counts)


def compare_pass_rate_with_leaderboard(
    our_pass_rate: float,
    task_type: str,
) -> dict[str, Any]:
    """把我们的 pass_rate 跟 leaderboard 多个 agent 对比

    Args:
        our_pass_rate: 0.0-1.0
        task_type: 'detection' / 'localization' / 'diagnosis' / 'mitigation'

    Returns:
        dict 包含我们的分数、leaderboard 排名、verdict
    """
    # 收集所有 agent 在该 task 的分数
    scores = [
        (name, s[task_type])
        for name, s in AIOPSLAB_LEADERBOARD.items()
        if task_type in s
    ]
    scores.sort(key=lambda x: -x[1])

    # 找我们的排名
    rank = sum(1 for _, s in scores if s > our_pass_rate * 100) + 1

    # 找最近的参考 agent
    nearest = min(scores, key=lambda x: abs(x[1] - our_pass_rate * 100))

    return {
        "our_pass_rate": round(our_pass_rate, 4),
        "task_type": task_type,
        "rank": rank,
        "total_agents": len(scores),
        "leaderboard_top3": scores[:3],
        "leaderboard_bottom3": scores[-3:],
        "nearest_agent": {"name": nearest[0], "score": nearest[1]},
        "verdict": (
            f"我们的 {task_type} 准确率 {our_pass_rate:.1%} "
            f"接近 {nearest[0]}（{nearest[1]:.2f}%）"
        ),
    }


def evaluate_aiopslab_static(
    pass_rates_by_task: dict[str, float],
) -> dict[str, Any]:
    """对一组 task_type 的 pass_rate 跟 leaderboard 对比

    Args:
        pass_rates_by_task: {"detection": 0.95, "localization": 0.5, ...}

    Returns:
        完整对照报告
    """
    if "diagnosis" not in pass_rates_by_task and "analysis" in pass_rates_by_task:
        # 兼容 mapping：analysis → diagnosis（跟 AIOpsLab leaderboard 字段对齐）
        pass_rates_by_task = dict(pass_rates_by_task)
        pass_rates_by_task["diagnosis"] = pass_rates_by_task.pop("analysis")

    per_task = {}
    for tt, rate in pass_rates_by_task.items():
        if tt == "average":
            continue
        per_task[tt] = compare_pass_rate_with_leaderboard(rate, tt)

    return {
        "overall_average": sum(pass_rates_by_task.values()) / len([v for v in pass_rates_by_task.values() if v is not None]),
        "per_task": per_task,
        "summary": (
            f"我们在 {len(per_task)} 个 task 上跟 leaderboard 对比，"
            f"平均 {sum(pass_rates_by_task.values()) / len(pass_rates_by_task):.1%}"
        ),
    }


# ============ CLI ============

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="AIOpsLab leaderboard 对照")
    sub = parser.add_subparsers(dest="action")

    # show leaderboard
    p_show = sub.add_parser("show", help="显示完整 leaderboard")
    p_show.add_argument("--task", default=None, help="只看某个 task_type（detection/localization/diagnosis/mitigation）")

    # compare
    p_compare = sub.add_parser("compare", help="对比我们的分数跟 leaderboard")
    p_compare.add_argument("--detection", type=float, default=None)
    p_compare.add_argument("--localization", type=float, default=None)
    p_compare.add_argument("--diagnosis", type=float, default=None)
    p_compare.add_argument("--mitigation", type=float, default=None)

    args = parser.parse_args()

    if args.action == "show" or args.action is None:
        print("=== AIOpsLab Leaderboard ===")
        print(f"{'Agent':<30} {'avg':<8} {'det':<8} {'loc':<8} {'dia':<8} {'mit':<8} {'org'}")
        print("-" * 100)
        for name, scores in sorted(AIOPSLAB_LEADERBOARD.items(), key=lambda x: -x[1]["average"]):
            row = f"{name:<30} "
            if args.task:
                row += f"{scores.get(args.task, 0):<8.2f}"
            else:
                row += (
                    f"{scores['average']:<8.2f} "
                    f"{scores['detection']:<8.2f} "
                    f"{scores['localization']:<8.2f} "
                    f"{scores['diagnosis']:<8.2f} "
                    f"{scores['mitigation']:<8.2f} "
                )
            row += f"{scores['org']}"
            print(row)
        return 0

    if args.action == "compare":
        rates = {}
        if args.detection is not None:
            rates["detection"] = args.detection
        if args.localization is not None:
            rates["localization"] = args.localization
        if args.diagnosis is not None:
            rates["diagnosis"] = args.diagnosis
        if args.mitigation is not None:
            rates["mitigation"] = args.mitigation
        result = evaluate_aiopslab_static(rates)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())