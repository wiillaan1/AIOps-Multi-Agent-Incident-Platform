"""本地跑 CI eval 门禁（不开 GitHub Actions 也能验证 baseline）

用法：
    cd backend && python ../scripts/eval_gate.py [--split dev|test|holdout]

效果：
    1. 跑当前数据集评测
    2. 对比 baseline（首次跑创建 baseline）
    3. 退出码 0 = 通过，2 = 失败
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 把 backend/ 加进 path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

# 强制 fallback 避免 LLM 调用
import os
os.environ["AIOPS_FORCE_FALLBACK"] = "1"

from app.eval.dataset import get_default_dataset
from app.eval.evaluator import run_evaluation
from app.eval.history import (
    check_against_baseline,
    load_baseline,
    save_baseline,
    get_latest_run,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "holdout"])
    parser.add_argument("--baseline", default="rules_engine_baseline")
    parser.add_argument("--reset-baseline", action="store_true", help="忽略现有 baseline，重置")
    parser.add_argument("--create-only", action="store_true", help="只创建 baseline 不检查")
    args = parser.parse_args()

    print(f"=== Eval gate: split={args.split}, baseline={args.baseline} ===")

    # 检查 baseline
    existing = load_baseline(args.baseline)

    # 跑一次 eval
    ds = get_default_dataset(eval_split=args.split)
    results, summary = run_evaluation(
        ds,
        record_history=True,
        analyzer_name=f"eval_gate_{args.split}",
        history_note="local eval gate",
    )

    print(f"\n📊 Current run: pass_rate={summary.pass_rate:.1%} ({summary.passed}/{summary.total})")
    print(f"   by_category: {dict((k, v.get('rate', 0)) for k, v in summary.by_category.items())}")

    if existing is None or args.reset_baseline:
        # 首次跑：创建 baseline
        print(f"\n📌 Creating baseline '{args.baseline}'...")
        save_baseline(
            args.baseline,
            pass_rate=summary.pass_rate,
            total=summary.total,
            passed=summary.passed,
            description=f"Baseline for split={args.split}",
            split=args.split,
            source="eval_gate",
            notes=f"created by scripts/eval_gate.py (split={args.split})",
        )
        print(f"   pass_rate={summary.pass_rate:.1%}")
        return 0

    if args.create_only:
        return 0

    # 对比 baseline
    check = check_against_baseline(summary, args.baseline)
    print(f"\n🔍 Baseline check:")
    print(f"   current_pass_rate: {check['current_pass_rate']:.1%}")
    print(f"   baseline_pass_rate: {check['baseline_pass_rate']:.1%}")
    print(f"   delta: {check['delta']:.1%}")
    print(f"   verdict: {check['reason']}")

    if check["passed"]:
        print(f"\n✅ Eval gate PASSED")
        return 0
    else:
        print(f"\n❌ Eval gate FAILED")
        return 2


if __name__ == "__main__":
    sys.exit(main())