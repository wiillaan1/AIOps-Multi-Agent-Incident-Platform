"""Eval CLI - 评测历史 + baseline + diff 命令。

子命令：
- run: 跑一次评测（自动记录历史）
- history: 列最近评测
- trend: 显示某 split 的 pass_rate 趋势
- baseline set/list/show: baseline 管理
- diff: 两次评测对比
- promote-badcase: 把 candidates.jsonl 的草稿提升到 v1_dev.json（v3 E10）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.eval.dataset import get_default_dataset
from app.eval.evaluator import run_evaluation
from app.eval.history import (
    check_against_baseline,
    get_latest_run,
    get_trend,
    list_baselines,
    list_runs,
    load_baseline,
    save_baseline,
)


def cmd_run(args) -> int:
    """跑一次评测，自动记录历史"""
    ds = get_default_dataset(eval_split=args.split) if args.split != "all" else get_default_dataset()
    if args.split == "all" and ds is None:
        print("Error: no data 集 found")
        return 1

    results, summary = run_evaluation(
        ds,
        verbose=args.verbose,
        record_history=True,
        history_note=args.note,
        analyzer_name=args.analyzer,
    )

    print(f"✅ Eval done: pass_rate={summary.pass_rate:.1%} ({summary.passed}/{summary.total})")

    if args.baseline:
        check = check_against_baseline(summary, args.baseline, min_pass_rate=args.min_pass_rate)
        if check["passed"]:
            print(f"✅ Baseline check passed: {check['reason']}")
            return 0
        else:
            print(f"❌ Baseline check failed: {check['reason']}")
            return 2

    return 0


def cmd_history(args) -> int:
    """列最近评测"""
    rows = list_runs(split=args.split, limit=args.limit)
    if not rows:
        print("(no eval runs yet)")
        return 0

    print(f"{'eval_id':<22} {'split':<8} {'model':<18} {'pass_rate':<10} {'total':<6} {'run_at'}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['eval_id'][:20]:<22} "
            f"{r['split']:<8} "
            f"{(r['model'] or '-'):<18} "
            f"{r['pass_rate']:.1%}        "
            f"{r['total']:<6} "
            f"{r['run_at'][:19]}"
        )
    return 0


def cmd_trend(args) -> int:
    """显示某 split 的 pass_rate 趋势"""
    rows = get_trend(split=args.split, limit=args.limit)
    if not rows:
        print(f"(no eval runs for split={args.split})")
        return 0

    print(f"=== pass_rate trend (split={args.split}) ===")
    print(f"{'run_at':<22} {'pass_rate':<12} {'total':<6}")
    for r in rows:
        print(f"{r['run_at'][:19]:<22} {r['pass_rate']:.1%}        {r['total']:<6}")
    return 0


def cmd_baseline_set(args) -> int:
    """保存 baseline"""
    summary = get_latest_run(split=args.split)
    if summary is None:
        print(f"❌ No recent eval run for split={args.split}, run one first")
        return 1

    path = save_baseline(
        args.name,
        pass_rate=summary["pass_rate"],
        total=summary["total"],
        passed=summary["passed"],
        description=args.description or f"saved from eval {summary['eval_id']}",
        split=args.split,
        source="latest_run",
        notes=args.notes or "",
    )
    print(f"✅ Saved baseline '{args.name}' to {path}")
    print(f"   pass_rate: {summary['pass_rate']:.1%} ({summary['passed']}/{summary['total']})")
    return 0


def cmd_baseline_list(args) -> int:
    """列出所有 baseline"""
    baselines = list_baselines()
    if not baselines:
        print("(no baselines saved)")
        return 0

    print(f"{'name':<28} {'pass_rate':<12} {'split':<10} {'source':<14} {'created_at'}")
    print("-" * 90)
    for b in baselines:
        print(
            f"{b['name']:<28} "
            f"{b['pass_rate']:.1%}        "
            f"{b['split']:<10} "
            f"{b['source']:<14} "
            f"{b['created_at'][:19]}"
        )
    return 0


def cmd_baseline_show(args) -> int:
    """显示某个 baseline 详情"""
    b = load_baseline(args.name)
    if b is None:
        print(f"❌ Baseline '{args.name}' 不存在")
        return 1
    print(json.dumps(b, indent=2, ensure_ascii=False))
    return 0


def cmd_diff(args) -> int:
    """对比两次评测"""
    runs = list_runs(split=args.split, limit=args.limit * 10)
    if len(runs) < 2:
        print(f"❌ 需要至少 2 次评测才能对比，当前 {len(runs)} 次")
        return 1

    # 取最近两次
    a, b = runs[1], runs[0]
    print(f"=== Eval Diff ===")
    print(f"Old: {a['eval_id'][:20]} @ {a['run_at'][:19]} pass_rate={a['pass_rate']:.1%}")
    print(f"New: {b['eval_id'][:20]} @ {b['run_at'][:19]} pass_rate={b['pass_rate']:.1%}")
    print(f"Delta: {(b['pass_rate'] - a['pass_rate']):.1%}")

    # diff failing_ids
    old_fail = set(json.loads(a['failing_ids_json'] or '[]'))
    new_fail = set(json.loads(b['failing_ids_json'] or '[]'))
    fixed = old_fail - new_fail
    broken = new_fail - old_fail
    print(f"\nFixed: {len(fixed)} badcase(s): {sorted(fixed)[:10]}")
    print(f"Broken: {len(broken)} badcase(s): {sorted(broken)[:10]}")
    return 0


def cmd_promote_badcase(args) -> int:
    """从 candidates.jsonl 把 badcase 草稿提升到 v1_dev.json（v3 E10）"""
    from app.eval.promote import promote_candidate
    try:
        result = promote_candidate(args.entry_id, dest_split=args.dest_split)
    except Exception as e:
        print(f"❌ Failed: {e}")
        return 1
    print(f"✅ Promoted {args.entry_id} → {result['dest']}")
    print(f"   expected_root_cause: {result['expected_root_cause']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.eval", description="评测 CLI")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # run
    p_run = sub.add_parser("run", help="跑一次评测")
    p_run.add_argument("--split", default="dev", choices=["dev", "test", "holdout", "all"])
    p_run.add_argument("--analyzer", default="default")
    p_run.add_argument("--note", default=None)
    p_run.add_argument("--baseline", default=None, help="门禁对比 baseline 名")
    p_run.add_argument("--min-pass-rate", type=float, default=None)
    p_run.add_argument("--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # history
    p_hist = sub.add_parser("history", help="列最近评测")
    p_hist.add_argument("--split", default=None)
    p_hist.add_argument("--limit", type=int, default=30)
    p_hist.set_defaults(func=cmd_history)

    # trend
    p_trend = sub.add_parser("trend", help="显示 pass_rate 趋势")
    p_trend.add_argument("--split", required=True)
    p_trend.add_argument("--limit", type=int, default=30)
    p_trend.set_defaults(func=cmd_trend)

    # baseline
    p_bset = sub.add_parser("baseline", help="baseline 管理")
    bsub = p_bset.add_subparsers(dest="baseline_action", required=True)
    p_bset_do = bsub.add_parser("set", help="保存 baseline（从最近一次 eval）")
    p_bset_do.add_argument("name")
    p_bset_do.add_argument("--split", default="test")
    p_bset_do.add_argument("--description", default="")
    p_bset_do.add_argument("--notes", default="")
    p_bset_do.set_defaults(func=cmd_baseline_set)
    p_blist = bsub.add_parser("list", help="列 baseline")
    p_blist.set_defaults(func=cmd_baseline_list)
    p_bshow = bsub.add_parser("show", help="显示 baseline 详情")
    p_bshow.add_argument("name")
    p_bshow.set_defaults(func=cmd_baseline_show)

    # diff
    p_diff = sub.add_parser("diff", help="对比两次评测")
    p_diff.add_argument("--split", default=None)
    p_diff.add_argument("--limit", type=int, default=2)
    p_diff.set_defaults(func=cmd_diff)

    # promote-badcase
    p_promote = sub.add_parser("promote-badcase", help="把 candidates.jsonl 草稿提升到 v1_dev.json")
    p_promote.add_argument("entry_id")
    p_promote.add_argument("--dest-split", default="dev", choices=["dev", "test", "holdout"])
    p_promote.set_defaults(func=cmd_promote_badcase)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())