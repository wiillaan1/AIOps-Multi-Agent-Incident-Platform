"""
Review CLI - oncall 评分命令行

用法：
    python -m app.review list                # 列最近评分
    python -m app.review list-pending ID1 ID2 ...   # 查待评分 incident
    python -m app.review show <incident_id>  # 显示 incident + 上次评分
    python -m app.review submit \\
        --incident inc-001 \\
        --badcase bc-001 \\
        --reviewer zhangsan \\
        --decision accept \\
        --accuracy 4 --explainability 4 --actionability 5 --overall 4 \\
        --comments "合理"
    python -m app.review stats               # 评分统计
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from app.review.models import (
    DEFAULT_DIMENSIONS,
    Review,
    ReviewDecision,
    ReviewDimension,
)
from app.review.store import ReviewStore, get_review_store


logger = logging.getLogger(__name__)


def cmd_list(args: argparse.Namespace, store: ReviewStore) -> int:
    """列出最近评分"""
    reviews = store.list_recent(limit=args.limit)
    if not reviews:
        print("📭 暂无评分记录")
        return 0

    print(f"=== 最近 {len(reviews)} 条评分 ===\n")
    for r in reviews:
        avg = r.avg_score()
        decision_icon = {
            "accept": "✅",
            "reject": "❌",
            "needs_improvement": "⚠️ ",
        }.get(r.decision, "?")

        print(f"{decision_icon} #{r.id}  {r.incident_id}  "
              f"reviewer={r.reviewer}  avg={avg:.1f}/5")
        print(f"    decision={r.decision}  scores={r.scores}")
        if r.comments:
            print(f"    💬 {r.comments}")
        print(f"    🕐 {r.timestamp}")
        print()
    return 0


def cmd_list_pending(args: argparse.Namespace, store: ReviewStore) -> int:
    incident_ids = args.incidents
    if not incident_ids:
        print("⚠️ 请提供 incident_id（空格分隔）")
        return 1
    pending = store.list_pending_incidents(incident_ids)
    if not pending:
        print(f"✅ 所有 {len(incident_ids)} 个 incident 都已评分")
        return 0
    print(f"=== 待评分 {len(pending)}/{len(incident_ids)} ===")
    for iid in pending:
        print(f"  ⏳ {iid}")
    return 0


def cmd_show(args: argparse.Namespace, store: ReviewStore) -> int:
    incident_id = args.incident_id
    reviews = store.list_recent(limit=1000)
    incident_reviews = [r for r in reviews if r.incident_id == incident_id]
    if not incident_reviews:
        print(f"📭 incident '{incident_id}' 暂无评分")
        return 0
    print(f"=== incident {incident_id} 的评分历史（{len(incident_reviews)} 条）===\n")
    for r in incident_reviews:
        print(f"#{r.id}  reviewer={r.reviewer}  decision={r.decision}  avg={r.avg_score():.1f}")
        print(f"  scores: {r.scores}")
        if r.comments:
            print(f"  💬 {r.comments}")
        print()
    return 0


def cmd_submit(args: argparse.Namespace, store: ReviewStore) -> int:
    """提交一条评分"""
    if not args.incident:
        print("❌ --incident 必填")
        return 1
    if not args.reviewer:
        print("❌ --reviewer 必填")
        return 1

    try:
        decision = ReviewDecision(args.decision)
    except ValueError:
        print(f"❌ --decision 必须是 {', '.join(d.value for d in ReviewDecision)}")
        return 1

    # 收集 4 个维度的分数
    scores: dict[str, int] = {}
    for dim in DEFAULT_DIMENSIONS:
        val = getattr(args, dim.value, None)
        if val is None:
            print(f"❌ --{dim.value} 必填（1-5）")
            return 1
        if not (1 <= val <= 5):
            print(f"❌ --{dim.value} 必须是 1-5，当前 {val}")
            return 1
        scores[dim.value] = val

    review = Review(
        incident_id=args.incident,
        badcase_id=args.badcase or "",
        reviewer=args.reviewer,
        decision=decision.value,
        scores=scores,
        comments=args.comments or "",
    )
    saved = store.submit(review)

    # 写审计（可选，避免循环依赖）
    try:
        from app.audit import AuditAction, get_audit_logger
        get_audit_logger().record(
            actor=f"user:{args.reviewer}",
            action=AuditAction.REVIEW_SUBMITTED,
            target_type="review",
            target_id=str(saved.id),
            input_data={"incident_id": args.incident, "scores": scores},
            output_data={"decision": decision.value, "comments": args.comments},
        )
    except Exception as e:
        logger.warning(f"审计写入失败（不影响评分）: {e}")

    print(f"✅ 评分已提交：id={saved.id}  decision={decision.value}  avg={saved.avg_score():.1f}/5")
    return 0


def cmd_stats(args: argparse.Namespace, store: ReviewStore) -> int:
    """评分统计"""
    stats = store.stats(since=args.since)
    print("=== 人评统计 ===\n")
    print(f"总评分数: {stats['total']}")
    print(f"按决策: {stats['by_decision']}")
    print(f"按 reviewr: {stats['by_reviewer']}")
    print(f"平均 overall: {stats['avg_overall']}/5")
    print(f"通过率（accept / (accept+reject)）: {stats['pass_rate']:.1%}")
    if args.since:
        print(f"时间范围: since {args.since}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.review",
        description="Oncall 评分 CLI（人评机制）",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = sub.add_parser("list", help="列出最近评分")
    p_list.add_argument("--limit", type=int, default=20)

    # list-pending
    p_lp = sub.add_parser("list-pending", help="查待评分 incident")
    p_lp.add_argument("incidents", nargs="+", help="incident_id 列表")

    # show
    p_show = sub.add_parser("show", help="显示 incident 评分历史")
    p_show.add_argument("incident_id")

    # submit
    p_submit = sub.add_parser("submit", help="提交一条评分")
    p_submit.add_argument("--incident", required=True)
    p_submit.add_argument("--badcase", default="")
    p_submit.add_argument("--reviewer", required=True)
    p_submit.add_argument("--decision", required=True,
                          help=f"accept | reject | needs_improvement")
    p_submit.add_argument("--accuracy", type=int)
    p_submit.add_argument("--explainability", type=int)
    p_submit.add_argument("--actionability", type=int)
    p_submit.add_argument("--overall", type=int)
    p_submit.add_argument("--comments", default="")

    # stats
    p_stats = sub.add_parser("stats", help="评分统计")
    p_stats.add_argument("--since", help="ISO 时间起点")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    store = get_review_store()
    handlers = {
        "list": cmd_list,
        "list-pending": cmd_list_pending,
        "show": cmd_show,
        "submit": cmd_submit,
        "stats": cmd_stats,
    }
    handler = handlers[args.command]
    return handler(args, store)


if __name__ == "__main__":
    sys.exit(main())
