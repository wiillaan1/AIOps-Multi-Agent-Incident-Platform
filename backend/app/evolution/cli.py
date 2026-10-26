"""
Evolution CLI - 自进化循环管理

用法：
    python -m app.evolution run \\
        --badcases bc-001,bc-002,bc-003 \\
        --prompt "现有 RCA prompt..."

    python -m app.evolution status              # 列出最近循环
    python -m app.evolution show <cycle_id>     # 查看详情
    python -m app.evolution promote <cycle_id> --reviewer zhangsan --notes "验证通过"
    python -m app.evolution reject <cycle_id> --reviewer zhangsan --notes "改动太大"
    python -m app.evolution prompt-list          # 列出所有 prompt 版本
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from app.evolution.models import PromotionDecision
from app.evolution.self_evolution import SelfEvolutionOrchestrator
from app.evolution.store import get_prompt_store, reset_prompt_store


logger = logging.getLogger(__name__)


def cmd_run(args: argparse.Namespace) -> int:
    """跑一次自进化循环"""
    if not args.badcases:
        print("❌ --badcases 必填（逗号分隔）")
        return 1
    if not args.prompt:
        print("❌ --prompt 必填（当前 RCA prompt）")
        return 1

    badcase_ids = [bid.strip() for bid in args.badcases.split(",") if bid.strip()]

    orch = SelfEvolutionOrchestrator()
    cycle = orch.run_cycle(
        badcase_ids=badcase_ids,
        current_prompt=args.prompt,
    )
    print(f"✅ cycle#{cycle.id} 已完成")
    print(f"   当前 stage: {cycle.current_stage}")
    print(f"   pattern: {cycle.pattern_analysis.get('pattern_type', '?')}")
    print(f"   AB test: old={cycle.ab_test_summary.get('old_pass_rate', 0):.2f}, "
          f"new={cycle.ab_test_summary.get('new_pass_rate', 0):.2f}, "
          f"delta={cycle.ab_test_summary.get('delta', 0):.3f}")
    print(f"   决策: {cycle.promotion_decision}")
    if cycle.promotion_decision == PromotionDecision.PENDING.value:
        print(f"   ⚠️ 待人工 review：python -m app.evolution promote {cycle.id} 或 reject {cycle.id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """列出最近循环"""
    store = get_prompt_store()
    cycles = store.list_cycles(limit=args.limit, decision=args.decision)
    if not cycles:
        print("📭 暂无循环记录")
        return 0

    print(f"=== 最近 {len(cycles)} 次循环 ===\n")
    for c in cycles:
        decision_icon = {
            "pending": "⏳",
            "promoted": "✅",
            "auto_accepted": "🟢",
            "rejected": "❌",
        }.get(c.promotion_decision, "?")

        ab = c.ab_test_summary
        delta = ab.get("delta", 0)
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"

        print(f"{decision_icon} cycle#{c.id}  badcases={len(c.badcase_ids)}  "
              f"decision={c.promotion_decision}")
        print(f"    stage={c.current_stage}  ab_test_delta={delta_str}")
        print(f"    started={c.started_at}")
        if c.reviewer:
            print(f"    reviewer={c.reviewer}  notes={c.review_notes[:60]}")
        print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    store = get_prompt_store()
    cycle = store.get_cycle(args.cycle_id)
    if not cycle:
        print(f"📭 cycle#{args.cycle_id} 不存在")
        return 1

    print(f"=== cycle#{cycle.id} 详情 ===\n")
    print(f"状态: {cycle.current_stage}")
    print(f"决策: {cycle.promotion_decision}")
    print(f"开始: {cycle.started_at}")
    if cycle.completed_at:
        print(f"完成: {cycle.completed_at}")
    print(f"\n--- Badcase IDs ---")
    for bid in cycle.badcase_ids:
        print(f"  {bid}")
    print(f"\n--- 模式分析 ---")
    print(json.dumps(cycle.pattern_analysis, ensure_ascii=False, indent=2))
    print(f"\n--- 生成 prompt（前 500 字） ---")
    print(cycle.generated_prompt[:500])
    if len(cycle.generated_prompt) > 500:
        print(f"...（共 {len(cycle.generated_prompt)} 字）")
    print(f"\n--- A/B 测试 ---")
    print(json.dumps(cycle.ab_test_summary, ensure_ascii=False, indent=2))
    if cycle.reviewer:
        print(f"\n--- 审核 ---")
        print(f"reviewer: {cycle.reviewer}")
        print(f"notes: {cycle.review_notes}")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    orch = SelfEvolutionOrchestrator()
    cycle = orch.promote(args.cycle_id, reviewer=args.reviewer, notes=args.notes or "")
    if not cycle:
        print(f"❌ cycle#{args.cycle_id} 不存在")
        return 1
    print(f"✅ cycle#{args.cycle_id} 已 PROMOTE")
    if args.reviewer:
        print(f"   reviewer={args.reviewer}")
    return 0


def cmd_reject(args: argparse.Namespace) -> int:
    orch = SelfEvolutionOrchestrator()
    cycle = orch.reject(args.cycle_id, reviewer=args.reviewer, notes=args.notes or "")
    if not cycle:
        print(f"❌ cycle#{args.cycle_id} 不存在")
        return 1
    print(f"❌ cycle#{args.cycle_id} 已 REJECT")
    return 0


def cmd_prompt_list(args: argparse.Namespace) -> int:
    """列出所有 prompt 版本"""
    store = get_prompt_store()
    # 简单做法：循环读取最近的
    # 实际应该加 list_prompts 方法到 store，这里简化
    print("=== Prompt 版本历史 ===")
    # 拿最近 cycle 的 prompt
    cycles = store.list_cycles(limit=100)
    seen_versions = set()
    for c in cycles:
        # 每个 cycle 关联的 prompt version（如果有）
        # 简化：通过 ab_test_results 反查
        pass

    # 用 store.get_latest_prompt() 然后找 chain
    latest = store.get_latest_prompt()
    if not latest:
        print("📭 暂无 prompt 版本")
        return 0

    print(f"\n✅ 最新版本: {latest.version}")
    print(f"   prompt 长度: {len(latest.prompt_segment)}")
    print(f"   target: {latest.target_dimension}")
    print(f"   cycle: #{latest.cycle_id}")
    print(f"   ab_test: {latest.ab_test_results}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.evolution",
        description="自进化循环 CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="跑一次自进化循环")
    p_run.add_argument("--badcases", required=True, help="逗号分隔的 badcase ID")
    p_run.add_argument("--prompt", required=True, help="当前 RCA prompt")

    # status
    p_status = sub.add_parser("status", help="列出最近循环")
    p_status.add_argument("--limit", type=int, default=10)
    p_status.add_argument("--decision", help="按决策过滤")

    # show
    p_show = sub.add_parser("show", help="查看循环详情")
    p_show.add_argument("cycle_id", type=int)

    # promote
    p_promote = sub.add_parser("promote", help="人工通过")
    p_promote.add_argument("cycle_id", type=int)
    p_promote.add_argument("--reviewer", required=True)
    p_promote.add_argument("--notes", default="")

    # reject
    p_reject = sub.add_parser("reject", help="人工拒绝")
    p_reject.add_argument("cycle_id", type=int)
    p_reject.add_argument("--reviewer", required=True)
    p_reject.add_argument("--notes", default="")

    # prompt-list
    sub.add_parser("prompt-list", help="列出 prompt 版本")

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {
        "run": cmd_run,
        "status": cmd_status,
        "show": cmd_show,
        "promote": cmd_promote,
        "reject": cmd_reject,
        "prompt-list": cmd_prompt_list,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
