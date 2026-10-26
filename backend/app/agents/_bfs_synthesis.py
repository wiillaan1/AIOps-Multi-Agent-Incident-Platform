"""
M9 - BFS 进合成 Implementation (GREEN 阶段)

借鉴 Devix §4.1 - 用 BFS 并行探索多条假设路径，合成最终结论。

设计：
- HypothesisNode: 单个假设节点（id/description/score/evidence）
- BFSScheduler: BFS 调度器（add_hypothesis → explore → synthesize）
- SynthesisResult: 合成结果（best_hypothesis_id / combined_evidence / confidence）

只做测试要求的功能：
- 并行探索（asyncio.gather）
- best_of / top_k / synthesize
- max_depth 配置
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable


@dataclass
class HypothesisNode:
    """单个假设节点"""
    id: str
    description: str
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    # ----- W7.5 候选归一化三元键（审查报告附录 B #5）-----
    # entity_id: 关联实体（service/instance/db 名）
    # cause_type: 根因类型（cpu/memory/network/disk/db/cache/deployment/config/...）
    # scope: 影响范围（service/cluster/global/region）
    # 三个字段都为可选；缺字段的候选按 (id, "unknown", "unknown") 归一，
    # 避免破坏现有调用方（RCA V2 候选入库前会自动 normalize）。
    entity_id: str | None = None
    cause_type: str | None = None
    scope: str | None = None


@dataclass
class SynthesisResult:
    """BFS 合成结果"""
    best_hypothesis_id: str
    best_hypothesis_description: str
    combined_evidence: list[str]
    confidence: float
    top_hypotheses: list[HypothesisNode]


class BFSScheduler:
    """BFS 调度器

    用法：
        scheduler = BFSScheduler(max_depth=3)
        scheduler.add_hypothesis(HypothesisNode(...))
        results = await scheduler.explore(check_fn)
        final = scheduler.synthesize(results)
    """

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self._hypotheses: list[HypothesisNode] = []

    def add_hypothesis(self, hyp: HypothesisNode) -> None:
        self._hypotheses.append(hyp)

    async def explore(
        self,
        check_fn: Callable[[HypothesisNode], Awaitable[HypothesisNode]],
    ) -> list[HypothesisNode]:
        """并行探索所有假设

        check_fn: 用户传入的检查函数（接收假设，返回更新后的假设）
        """
        if not self._hypotheses:
            return []
        # asyncio.gather 实现真正并行
        results = await asyncio.gather(
            *[check_fn(hyp) for hyp in self._hypotheses]
        )
        return list(results)

    def best_of(self, results: list[HypothesisNode]) -> HypothesisNode:
        """返回评分最高的假设"""
        if not results:
            raise ValueError("No hypotheses to evaluate")
        return max(results, key=lambda h: h.score)

    def top_k(
        self,
        results: list[HypothesisNode],
        k: int = 3,
    ) -> list[HypothesisNode]:
        """返回 top k 个假设，按评分降序"""
        sorted_results = sorted(results, key=lambda h: h.score, reverse=True)
        return sorted_results[:k]

    def synthesize(
        self,
        results: list[HypothesisNode],
        top_n: int = 3,
    ) -> SynthesisResult:
        """合成最终结论

        1. 选 top_n 个假设
        2. 合并它们的 evidence（去重）
        3. confidence = top 假设的 score（加权）
        """
        top = self.top_k(results, k=top_n)
        if not top:
            raise ValueError("No results to synthesize")

        # 合并证据（去重，保留顺序）
        seen = set()
        combined_evidence = []
        for h in top:
            for e in h.evidence:
                if e not in seen:
                    seen.add(e)
                    combined_evidence.append(e)

        # confidence = 最佳 score × top 假设平均分的加权和
        best_score = top[0].score
        avg_top_score = sum(h.score for h in top) / len(top)
        confidence = best_score * 0.7 + avg_top_score * 0.3

        return SynthesisResult(
            best_hypothesis_id=top[0].id,
            best_hypothesis_description=top[0].description,
            combined_evidence=combined_evidence,
            confidence=confidence,
            top_hypotheses=top,
        )


# ============================================================
# 候选归一化三元键（审查报告附录 B · 代码改动 #5）
# ============================================================

# 因果类型别名映射：把口语化/不同来源的 cause_type 收敛到标准枚举
# 设计原则：保守映射，未识别的 token 不强行归类（保留原值）
CAUSE_TYPE_ALIAS_MAP: dict[str, str] = {
    # CPU
    "cpu": "cpu", "cpu_high": "cpu", "cpu_spike": "cpu", "cpu_saturation": "cpu",
    "high_cpu": "cpu", "cpu_load": "cpu",
    # 内存
    "memory": "memory", "mem": "memory", "oom": "memory", "memory_pressure": "memory",
    "memory_leak": "memory", "high_memory": "memory", "mem_high": "memory",
    # 磁盘
    "disk": "disk", "io": "disk", "io_saturation": "disk", "disk_full": "disk",
    "high_disk": "disk", "io_wait": "disk",
    # 网络
    "network": "network", "net": "network", "latency": "network",
    "network_partition": "network", "packet_loss": "network", "high_latency": "network",
    # 数据库
    "db": "db", "database": "db", "sql": "db", "mysql": "db", "postgres": "db",
    "connection_pool": "db", "slow_query": "db", "deadlock": "db",
    # 缓存
    "cache": "cache", "redis": "cache", "memcached": "cache", "cache_miss": "cache",
    # 部署/变更
    "deployment": "deployment", "release": "deployment", "rollback": "deployment",
    "config_change": "config_change", "config": "config_change",
    "feature_flag": "config_change",
    # 外部依赖
    "upstream": "upstream", "downstream": "upstream", "third_party": "upstream",
    "external_api": "upstream",
}

# Scope 标准化映射
SCOPE_ALIAS_MAP: dict[str, str] = {
    "service": "service", "svc": "service",
    "cluster": "cluster", "node": "cluster", "instance": "cluster",
    "global": "global", "all": "global",
    "region": "region", "zone": "region", "az": "region",
    "shard": "shard", "partition": "shard",
}


def normalize_candidate_key(
    entity_id: str | None,
    cause_type: str | None,
    scope: str | None,
) -> tuple[str, str, str]:
    """
    把候选归一化为 (entity_id, cause_type, scope) 三元键。

    规则：
    - 全部转小写 + strip 空白
    - entity_id：None 或空 → "unknown"
    - cause_type：通过 CAUSE_TYPE_ALIAS_MAP 收敛；未命中 → 原值（小写）
    - scope：通过 SCOPE_ALIAS_MAP 收敛；未命中 → 原值（小写）
    - 输入不是字符串（int/None）→ 强转 str

    返回的 tuple 可直接用作 dict key / set member。
    """
    # entity_id
    if entity_id is None:
        eid = "unknown"
    else:
        eid = str(entity_id).strip().lower() or "unknown"

    # cause_type
    if cause_type is None:
        ct = "unknown"
    else:
        ct_raw = str(cause_type).strip().lower()
        ct = CAUSE_TYPE_ALIAS_MAP.get(ct_raw, ct_raw) if ct_raw else "unknown"

    # scope
    if scope is None:
        sc = "unknown"
    else:
        sc_raw = str(scope).strip().lower()
        sc = SCOPE_ALIAS_MAP.get(sc_raw, sc_raw) if sc_raw else "unknown"

    return (eid, ct, sc)


def normalize_hypothesis_key(hyp: HypothesisNode | dict) -> tuple[str, str, str]:
    """
    从 HypothesisNode 或 dict 抽取三元键的便捷函数。

    兼容 dict 输入（用于反序列化 JSONL 或 RCA V2 候选格式）。
    """
    if isinstance(hyp, HypothesisNode):
        return normalize_candidate_key(hyp.entity_id, hyp.cause_type, hyp.scope)
    return normalize_candidate_key(
        hyp.get("entity_id"),
        hyp.get("cause_type"),
        hyp.get("scope"),
    )


def deduplicate_candidates(
    candidates: list[HypothesisNode],
) -> list[HypothesisNode]:
    """
    按 (entity_id, cause_type, scope) 三元键去重，相同 key 保留 score 最高的。

    入参：HypothesisNode 列表
    出参：去重后的列表（保序：按原 list 顺序，每个 key 只保留首次遇到的最高分候选）

    设计要点：
    - 稳定去重：相同三元键只保留 score 最高的那条
    - 保序：保留原 list 中"首次遇到该 key 的位置"，方便审计
    - 缺字段候选：以 (id, "unknown", "unknown") 归一（不会被错误合并到有完整 key 的候选）
    """
    best_by_key: dict[tuple[str, str, str], HypothesisNode] = {}
    first_seen_order: list[tuple[str, str, str]] = []

    for hyp in candidates:
        key = normalize_hypothesis_key(hyp)
        if key not in best_by_key:
            best_by_key[key] = hyp
            first_seen_order.append(key)
        else:
            # 同 key：保留分数更高的
            if hyp.score > best_by_key[key].score:
                best_by_key[key] = hyp

    # 按 first_seen_order 顺序输出（稳定）
    return [best_by_key[k] for k in first_seen_order]


def dedupe_stats(
    original: list[HypothesisNode],
    deduped: list[HypothesisNode],
) -> dict[str, int]:
    """
    计算去重统计信息（用于 RCA 报告 + 审计）。

    返回：
        {
            "original_count": int,
            "deduped_count": int,
            "merged_count": int,  # original_count - deduped_count
        }
    """
    return {
        "original_count": len(original),
        "deduped_count": len(deduped),
        "merged_count": len(original) - len(deduped),
    }