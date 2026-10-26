"""
测试候选归一化三元键（审查报告附录 B · 代码改动 #5）

覆盖：
1. normalize_candidate_key: 大小写、空白、None、别名映射
2. normalize_hypothesis_key: HypothesisNode + dict 输入
3. deduplicate_candidates: 相同 key 保留 score 最高
4. dedupe_stats: 统计正确
5. 向后兼容：现有 HypothesisNode（无 entity_id/cause_type/scope）不破坏
"""
from __future__ import annotations

import pytest

from app.agents._bfs_synthesis import (
    CAUSE_TYPE_ALIAS_MAP,
    SCOPE_ALIAS_MAP,
    HypothesisNode,
    dedupe_stats,
    deduplicate_candidates,
    normalize_candidate_key,
    normalize_hypothesis_key,
)


# ============================================================
# 1. normalize_candidate_key
# ============================================================

class TestNormalizeCandidateKey:
    def test_basic_three_fields(self):
        assert normalize_candidate_key("order-service", "cpu", "service") == (
            "order-service", "cpu", "service"
        )

    def test_lowercase_and_strip(self):
        assert normalize_candidate_key(
            "  ORDER-Service  ", " CPU_HIGH ", " Service "
        ) == ("order-service", "cpu", "service")

    def test_none_entity_id_becomes_unknown(self):
        assert normalize_candidate_key(None, "cpu", "service")[0] == "unknown"

    def test_empty_entity_id_becomes_unknown(self):
        assert normalize_candidate_key("", "cpu", "service")[0] == "unknown"

    def test_whitespace_entity_id_becomes_unknown(self):
        assert normalize_candidate_key("   ", "cpu", "service")[0] == "unknown"

    def test_unknown_cause_type_keeps_normalized_value(self):
        # 未在别名表里 → 原值（小写）
        assert normalize_candidate_key("svc", "CustomCause", "service")[1] == "customcause"

    def test_alias_cause_type_cpu_variants(self):
        # 多种 cpu 写法都收敛到 cpu
        for variant in ["CPU", "cpu_high", "CPU_Spike", "high_cpu", "Cpu_Load"]:
            result = normalize_candidate_key("svc", variant, "cluster")
            assert result[1] == "cpu", f"variant {variant} should map to cpu"

    def test_alias_cause_type_memory(self):
        for variant in ["memory", "OOM", "mem", "Memory_Leak", "high_memory"]:
            result = normalize_candidate_key("svc", variant, "cluster")
            assert result[1] == "memory", f"variant {variant} should map to memory"

    def test_alias_cause_type_disk(self):
        for variant in ["disk", "IO", "io_saturation", "disk_full"]:
            result = normalize_candidate_key("svc", variant, "cluster")
            assert result[1] == "disk"

    def test_alias_cause_type_db(self):
        for variant in ["db", "database", "mysql", "postgres", "slow_query", "deadlock"]:
            result = normalize_candidate_key("svc", variant, "cluster")
            assert result[1] == "db"

    def test_alias_scope_zone_to_region(self):
        assert normalize_candidate_key("svc", "cpu", "zone")[2] == "region"
        assert normalize_candidate_key("svc", "cpu", "AZ")[2] == "region"

    def test_alias_scope_node_to_cluster(self):
        assert normalize_candidate_key("svc", "cpu", "node")[2] == "cluster"
        assert normalize_candidate_key("svc", "cpu", "instance")[2] == "cluster"

    def test_non_string_input_coerced(self):
        # 数字 / 列表都强转 str
        assert normalize_candidate_key(123, "cpu", 456) == ("123", "cpu", "456")

    def test_none_cause_type_and_scope_become_unknown(self):
        result = normalize_candidate_key("svc", None, None)
        assert result == ("svc", "unknown", "unknown")

    def test_all_none_returns_three_unknowns(self):
        assert normalize_candidate_key(None, None, None) == (
            "unknown", "unknown", "unknown"
        )


# ============================================================
# 2. normalize_hypothesis_key: HypothesisNode + dict
# ============================================================

class TestNormalizeHypothesisKey:
    def test_from_hypothesis_node(self):
        h = HypothesisNode(
            id="h1", description="CPU spike",
            entity_id="order-service", cause_type="CPU_HIGH", scope="Service"
        )
        assert normalize_hypothesis_key(h) == ("order-service", "cpu", "service")

    def test_from_hypothesis_node_missing_fields(self):
        h = HypothesisNode(id="h1", description="x")
        assert normalize_hypothesis_key(h) == ("unknown", "unknown", "unknown")

    def test_from_dict(self):
        d = {"entity_id": "payment-svc", "cause_type": "OOM", "scope": "Cluster"}
        assert normalize_hypothesis_key(d) == ("payment-svc", "memory", "cluster")

    def test_from_partial_dict(self):
        d = {"entity_id": "auth-svc"}  # 只给 entity_id
        assert normalize_hypothesis_key(d) == ("auth-svc", "unknown", "unknown")

    def test_node_and_dict_equivalent(self):
        h = HypothesisNode(
            id="h1", description="x",
            entity_id="svc", cause_type="cpu", scope="cluster"
        )
        d = {"entity_id": "svc", "cause_type": "cpu", "scope": "cluster"}
        assert normalize_hypothesis_key(h) == normalize_hypothesis_key(d)


# ============================================================
# 3. deduplicate_candidates
# ============================================================

def _h(id_: str, desc: str, score: float, eid=None, ct=None, sc=None) -> HypothesisNode:
    return HypothesisNode(
        id=id_, description=desc, score=score,
        entity_id=eid, cause_type=ct, scope=sc,
    )


class TestDeduplicateCandidates:
    def test_empty_list(self):
        assert deduplicate_candidates([]) == []

    def test_no_duplicates(self):
        hyps = [
            _h("h1", "CPU", 0.9, eid="svc1", ct="cpu", sc="service"),
            _h("h2", "MEM", 0.8, eid="svc1", ct="memory", sc="service"),
            _h("h3", "DB",  0.7, eid="svc1", ct="db", sc="service"),
        ]
        result = deduplicate_candidates(hyps)
        assert len(result) == 3
        assert [h.id for h in result] == ["h1", "h2", "h3"]

    def test_duplicate_key_keeps_higher_score(self):
        hyps = [
            _h("h1", "CPU first", 0.7, eid="svc1", ct="cpu", sc="service"),
            _h("h2", "CPU second (higher)", 0.9, eid="svc1", ct="cpu", sc="service"),
            _h("h3", "CPU third", 0.6, eid="svc1", ct="cpu", sc="service"),
        ]
        result = deduplicate_candidates(hyps)
        assert len(result) == 1
        assert result[0].id == "h2"  # score 最高

    def test_different_entity_kept(self):
        hyps = [
            _h("h1", "x", 0.5, eid="svc1", ct="cpu", sc="service"),
            _h("h2", "y", 0.5, eid="svc2", ct="cpu", sc="service"),  # 不同 entity_id
        ]
        result = deduplicate_candidates(hyps)
        assert len(result) == 2

    def test_different_cause_kept(self):
        hyps = [
            _h("h1", "x", 0.5, eid="svc1", ct="cpu", sc="service"),
            _h("h2", "y", 0.5, eid="svc1", ct="memory", sc="service"),  # 不同 cause_type
        ]
        result = deduplicate_candidates(hyps)
        assert len(result) == 2

    def test_different_scope_kept(self):
        hyps = [
            _h("h1", "x", 0.5, eid="svc1", ct="cpu", sc="service"),
            _h("h2", "y", 0.5, eid="svc1", ct="cpu", sc="cluster"),  # 不同 scope
        ]
        result = deduplicate_candidates(hyps)
        assert len(result) == 2

    def test_alias_collapses_duplicates(self):
        # "CPU" 和 "cpu_high" 通过别名映射 → 同一 key
        hyps = [
            _h("h1", "CPU", 0.7, eid="svc1", ct="CPU", sc="service"),
            _h("h2", "CPU_HIGH", 0.85, eid="svc1", ct="cpu_high", sc="SERVICE"),
        ]
        result = deduplicate_candidates(hyps)
        assert len(result) == 1
        assert result[0].id == "h2"  # 分数更高

    def test_missing_fields_treated_as_unknown(self):
        # 缺字段的候选 → (id, "unknown", "unknown")，不会被错误合并
        h_with_fields = _h("h1", "x", 0.9, eid="svc1", ct="cpu", sc="service")
        h_no_fields_1 = _h("h2", "y", 0.7)
        h_no_fields_2 = _h("h3", "z", 0.5)
        result = deduplicate_candidates([h_with_fields, h_no_fields_1, h_no_fields_2])
        # h1 独立；h2 和 h3 同 key（unknown,unknown,unknown）合并 → h2（高分）
        assert len(result) == 2
        assert result[0].id == "h1"
        assert result[1].id == "h2"

    def test_preserves_first_seen_order(self):
        hyps = [
            _h("h1", "low", 0.5, eid="svc", ct="cpu", sc="service"),
            _h("h2", "high", 0.9, eid="svc", ct="cpu", sc="service"),
            _h("h3", "different", 0.7, eid="svc2", ct="cpu", sc="service"),
        ]
        result = deduplicate_candidates(hyps)
        # h1 和 h2 同 key，h2 胜出（高分）
        # h1 的位置保留（first_seen），h2 替代 h1
        # h3 不同 key，单独
        assert [h.id for h in result] == ["h2", "h3"]

    def test_legacy_node_without_fields(self):
        """向后兼容：旧版 HypothesisNode（无三元键字段）能正常去重"""
        hyps = [
            HypothesisNode(id="h1", description="x"),  # 旧风格
            HypothesisNode(id="h2", description="y"),  # 旧风格
        ]
        # 都归一到 ("unknown","unknown","unknown")，合并保留高分
        result = deduplicate_candidates(hyps)
        assert len(result) == 1


# ============================================================
# 4. dedupe_stats
# ============================================================

class TestDedupeStats:
    def test_basic(self):
        original = [_h("h1", "x", 0.5), _h("h2", "y", 0.5), _h("h3", "z", 0.5)]
        deduped = [_h("h1", "x", 0.5), _h("h3", "z", 0.5)]
        stats = dedupe_stats(original, deduped)
        assert stats == {"original_count": 3, "deduped_count": 2, "merged_count": 1}

    def test_empty(self):
        stats = dedupe_stats([], [])
        assert stats["merged_count"] == 0

    def test_no_duplicates(self):
        original = [_h("h1", "x", 0.5), _h("h2", "y", 0.5)]
        stats = dedupe_stats(original, original)
        assert stats["merged_count"] == 0
