"""
AIOps Agent Platform - MemoryEntry 验证状态测试

测试覆盖：
1. MemoryEntry 模型字段（含 verification_status 等顶层字段）
2. LongTermMemory.record_verification() 晋升逻辑
3. LongTermMemory.record_verifications_batch() 批量验证
4. LongTermMemory.get_verification_stats() 统计
5. RCAAgent._extract_verification_status() 读取兼容
6. v2 设计哲学：unverified 记忆不参与加成

关联：审查报告 v1.2 代码改动 #1（Memory verification_status 字段）
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents.rca_agent import RCAAgent
from app.memory.long_term import LongTermMemory
from app.memory.storage import BaseStorage
from app.models.memory import (
    VERIFICATION_BOOST_MAP,
    MemoryEntry,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Mock Storage（不依赖 ChromaDB）
# ---------------------------------------------------------------------------


class MockStorage(BaseStorage):
    """测试用内存存储"""

    def __init__(self) -> None:
        self.entries: dict[str, MemoryEntry] = {}

    async def add(self, entry: MemoryEntry) -> None:
        self.entries[entry.memory_id] = entry

    async def update(self, entry: MemoryEntry) -> bool:
        self.entries[entry.memory_id] = entry
        return True

    async def get(self, memory_id: str) -> MemoryEntry | None:
        return self.entries.get(memory_id)

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[MemoryEntry]:
        return []

    async def get_by_type(
        self,
        memory_level=None,
        limit: int = 10000,
    ) -> list[MemoryEntry]:
        return list(self.entries.values())[:limit]

    async def delete(self, memory_id: str) -> bool:
        if memory_id in self.entries:
            del self.entries[memory_id]
            return True
        return False

    async def clear(self) -> None:
        self.entries.clear()

    async def get_stats(self) -> dict:
        return {"count": len(self.entries)}


@pytest.fixture
def mock_storage() -> MockStorage:
    return MockStorage()


@pytest.fixture
def ltm(mock_storage: MockStorage) -> LongTermMemory:
    return LongTermMemory(storage=mock_storage, similarity_threshold=0.85)


# ---------------------------------------------------------------------------
# 1. MemoryEntry 模型字段
# ---------------------------------------------------------------------------


class TestMemoryEntryFields:
    """测试 MemoryEntry 新增字段（v2 缺口 4 修复）"""

    def test_default_verification_status_is_unverified(self):
        """新写入的记忆默认是 UNVERIFIED（v2 设计哲学：未验证 = 不加成）"""
        e = MemoryEntry(content="test")
        assert e.verification_status == VerificationStatus.UNVERIFIED
        assert e.verification_count == 0
        assert e.last_verified_at is None
        assert e.verified_by == []

    def test_verification_boost_mapping(self):
        """三档状态对应的加成系数"""
        e_unverified = MemoryEntry(content="t")
        e_provisional = MemoryEntry(
            content="t",
            verification_status=VerificationStatus.PROVISIONAL,
        )
        e_verified = MemoryEntry(
            content="t",
            verification_status=VerificationStatus.VERIFIED,
        )

        assert e_unverified.verification_boost == 0.0
        assert e_provisional.verification_boost == 0.05
        assert e_verified.verification_boost == 0.15

    def test_verification_boost_map_module_constant(self):
        """模块级 VERIFICATION_BOOST_MAP 与 property 一致"""
        assert VERIFICATION_BOOST_MAP[VerificationStatus.UNVERIFIED] == 0.0
        assert VERIFICATION_BOOST_MAP[VerificationStatus.PROVISIONAL] == 0.05
        assert VERIFICATION_BOOST_MAP[VerificationStatus.VERIFIED] == 0.15

    def test_verified_by_accepts_duplicate_at_field_level(self):
        """Pydantic 模型层不做去重（去重是 record_verification 的职责）"""
        e = MemoryEntry(content="t", verified_by=["a", "a", "b"])
        # 模型接受重复值；去重由 LongTermMemory.record_verification() 在写入时处理
        assert len(e.verified_by) == 3
        # 验证 record_verification() 确实去重
        # （这条逻辑在 TestRecordVerificationPromotion 中覆盖）

    def test_metadata_backward_compatibility(self):
        """旧数据 metadata['verification_status'] 仍可读（向后兼容）"""
        e = MemoryEntry(content="t", metadata={"verification_status": "verified"})
        assert e.metadata.get("verification_status") == "verified"


# ---------------------------------------------------------------------------
# 2. LongTermMemory.record_verification() 晋升逻辑
# ---------------------------------------------------------------------------


class TestRecordVerificationPromotion:
    """测试 record_verification() 的晋升规则"""

    async def test_first_verify_promotes_to_provisional(self, ltm: LongTermMemory):
        """第 1 次验证 → PROVISIONAL"""
        e = MemoryEntry(content="payment timeout")
        await ltm.store(e)

        ok, status = await ltm.record_verification(e.memory_id, "incident_001")

        assert ok is True
        assert status == VerificationStatus.PROVISIONAL
        assert e.verification_count == 1
        assert e.verified_by == ["incident_001"]
        assert e.last_verified_at is not None

    async def test_same_verifier_does_not_count_as_new_source(
        self, ltm: LongTermMemory
    ):
        """同一 verifier 多次验证，不增加 distinct_verifiers 数"""
        e = MemoryEntry(content="db connection leak")
        await ltm.store(e)

        await ltm.record_verification(e.memory_id, "incident_001")
        await ltm.record_verification(e.memory_id, "incident_001")
        await ltm.record_verification(e.memory_id, "incident_001")

        # 3 次验证但只有 1 个来源 → 仍 PROVISIONAL
        assert e.verification_count == 3
        assert e.verified_by == ["incident_001"]
        assert e.verification_status == VerificationStatus.PROVISIONAL

    async def test_three_verifies_two_sources_promotes_to_verified(
        self, ltm: LongTermMemory
    ):
        """≥3 次验证 + ≥2 个不同来源 → VERIFIED"""
        e = MemoryEntry(content="cpu spike")
        await ltm.store(e)

        await ltm.record_verification(e.memory_id, "incident_001")
        await ltm.record_verification(e.memory_id, "incident_002")
        await ltm.record_verification(e.memory_id, "incident_003")

        assert e.verification_count == 3
        assert len(e.verified_by) == 3
        assert e.verification_status == VerificationStatus.VERIFIED

    async def test_two_sources_two_verifies_still_provisional(
        self, ltm: LongTermMemory
    ):
        """2 来源 + 2 次验证 → 仍 PROVISIONAL（< 3 次验证要求）"""
        e = MemoryEntry(content="disk full")
        await ltm.store(e)

        await ltm.record_verification(e.memory_id, "incident_001")
        await ltm.record_verification(e.memory_id, "incident_002")

        assert e.verification_count == 2
        assert len(e.verified_by) == 2
        # 验证次数 < 3 → 不能升 VERIFIED
        assert e.verification_status == VerificationStatus.PROVISIONAL

    async def test_nonexistent_memory_id_returns_failure(
        self, ltm: LongTermMemory
    ):
        """不存在的 memory_id 返回 (False, UNVERIFIED)，不抛异常"""
        ok, status = await ltm.record_verification("nonexistent_id", "x")
        assert ok is False
        assert status == VerificationStatus.UNVERIFIED

    async def test_metadata_synced_after_verification(
        self, ltm: LongTermMemory
    ):
        """验证后 metadata['verification_status'] 同步更新（向后兼容）"""
        e = MemoryEntry(content="oom")
        await ltm.store(e)

        await ltm.record_verification(e.memory_id, "incident_001")
        assert e.metadata.get("verification_status") == "provisional"

        await ltm.record_verification(e.memory_id, "incident_002")
        await ltm.record_verification(e.memory_id, "incident_003")
        assert e.metadata.get("verification_status") == "verified"


# ---------------------------------------------------------------------------
# 3. LongTermMemory.record_verifications_batch() 批量验证
# ---------------------------------------------------------------------------


class TestBatchVerification:
    """测试批量验证（RCA 完成后回填）"""

    async def test_batch_records_multiple_memories(self, ltm: LongTermMemory):
        """批量验证多记忆，每条独立晋升"""
        e1 = MemoryEntry(content="e1")
        e2 = MemoryEntry(content="e2")
        await ltm.store(e1)
        await ltm.store(e2)

        results = await ltm.record_verifications_batch(
            [e1.memory_id, e2.memory_id],
            verifier="agent_harness",
            note="RCA 验证通过",
        )

        assert e1.memory_id in results
        assert e2.memory_id in results
        assert results[e1.memory_id] == VerificationStatus.PROVISIONAL
        assert results[e2.memory_id] == VerificationStatus.PROVISIONAL

    async def test_batch_skips_nonexistent_ids(self, ltm: LongTermMemory):
        """批量跳过不存在的 ID"""
        e1 = MemoryEntry(content="e1")
        await ltm.store(e1)

        results = await ltm.record_verifications_batch(
            [e1.memory_id, "nonexistent"],
            verifier="incident_x",
        )

        assert e1.memory_id in results
        assert "nonexistent" not in results


# ---------------------------------------------------------------------------
# 4. LongTermMemory.get_verification_stats() 统计
# ---------------------------------------------------------------------------


class TestVerificationStats:
    """测试验证状态分布统计"""

    async def test_empty_storage_returns_zeros(self, ltm: LongTermMemory):
        """空存储 → 全 0"""
        stats = await ltm.get_verification_stats()
        assert stats["total"] == 0
        assert stats["unverified"] == 0
        assert stats["provisional"] == 0
        assert stats["verified"] == 0

    async def test_stats_reflects_current_state(self, ltm: LongTermMemory):
        """统计反映当前状态分布"""
        # 1 unverified
        e1 = MemoryEntry(content="e1")
        await ltm.store(e1)

        # 1 provisional
        e2 = MemoryEntry(content="e2")
        await ltm.store(e2)
        await ltm.record_verification(e2.memory_id, "inc_1")

        # 1 verified
        e3 = MemoryEntry(content="e3")
        await ltm.store(e3)
        await ltm.record_verification(e3.memory_id, "inc_1")
        await ltm.record_verification(e3.memory_id, "inc_2")
        await ltm.record_verification(e3.memory_id, "inc_3")

        stats = await ltm.get_verification_stats()
        assert stats["total"] == 3
        assert stats["unverified"] == 1
        assert stats["provisional"] == 1
        assert stats["verified"] == 1


# ---------------------------------------------------------------------------
# 5. RCAAgent._extract_verification_status() 读取兼容
# ---------------------------------------------------------------------------


class TestExtractVerificationStatus:
    """测试 RCAAgent._extract_verification_status() 的读取兼容性"""

    def test_top_level_verified(self):
        """顶层字段 VERIFIED → 返回 'verified'"""
        e = MemoryEntry(content="t", verification_status=VerificationStatus.VERIFIED)
        assert RCAAgent._extract_verification_status(e) == "verified"

    def test_top_level_provisional(self):
        """顶层字段 PROVISIONAL → 返回 'provisional'"""
        e = MemoryEntry(
            content="t", verification_status=VerificationStatus.PROVISIONAL
        )
        assert RCAAgent._extract_verification_status(e) == "provisional"

    def test_default_is_unverified(self):
        """默认 UNVERIFIED → 返回 'unverified'（v2 设计哲学）"""
        e = MemoryEntry(content="t")
        assert RCAAgent._extract_verification_status(e) == "unverified"

    def test_top_level_wins_over_legacy_metadata(self):
        """顶层字段胜出 metadata（v2 缺省值设计：未验证 = 不加成）"""
        e = MemoryEntry(
            content="t",
            metadata={"verification_status": "verified"},
        )
        # 即使 metadata 写 verified，顶层默认 UNVERIFIED，胜出
        assert RCAAgent._extract_verification_status(e) == "unverified"

    def test_string_value_supported(self):
        """支持字符串值（非 Enum）"""
        class FakeEntry:
            verification_status = "verified"

        assert RCAAgent._extract_verification_status(FakeEntry()) == "verified"

    def test_empty_object_returns_unverified(self):
        """空对象 → 默认 'unverified'"""
        class Empty:
            pass

        assert RCAAgent._extract_verification_status(Empty()) == "unverified"

    def test_none_returns_unverified(self):
        """None 输入 → 'unverified'（不抛异常）"""
        assert RCAAgent._extract_verification_status(None) == "unverified"

    def test_dict_input(self):
        """支持 dict 输入（少数场景）"""
        e = {"verification_status": "provisional"}
        assert RCAAgent._extract_verification_status(e) == "provisional"


# ---------------------------------------------------------------------------
# 6. v2 设计哲学集成测试
# ---------------------------------------------------------------------------


class TestV2DesignPhilosophy:
    """测试 v2 设计哲学：未验证记忆不参与加成"""

    async def test_unverified_memory_returns_zero_boost(
        self, ltm: LongTermMemory
    ):
        """未验证记忆的 verification_boost = 0.0"""
        e = MemoryEntry(content="new lesson learned")
        await ltm.store(e)

        # 未验证时 boost = 0
        assert e.verification_boost == 0.0

    async def test_provisional_memory_partial_boost(
        self, ltm: LongTermMemory
    ):
        """PROVISIONAL 记忆加成 0.05（受限加成）"""
        e = MemoryEntry(content="lesson")
        await ltm.store(e)
        await ltm.record_verification(e.memory_id, "inc_1")

        assert e.verification_boost == 0.05

    async def test_verified_memory_full_boost(
        self, ltm: LongTermMemory
    ):
        """VERIFIED 记忆加成 0.15（MAX_MEMORY_BOOST）"""
        e = MemoryEntry(content="lesson")
        await ltm.store(e)
        await ltm.record_verification(e.memory_id, "inc_1")
        await ltm.record_verification(e.memory_id, "inc_2")
        await ltm.record_verification(e.memory_id, "inc_3")

        assert e.verification_boost == 0.15


# ---------------------------------------------------------------------------
# 模块级测试 runner（方便独立运行）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 允许直接 python -m test_memory_verification_status 跑
    asyncio.run(_run_all())


async def _run_all() -> None:
    """手动运行所有测试（pytest 不可用时的备用）"""
    storage = MockStorage()
    ltm = LongTermMemory(storage=storage)

    # 模型字段
    e = MemoryEntry(content="t")
    assert e.verification_status == VerificationStatus.UNVERIFIED
    print("✓ MemoryEntry defaults")

    # 晋升逻辑
    await ltm.store(e)
    await ltm.record_verification(e.memory_id, "inc_1")
    assert e.verification_status == VerificationStatus.PROVISIONAL
    print("✓ 1 verify -> PROVISIONAL")

    await ltm.record_verification(e.memory_id, "inc_2")
    await ltm.record_verification(e.memory_id, "inc_3")
    assert e.verification_status == VerificationStatus.VERIFIED
    print("✓ 3 verifies + 2 sources -> VERIFIED")

    # 读取兼容
    assert RCAAgent._extract_verification_status(e) == "verified"
    print("✓ _extract_verification_status works")

    print()
    print("ALL SMOKE TESTS PASSED")
