"""
测试 LLM 成本核算（审查报告七、中期补 #7）

覆盖范围：
1. compute_cost_usd 数学正确性
2. extract_usage_metadata 三种 langchain 响应格式
3. LLMUsageRecord / LLMUsageRecorder 持久化 + 聚合
4. LLMConfig cost 字段配置生效
5. LLMClient.chat() 返回值包含 cost_usd / token_in / token_out
6. LLMClient.chat() 调用 LLMUsageRecorder 写持久化
7. 现有 1131 个测试不回归（向后兼容）
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import LLMConfig
from app.core.llm_client import LLMCallResult, LLMClient, LLMUnavailable
from app.core.llm_cost import (
    LLMUsageRecord,
    LLMUsageRecorder,
    compute_cost_usd,
    extract_usage_metadata,
    get_llm_usage_recorder,
    reset_llm_usage_recorder,
)


# ============================================================
# 1. compute_cost_usd 数学正确性
# ============================================================

class TestComputeCostUsd:
    def test_basic_glm_rates(self):
        # glm-4-flash 类：input 0.001, output 0.002 / 1k tokens
        # 1000 input + 2000 output = 0.001*1 + 0.002*2 = 0.005
        assert compute_cost_usd(1000, 2000, 0.001, 0.002) == 0.005

    def test_zero_tokens_returns_zero(self):
        assert compute_cost_usd(0, 0, 0.001, 0.002) == 0.0

    def test_zero_rates_returns_zero_but_tokens_still_recorded(self):
        # 即使 cost_usd=0，token 也会被 caller 单独记录，不影响审计
        assert compute_cost_usd(1000, 2000, 0.0, 0.0) == 0.0

    def test_high_volume_gpt4_rates(self):
        # gpt-4o 类：input 0.005, output 0.015 / 1k tokens
        # 10k input + 5k output = 0.005*10 + 0.015*5 = 0.05 + 0.075 = 0.125
        assert compute_cost_usd(10000, 5000, 0.005, 0.015) == pytest.approx(0.125, abs=1e-6)

    def test_fractional_tokens(self):
        # 1234 input + 567 output @ 0.001/0.002
        # = 0.001234 + 0.001134 = 0.002368
        cost = compute_cost_usd(1234, 567, 0.001, 0.002)
        assert cost == pytest.approx(0.002368, abs=1e-6)

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError, match="必须非负"):
            compute_cost_usd(100, 100, -0.001, 0.002)
        with pytest.raises(ValueError, match="必须非负"):
            compute_cost_usd(100, 100, 0.001, -0.002)

    def test_rounds_to_6_decimals(self):
        # 长尾浮点：0.001/3 = 0.000333333... → round 到 6 位 = 0.000333
        cost = compute_cost_usd(1, 0, 0.001, 0.0)
        assert cost == 0.000001  # 1/1000 * 0.001 = 0.000001


# ============================================================
# 2. extract_usage_metadata 三种格式
# ============================================================

class TestExtractUsageMetadata:
    def test_new_langchain_usage_metadata(self):
        # langchain >= 0.1.0 格式
        class FakeMsg:
            usage_metadata = {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
            }
        result = extract_usage_metadata(FakeMsg())
        assert result == {"token_in": 100, "token_out": 50}

    def test_old_openai_response_metadata(self):
        # 旧版 openai 格式
        class FakeMsg:
            response_metadata = {
                "token_usage": {"prompt_tokens": 200, "completion_tokens": 80}
            }
        result = extract_usage_metadata(FakeMsg())
        assert result == {"token_in": 200, "token_out": 80}

    def test_openai_usage_key(self):
        # 部分 provider 用 "usage" 而非 "token_usage"
        class FakeMsg:
            response_metadata = {"usage": {"prompt_tokens": 300, "completion_tokens": 150}}
        result = extract_usage_metadata(FakeMsg())
        assert result == {"token_in": 300, "token_out": 150}

    def test_dict_response(self):
        # 模拟 dict 形态
        result = extract_usage_metadata({"token_in": 400, "token_out": 200})
        assert result == {"token_in": 400, "token_out": 200}

    def test_none_response(self):
        assert extract_usage_metadata(None) == {"token_in": 0, "token_out": 0}

    def test_empty_response(self):
        class Empty:
            pass
        assert extract_usage_metadata(Empty()) == {"token_in": 0, "token_out": 0}

    def test_zero_token_metadata(self):
        class FakeMsg:
            usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        # 即使 metadata 存在但全 0，也允许返回 0（不强制要求必须 >0）
        assert extract_usage_metadata(FakeMsg()) == {"token_in": 0, "token_out": 0}


# ============================================================
# 3. LLMUsageRecord / LLMUsageRecorder 持久化 + 聚合
# ============================================================

@pytest.fixture
def temp_db(monkeypatch):
    """临时 audit.db 文件"""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "audit.db")
        recorder = LLMUsageRecorder(db_path=db_path)
        yield recorder, db_path


class TestLLMUsageRecorder:
    def test_record_returns_id(self, temp_db):
        recorder, _ = temp_db
        rec = LLMUsageRecord(
            incident_id="inc-1",
            model="gpt-4o-mini",
            provider="openai",
            token_in=100,
            token_out=50,
            cost_usd=0.0002,
            latency_ms=500,
            caller_agent="rca_agent",
        )
        rid = recorder.record(rec)
        assert rid is not None
        assert isinstance(rid, int)

    def test_record_persists_to_db(self, temp_db):
        recorder, db_path = temp_db
        recorder.record(LLMUsageRecord(
            incident_id="inc-2", model="glm-5.3", provider="mimimax",
            token_in=1000, token_out=500, cost_usd=0.005, latency_ms=800,
            caller_agent="rca_agent",
        ))
        # 直接查 db 验证
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT incident_id, model, token_in, token_out, cost_usd FROM aiops_model_invocations"
        ).fetchone()
        conn.close()
        assert row == ("inc-2", "glm-5.3", 1000, 500, 0.005)

    def test_record_failure_does_not_raise(self, temp_db, monkeypatch):
        recorder, _ = temp_db

        def _broken_connect(self):
            raise OSError("db broken")
        monkeypatch.setattr(LLMUsageRecorder, "_connect", _broken_connect)
        # 写入失败应返回 None，不抛异常
        rid = recorder.record(LLMUsageRecord(model="x", provider="y"))
        assert rid is None

    def test_aggregate_total(self, temp_db):
        recorder, _ = temp_db
        for i in range(3):
            recorder.record(LLMUsageRecord(
                incident_id=f"inc-{i}", model="gpt-4o-mini", provider="openai",
                token_in=1000, token_out=500, cost_usd=0.005, latency_ms=500,
            ))
        agg = recorder.aggregate()
        assert agg["total_calls"] == 3
        assert agg["total_token_in"] == 3000
        assert agg["total_token_out"] == 1500
        assert agg["total_cost_usd"] == pytest.approx(0.015, abs=1e-6)
        assert agg["currency"] == "USD"
        assert len(agg["by_model"]) == 1
        assert agg["by_model"][0]["calls"] == 3

    def test_aggregate_by_model(self, temp_db):
        recorder, _ = temp_db
        recorder.record(LLMUsageRecord(model="gpt-4o", provider="openai",
                                       token_in=100, token_out=50, cost_usd=0.001))
        recorder.record(LLMUsageRecord(model="gpt-4o", provider="openai",
                                       token_in=200, token_out=100, cost_usd=0.002))
        recorder.record(LLMUsageRecord(model="glm-5.3", provider="mimimax",
                                       token_in=500, token_out=200, cost_usd=0.005))
        agg = recorder.aggregate()
        assert len(agg["by_model"]) == 2
        # 按 cost_usd DESC 排序：glm-5.3 (0.005) 应在前
        assert agg["by_model"][0]["model"] == "glm-5.3"
        assert agg["by_model"][0]["cost_usd"] == pytest.approx(0.005)
        assert agg["by_model"][1]["model"] == "gpt-4o"
        assert agg["by_model"][1]["cost_usd"] == pytest.approx(0.003)

    def test_aggregate_filter_by_incident(self, temp_db):
        recorder, _ = temp_db
        recorder.record(LLMUsageRecord(incident_id="inc-A", model="m1", provider="p1",
                                       token_in=100, token_out=50, cost_usd=0.001))
        recorder.record(LLMUsageRecord(incident_id="inc-B", model="m1", provider="p1",
                                       token_in=200, token_out=100, cost_usd=0.002))
        agg_a = recorder.aggregate(incident_id="inc-A")
        agg_b = recorder.aggregate(incident_id="inc-B")
        assert agg_a["total_calls"] == 1
        assert agg_a["total_token_in"] == 100
        assert agg_b["total_calls"] == 1
        assert agg_b["total_token_in"] == 200

    def test_via_fallback_persisted_as_int(self, temp_db):
        recorder, db_path = temp_db
        recorder.record(LLMUsageRecord(model="x", provider="y", via_fallback=True))
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT via_fallback FROM aiops_model_invocations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        assert row[0] == 1


# ============================================================
# 4. LLMConfig cost 字段配置生效
# ============================================================

class TestLLMConfigCost:
    def test_default_cost_zero(self):
        c = LLMConfig(api_key="test")
        assert c.input_cost_per_1k == 0.0
        assert c.output_cost_per_1k == 0.0
        assert c.cost_currency == "USD"

    def test_custom_cost(self):
        c = LLMConfig(api_key="test", input_cost_per_1k=0.001, output_cost_per_1k=0.002)
        assert c.input_cost_per_1k == 0.001
        assert c.output_cost_per_1k == 0.002

    def test_negative_cost_rejected(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            LLMConfig(api_key="test", input_cost_per_1k=-0.001)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("LLM_INPUT_COST_PER_1K", "0.005")
        monkeypatch.setenv("LLM_OUTPUT_COST_PER_1K", "0.015")
        c = LLMConfig(api_key="test")
        assert c.input_cost_per_1k == 0.005
        assert c.output_cost_per_1k == 0.015


# ============================================================
# 5. LLMClient.chat() 返回值包含 cost / token
# ============================================================

def _make_client_with_mock_chat(token_in: int, token_out: int, cost_input: float, cost_output: float):
    """构造 LLMClient，注入 mock _sync_chat 返回 (content, usage_meta)"""
    config = LLMConfig(
        api_key="test-key",
        model="test-model",
        provider="openai",
        input_cost_per_1k=cost_input,
        output_cost_per_1k=cost_output,
    )
    client = LLMClient(config=config)
    # 直接替换 _sync_chat
    client._sync_chat = MagicMock(
        return_value=("test response", {"token_in": token_in, "token_out": token_out})
    )
    return client


class TestLLMClientChatReturnsCost:
    def test_chat_returns_token_in_out_and_cost(self):
        client = _make_client_with_mock_chat(1000, 500, 0.001, 0.002)
        result = asyncio.run(client.chat(
            messages=[{"role": "user", "content": "hi"}],
            incident_id="inc-1",
            caller_agent="rca_agent",
        ))
        assert isinstance(result, LLMCallResult)
        assert result.token_in == 1000
        assert result.token_out == 500
        assert result.cost_usd == pytest.approx(0.002, abs=1e-6)  # 1*0.001 + 0.5*0.002 = 0.002
        assert result.currency == "USD"
        assert result.via_fallback is False

    def test_chat_zero_cost_when_rate_zero(self):
        client = _make_client_with_mock_chat(1000, 500, 0.0, 0.0)
        result = asyncio.run(client.chat(messages=["hi"]))
        assert result.cost_usd == 0.0
        # token 仍然记录（用于成本核算之外的"使用量"统计）
        assert result.token_in == 1000
        assert result.token_out == 500

    def test_chat_legacy_sync_chat_returns_string(self):
        """兼容：旧版 _sync_chat 只返回 string（不是 tuple）—— 不能破坏"""
        config = LLMConfig(api_key="test-key", model="test-model", provider="openai")
        client = LLMClient(config=config)
        client._sync_chat = MagicMock(return_value="legacy content")  # 只返 string
        result = asyncio.run(client.chat(messages=["hi"]))
        assert result.content == "legacy content"
        assert result.token_in == 0
        assert result.token_out == 0
        assert result.cost_usd == 0.0

    def test_chat_persists_to_usage_recorder(self, monkeypatch):
        # 用临时 db
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "audit.db")
            recorder = LLMUsageRecorder(db_path=db_path)
            monkeypatch.setattr(
                "app.core.llm_client.get_llm_usage_recorder",
                lambda: recorder,
            )

            client = _make_client_with_mock_chat(2000, 1000, 0.001, 0.002)
            asyncio.run(client.chat(
                messages=[{"role": "user", "content": "hi"}],
                incident_id="inc-persist",
                caller_agent="rca_agent",
            ))

            agg = recorder.aggregate(incident_id="inc-persist")
            assert agg["total_calls"] == 1
            assert agg["total_token_in"] == 2000
            assert agg["total_token_out"] == 1000
            assert agg["total_cost_usd"] == pytest.approx(0.004, abs=1e-6)  # 2*0.001 + 1*0.002

    def test_chat_breaker_open_raises_without_persisting(self, monkeypatch):
        # 熔断器 OPEN 不应写持久化（避免污染）
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "audit.db")
            recorder = LLMUsageRecorder(db_path=db_path)
            monkeypatch.setattr(
                "app.core.llm_client.get_llm_usage_recorder",
                lambda: recorder,
            )

            config = LLMConfig(api_key="test-key")
            client = LLMClient(config=config)
            # 强制 OPEN
            client.breaker._state = client.breaker._state.OPEN

            with pytest.raises(LLMUnavailable, match="熔断器"):
                asyncio.run(client.chat(messages=["hi"]))

            agg = recorder.aggregate()
            assert agg["total_calls"] == 0

    def test_chat_records_failure_does_not_break_main(self, monkeypatch):
        # 持久化失败不影响主链路（fire-and-forget）
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "audit.db")
            recorder = LLMUsageRecorder(db_path=db_path)

            def broken_record(_):
                raise OSError("disk full")
            monkeypatch.setattr(recorder, "record", broken_record)
            monkeypatch.setattr(
                "app.core.llm_client.get_llm_usage_recorder",
                lambda: recorder,
            )

            client = _make_client_with_mock_chat(100, 50, 0.001, 0.002)
            # 不应抛异常
            result = asyncio.run(client.chat(messages=["hi"]))
            assert result.cost_usd == pytest.approx(0.0002, abs=1e-6)
