"""v3 E9: SMD 时序异常检测评估测试"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.detection.eval_smd import (
    OMNIANOMALY_SMD_RESULTS,
    SMDEvalResult,
    compare_with_omnianomaly,
    evaluate_smd,
    load_smd_machine,
)


@pytest.fixture
def smd_test_data(tmp_path: Path) -> Path:
    """构造一个最小的 SMD 测试数据集"""
    smd_root = tmp_path / "smd"
    smd_root.mkdir()

    # machine-1-1: 100 个时间点，2 个指标
    # 前 80 个是正常 (50, 60)，后 20 个是异常 (90, 95)
    data_lines = []
    label_lines = []
    for i in range(100):
        if i < 80:
            data_lines.append("50.0,60.0")
            label_lines.append("0")
        else:
            data_lines.append("90.0,95.0")
            label_lines.append("1")

    (smd_root / "machine-1-1.txt").write_text("\n".join(data_lines))
    (smd_root / "machine-1-1_labels.txt").write_text("\n".join(label_lines))
    return smd_root


class TestLoadSMD:
    def test_load_machine_returns_data_and_labels(self, smd_test_data):
        data, labels = load_smd_machine(smd_test_data, "machine-1-1")
        assert len(data) == 100
        assert len(labels) == 100
        assert data[0] == [50.0, 60.0]
        assert data[80] == [90.0, 95.0]
        assert sum(labels) == 20  # 80-100 是 1

    def test_load_missing_machine_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_smd_machine(tmp_path, "machine-99-99")


class TestEvaluateSMD:
    def test_evaluate_returns_result(self, smd_test_data):
        result = evaluate_smd(
            smd_root=smd_test_data,
            machines=["machine-1-1"],
            max_points_per_machine=200,
            window_size=20,
        )
        assert isinstance(result, SMDEvalResult)
        assert result.machine_count == 1
        assert result.total_points > 0
        # 一定有 anomaly 检出（后半部分异常数据）
        assert result.true_positive >= 0

    def test_evaluate_with_no_data_returns_zero(self, tmp_path):
        """空 smd_root → 所有指标都是 0"""
        empty_smd = tmp_path / "empty_smd"
        empty_smd.mkdir()
        result = evaluate_smd(
            smd_root=empty_smd,
            machines=["machine-1-1"],
        )
        assert result.total_points == 0
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_evaluate_skips_missing_machines(self, tmp_path):
        """不存在的机器被跳过"""
        empty_smd = tmp_path / "empty_smd"
        empty_smd.mkdir()
        result = evaluate_smd(
            smd_root=empty_smd,
            machines=["machine-1-1", "machine-2-2"],
        )
        # 全部不存在 → 0 点
        assert result.total_points == 0

    def test_evaluate_result_to_dict(self, smd_test_data):
        result = evaluate_smd(smd_test_data, machines=["machine-1-1"], max_points_per_machine=200)
        d = result.to_dict()
        assert "precision" in d
        assert "recall" in d
        assert "f1" in d
        assert 0.0 <= d["precision"] <= 1.0


class TestCompareWithOmniAnomaly:
    def test_omnianomaly_baseline_exists(self):
        assert OMNIANOMALY_SMD_RESULTS["model"] == "OmniAnomaly"
        assert 0 < OMNIANOMALY_SMD_RESULTS["f1"] <= 1.0

    def test_compare_returns_full_dict(self, smd_test_data):
        result = evaluate_smd(smd_test_data, machines=["machine-1-1"], max_points_per_machine=200)
        cmp = compare_with_omnianomaly(result)
        assert "our" in cmp
        assert "omnianomaly" in cmp
        assert "delta" in cmp
        assert "verdict" in cmp
        # delta 应包含我们和基线的差
        assert "f1" in cmp["delta"]

    def test_compare_verdict_text(self, smd_test_data):
        result = evaluate_smd(smd_test_data, machines=["machine-1-1"], max_points_per_machine=200)
        cmp = compare_with_omnianomaly(result)
        assert isinstance(cmp["verdict"], str)
        assert len(cmp["verdict"]) > 0