"""
M2 - 4 层幻觉控制 Implementation (GREEN 阶段)

借鉴 Troubleshooter §4 - 4 层防护：
1. Input Validation: 输入校验
2. Evidence Grounding: 输出必须基于 evidence
3. Cross-Check: 多源交叉验证
4. Output Schema: 输出 schema 校验

只做测试要求的功能：
- 4 个独立 validator
- 1 个 pipeline 串联
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    is_valid: bool
    reason: str = ""
    unsupported_claims: list[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """别名：grounding 检查专用"""
        return self.is_valid


@dataclass
class CrossCheckResult:
    confirmed: bool
    confidence: float
    has_conflict: bool = False


@dataclass
class PipelineResult:
    passed: bool
    failure_layer: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ============================================================
# Layer 1: Input Validation
# ============================================================
class InputValidator:
    """第 1 层：输入校验"""

    REQUIRED_FIELDS = {"incident_id", "service", "alert_name", "evidence"}

    def validate(self, data: dict[str, Any]) -> ValidationResult:
        if not data:
            return ValidationResult(
                is_valid=False,
                reason="Empty input: required fields are missing",
            )

        missing = self.REQUIRED_FIELDS - set(data.keys())
        if missing:
            return ValidationResult(
                is_valid=False,
                reason=f"Missing required fields: {sorted(missing)}",
            )

        # 检查 evidence 非空
        if not data.get("evidence"):
            return ValidationResult(
                is_valid=False,
                reason="evidence must be non-empty list",
            )

        return ValidationResult(is_valid=True)


# ============================================================
# Layer 2: Evidence Grounding
# ============================================================
class EvidenceGroundingChecker:
    """第 2 层：证据落地检查"""

    # 简单数值/事实提取正则
    _NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    _METRIC_KEYWORDS = {
        "error_rate", "cpu", "memory", "latency", "throughput",
        "qps", "rps", "conn", "pool", "disk", "load",
    }

    def __init__(self, evidence: list[str]) -> None:
        self.evidence = evidence
        # 把 evidence 拼成一个文本，便于子串匹配
        self._evidence_text = " ".join(evidence).lower()

    def check(self, output: str) -> ValidationResult:
        """检查 output 中的事实是否在 evidence 中有支撑

        简单策略：
        1. 提取 output 中的数字
        2. 数字必须出现在 evidence 中（否则视为幻觉）
        """
        output_numbers = self._NUMBER_RE.findall(output)
        output_lower = output.lower()

        # 提取 output 中的指标关键词
        metric_keywords_in_output = [
            kw for kw in self._METRIC_KEYWORDS
            if kw in output_lower
        ]

        unsupported: list[str] = []

        # 检查数字是否在 evidence 中
        for num in output_numbers:
            if num not in self._evidence_text:
                unsupported.append(f"value:{num}")

        # 检查 metric 关键词是否在 evidence 中
        for kw in metric_keywords_in_output:
            if kw not in self._evidence_text:
                unsupported.append(f"metric:{kw}")

        return ValidationResult(
            is_valid=len(unsupported) == 0,
            reason="OK" if not unsupported else f"unsupported claims: {unsupported}",
            unsupported_claims=unsupported,
        )


# ============================================================
# Layer 3: Cross-Check
# ============================================================
class CrossChecker:
    """第 3 层：交叉验证"""

    MIN_SOURCES_FOR_CONFIRM = 2
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    LOW_CONFIDENCE_THRESHOLD = 0.6

    def verify(
        self,
        claim: str,
        sources: list[tuple[str, str]],
    ) -> CrossCheckResult:
        """验证 claim 是否被多个独立来源支持"""
        # 简化策略：每个 source 视为"同意"
        # 实际生产需要 NLP 判断每个 source 是否支持 claim
        supports = sum(1 for _ in sources)
        contradicts = 0

        # 检测冲突（简单：sources 内容是否互斥）
        if supports >= 2:
            texts = [s[1].lower() for s in sources]
            # 简单检测：如果有 "down" 和 "up" 同时出现
            downs = sum("down" in t for t in texts)
            ups = sum("up" in t for t in texts)
            if downs > 0 and ups > 0:
                contradicts = min(downs, ups)

        has_conflict = contradicts > 0
        confirmed = supports >= self.MIN_SOURCES_FOR_CONFIRM and not has_conflict

        # 置信度 = 支持数 / 3（3 个源算满分）
        if has_conflict:
            confidence = 0.3  # 冲突时低置信度
        else:
            confidence = min(supports / 3.0, 1.0)

        return CrossCheckResult(
            confirmed=confirmed,
            confidence=confidence,
            has_conflict=has_conflict,
        )


# ============================================================
# Layer 4: Output Schema
# ============================================================
class OutputSchemaValidator:
    """第 4 层：输出 schema 校验"""

    REQUIRED_FIELDS = {"incident_id", "root_cause", "confidence", "evidence_refs"}

    def validate(self, output: dict[str, Any]) -> ValidationResult:
        if not output:
            return ValidationResult(is_valid=False, reason="Empty output")

        missing = self.REQUIRED_FIELDS - set(output.keys())
        if missing:
            return ValidationResult(
                is_valid=False,
                reason=f"Missing fields: {sorted(missing)}",
            )

        # confidence 必须是数字
        conf = output.get("confidence")
        if not isinstance(conf, (int, float)):
            return ValidationResult(is_valid=False, reason="confidence must be number")

        # evidence_refs 必须是非空列表
        refs = output.get("evidence_refs", [])
        if not isinstance(refs, list) or len(refs) == 0:
            return ValidationResult(
                is_valid=False,
                reason="evidence_refs must be non-empty list",
            )

        return ValidationResult(is_valid=True)


# ============================================================
# Pipeline: 4 层串联
# ============================================================
class HallucinationGuard:
    """4 层幻觉防护 Pipeline"""

    def __init__(self) -> None:
        self.input_validator = InputValidator()
        self.cross_checker = CrossChecker()
        self.output_validator = OutputSchemaValidator()

    # 跨层 LLM 结论 → (root_cause 字符串, evidence 字符串列表) 适配器
    @staticmethod
    def _extract_claim_and_sources(
        input_data: dict[str, Any],
        llm_output: Any,
    ) -> tuple[str, list[tuple[str, str]]]:
        """把 RCA 输入和 LLM 草稿归一成 (claim, sources) 给 CrossChecker.verify 用。

        claim 优先取 llm_output["root_cause"]，退到 llm_output 字符串；
        sources 从 input_data["evidence"] 拼出，每个 evidence 字符串视为一条独立来源。
        """
        if isinstance(llm_output, dict):
            claim = str(llm_output.get("root_cause") or "").strip()
            if not claim:
                claim = json.dumps(llm_output, ensure_ascii=False)[:512]
        elif isinstance(llm_output, str):
            claim = llm_output.strip()
        else:
            claim = str(llm_output)

        evidence = input_data.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        sources: list[tuple[str, str]] = []
        for idx, ev in enumerate(evidence):
            sources.append((f"src_{idx}", str(ev)))
        return claim, sources

    def evaluate(
        self,
        input_data: dict[str, Any],
        llm_output: Any,
    ) -> PipelineResult:
        # Layer 1: Input validation
        r1 = self.input_validator.validate(input_data)
        if not r1.is_valid:
            return PipelineResult(
                passed=False,
                failure_layer="input_validation",
                details={"reason": r1.reason},
            )

        # Layer 2: Evidence grounding (LLM 输出是字符串时)
        if isinstance(llm_output, str):
            checker = EvidenceGroundingChecker(input_data.get("evidence", []))
            r2 = checker.check(llm_output)
            if not r2.is_valid:
                return PipelineResult(
                    passed=False,
                    failure_layer="evidence_grounding",
                    details={"reason": r2.reason, "unsupported": r2.unsupported_claims},
                )

        # Layer 3: Cross-Check (多源交叉验证)
        claim, sources = self._extract_claim_and_sources(input_data, llm_output)
        r3 = self.cross_checker.verify(claim, sources)
        details: dict[str, Any] = {
            "cross_check": {
                "confirmed": r3.confirmed,
                "cross_confidence": r3.confidence,
                "has_conflict": r3.has_conflict,
                "sources_count": len(sources),
            }
        }
        # 多源冲突或来源不足 → 直接拦截
        if r3.has_conflict or not r3.confirmed:
            return PipelineResult(
                passed=False,
                failure_layer="cross_check",
                details={
                    **details,
                    "reason": (
                        "cross-check conflict detected"
                        if r3.has_conflict
                        else "cross-check insufficient sources"
                    ),
                },
            )

        # Layer 4: Output schema (LLM 输出是 dict 时)
        if isinstance(llm_output, dict):
            r4 = self.output_validator.validate(llm_output)
            if not r4.is_valid:
                return PipelineResult(
                    passed=False,
                    failure_layer="output_schema",
                    details={**details, "reason": r4.reason},
                )

        return PipelineResult(passed=True, details=details)