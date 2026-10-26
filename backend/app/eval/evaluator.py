"""
RCAEvaluator - Badcase 评估器

评估维度：
1. root_cause_match：RCA 输出的根因是否匹配 expected（容错匹配）
2. confidence_meet：置信度是否 ≥ expected_confidence_min
3. citation_present：是否含 evidence_refs（防幻觉的核心信号）
4. no_hallucination：未编造不在证据中的 metric/service

输出：
- EvalResult（单条）：4 个维度 + 总评
- EvalSummary（汇总）：pass_rate / 平均置信度 / 类别分析
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.eval.dataset import Badcase, BadcaseDataset, get_default_dataset
from app.core.llm_fallback import rca_root_cause


logger = logging.getLogger(__name__)


@dataclass
class RCAOutput:
    """RCA Agent 输出的统一格式"""
    root_cause: str
    confidence: float
    evidence_refs: list[str] = field(default_factory=list)
    method: str = "unknown"
    reasoning: str = ""


@dataclass
class EvalResult:
    """单条 badcase 评估结果"""
    badcase_id: str
    expected_root_cause: str
    actual_root_cause: str
    confidence: float
    expected_confidence_min: float

    pass_root_cause_match: bool
    pass_confidence_meet: bool
    pass_citation_present: bool
    pass_no_hallucination: bool
    pass_overall: bool  # 4 个都通过

    # v3 E6: 第 5 维度（不计入 pass_overall，避免回归现有数据）
    pass_semantic_match: bool = True
    semantic_match_method: str = "keyword"  # "keyword" | "llm_judge"
    semantic_match_score: float = 0.0

    notes: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    """整体评估汇总"""
    total: int
    passed: int
    pass_rate: float
    avg_confidence: float

    by_dimension: dict[str, dict[str, float]] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    failing_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 3),
            "avg_confidence": round(self.avg_confidence, 3),
            "by_dimension": self.by_dimension,
            "by_category": self.by_category,
            "failing_ids": self.failing_ids,
        }


class RCAEvaluator:
    """RCA Agent 评估器（可插拔 rca_func）"""

    def __init__(self, rca_func=None):
        """
        参数：
            rca_func: 可调用 (alert, evidence) -> RCAOutput 或 dict
                     默认用 llm_fallback.rca_root_cause（规则引擎）
        """
        self.rca_func = rca_func or self._default_rca_func

    @staticmethod
    def _default_rca_func(alert: dict, evidence: dict) -> RCAOutput:
        result = rca_root_cause(alert, evidence)
        return RCAOutput(
            root_cause=result["root_cause"],
            confidence=result["confidence"],
            evidence_refs=result.get("evidence_refs", []),
            method=result.get("method", "rules_engine"),
            reasoning=f"confidence={result['confidence']}, method={result.get('method')}",
        )

    def evaluate_one(self, badcase: Badcase) -> EvalResult:
        """评估单条 badcase

        v3 E6: 加第 5 维度 semantic_match（LLM judge，失败 fallback 到 keyword_match）
        v3 E7: 升级 _contains_hallucination（提取实体对照 evidence）
        """
        try:
            output = self.rca_func(badcase.alert, badcase.evidence)
        except Exception as e:
            logger.exception("RCA 函数执行失败: badcase=%s", badcase.id)
            output = RCAOutput(
                root_cause="<evaluation_error>",
                confidence=0.0,
                reasoning=f"error: {e}",
            )

        # 容错匹配：actual 包含 expected 关键词（部分匹配）
        match_score = self._match_score(output.root_cause, badcase.expected_root_cause)

        # v3 E6: 加 LLM judge 语义匹配（如可用；不可用退化为 keyword match）
        semantic_pass, semantic_score, semantic_method = self._semantic_match(
            output.root_cause, badcase.expected_root_cause
        )

        pass_match = match_score >= 0.5  # 至少 50% 关键词重合
        pass_conf = output.confidence >= badcase.expected_confidence_min
        pass_citation = len(output.evidence_refs) > 0
        pass_no_hallu = not self._contains_hallucination(output, badcase)

        notes = []
        if not pass_match:
            notes.append(f"根因失配: expected={badcase.expected_root_cause}, actual={output.root_cause}")
        if not pass_conf:
            notes.append(f"置信度不足: expected≥{badcase.expected_confidence_min}, actual={output.confidence}")
        if not pass_citation:
            notes.append("缺少 evidence_refs（幻觉风险）")
        if not pass_no_hallu:
            notes.append("检测到幻觉痕迹")
        if semantic_method == "llm_judge" and not semantic_pass:
            notes.append(f"LLM 语义判定不通过 (score={semantic_score:.2f})")

        return EvalResult(
            badcase_id=badcase.id,
            expected_root_cause=badcase.expected_root_cause,
            actual_root_cause=output.root_cause,
            confidence=output.confidence,
            expected_confidence_min=badcase.expected_confidence_min,
            pass_root_cause_match=pass_match,
            pass_confidence_meet=pass_conf,
            pass_citation_present=pass_citation,
            pass_no_hallucination=pass_no_hallu,
            # v3 E6: 第 5 维度（默认 pass，不计入 pass_overall 以避免回归）
            pass_semantic_match=semantic_pass,
            semantic_match_method=semantic_method,
            semantic_match_score=semantic_score,
            pass_overall=pass_match and pass_conf and pass_citation and pass_no_hallu,
            notes=notes,
        )

    def evaluate_all(self, dataset: BadcaseDataset) -> list[EvalResult]:
        """评估整个数据集"""
        return [self.evaluate_one(bc) for bc in dataset.badcases]

    def summarize(self, results: list[EvalResult], dataset: BadcaseDataset) -> EvalSummary:
        """汇总评估结果"""
        total = len(results)
        passed = sum(1 for r in results if r.pass_overall)
        avg_conf = sum(r.confidence for r in results) / total if total > 0 else 0.0

        # 按维度
        dim_stats = {}
        for dim in [
            "pass_root_cause_match",
            "pass_confidence_meet",
            "pass_citation_present",
            "pass_no_hallucination",
            "pass_semantic_match",  # v3 E6
        ]:
            p = sum(1 for r in results if getattr(r, dim))
            dim_stats[dim] = {"passed": p, "total": total, "rate": round(p / total, 3) if total else 0}

        # 按类别
        cat_stats = {}
        bc_by_id = {bc.id: bc for bc in dataset.badcases}
        for cat in {bc.category for bc in dataset.badcases}:
            cat_results = [r for r in results if bc_by_id[r.badcase_id].category == cat]
            if not cat_results:
                continue
            p = sum(1 for r in cat_results if r.pass_overall)
            cat_stats[cat] = {
                "passed": p,
                "total": len(cat_results),
                "rate": round(p / len(cat_results), 3),
            }

        return EvalSummary(
            total=total,
            passed=passed,
            pass_rate=passed / total if total else 0,
            avg_confidence=avg_conf,
            by_dimension=dim_stats,
            by_category=cat_stats,
            failing_ids=[r.badcase_id for r in results if not r.pass_overall],
        )

    # ---------- 辅助方法 ----------
    @staticmethod
    def _match_score(actual: str, expected: str) -> float:
        """容错匹配：关键词重合度"""
        if not actual or not expected:
            return 0.0
        actual_lower = actual.lower()
        expected_lower = expected.lower()
        if actual_lower == expected_lower:
            return 1.0
        if actual_lower in expected_lower or expected_lower in actual_lower:
            return 0.8
        # 提取关键词（去下划线、连字符）
        def keywords(s: str) -> set[str]:
            return set(re.split(r"[_\-\s]+", s.lower())) - {""}
        ak = keywords(actual_lower)
        ek = keywords(expected_lower)
        if not ek:
            return 0.0
        common = ak & ek
        return len(common) / len(ek)

    @staticmethod
    def _contains_hallucination(output: RCAOutput, badcase: Badcase) -> bool:
        """检测幻觉：output 是否编造了 evidence/alert 里没有的实体（v3 E7 升级）

        升级版：
        - 提取 output.root_cause + output.reasoning 中所有 entity（service / metric / host / KPI 名）
        - 收集 alert 和 evidence 中所有真实出现的 entity
        - 任何 output 提及但 evidence/alert 没出现的实体都算幻觉
        """
        # 简化检查：output 提及的 service 是否在 alert 中
        if not output.root_cause or output.root_cause in (
            "Unknown pattern (insufficient evidence)",
            "<evaluation_error>",
        ):
            return True

        # v3 E7: 提取 evidence/alert 中所有已知实体（key + value）
        known_entities: set[str] = set()
        for src in [badcase.alert, badcase.evidence]:
            for k, v in src.items():
                known_entities.add(str(k).lower())
                # values 里也是 evidence（KPI 数值列表也算）
                if isinstance(v, list):
                    for item in v[:50]:  # 限大小
                        known_entities.add(str(item).lower())
                elif isinstance(v, str):
                    known_entities.add(str(v).lower())
                elif isinstance(v, (int, float)):
                    known_entities.add(str(v))

        # 提取 output 中所有可能的实体（service / metric / host）
        output_text = (output.root_cause + " " + output.reasoning).lower()
        entities_in_output: set[str] = set()

        # 1. xxx-service / xxx_svc / xxx-server 形式
        for word in re.findall(r"[a-z][a-z0-9\-]+-(?:service|svc|server|host|pod|container)", output_text):
            entities_in_output.add(word)
        # 2. metric_name 形式 (cpu_usage / memory_used 等)
        for word in re.findall(r"[a-z][a-z0-9_]+_(?:usage|usage_pct|count|rate|errors|latency|qps|tps)", output_text):
            entities_in_output.add(word)
        # 3. host / ip 形式
        for word in re.findall(r"host[:\s]+([a-z0-9\-]+)", output_text):
            entities_in_output.add(word)
        for word in re.findall(r"\b(?:10|192|172)\.\d+\.\d+\.\d+\b", output_text):
            entities_in_output.add(word)

        # 任何在 output 但不在 evidence 的实体都算幻觉
        unknown_entities = entities_in_output - known_entities
        if unknown_entities:
            # 但允许 alert 中明确出现的 service
            alert_service = badcase.alert.get("service", "").lower()
            unknown_entities = {e for e in unknown_entities if alert_service not in e}

        if unknown_entities:
            logger.debug(f"幻觉检测发现未引用实体 badcase={badcase.id}, unknown={unknown_entities}")
            return True

        return False

    @staticmethod
    def _semantic_match(actual: str, expected: str) -> tuple[bool, float, str]:
        """v3 E6: 语义匹配（LLM judge + fallback）

        Returns:
            (pass, score, method)
            - pass: 是否语义一致
            - score: 0.0-1.0 置信度
            - method: "llm_judge" | "keyword" | "skipped"
        """
        if not actual or not expected:
            return True, 0.0, "skipped"
        if actual == expected:
            return True, 1.0, "keyword"

        # 快速通过：actual 包含 expected（或反向）
        if expected.lower() in actual.lower() or actual.lower() in expected.lower():
            return True, 0.8, "keyword"

        # 尝试 LLM judge（异步场景跳过，避免阻塞）
        try:
            import os
            if os.environ.get("AIOPS_FORCE_FALLBACK") == "1":
                return False, 0.0, "keyword"

            from app.core.llm_client import get_llm_client, LLMUnavailable
            client = get_llm_client()
            if not client.is_available():
                return False, 0.0, "keyword"

            system = (
                "你是 AIOps 根因匹配专家。判定 actual_root_cause 是否在语义上等价于 expected_root_cause。\n"
                '只返回 JSON: {"match": true|false, "score": 0.X, "reason": "<20 字理由>"}'
            )
            user = (
                f"expected_root_cause: {expected}\n"
                f"actual_root_cause: {actual}\n\n"
                f"两者是否指向同一根因？"
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            result = client.chat_sync(messages, temperature=0.0, max_tokens=100)
            content = result.content.strip()

            # 解析 JSON
            import json
            match_data = json.loads(content)
            return (
                bool(match_data.get("match", False)),
                float(match_data.get("score", 0.0)),
                "llm_judge",
            )
        except (LLMUnavailable, Exception) as e:
            logger.debug(f"LLM judge 不可用，fallback 到 keyword match: {e}")
            return False, 0.0, "keyword"


def run_evaluation(
    dataset: BadcaseDataset | None = None,
    rca_func=None,
    verbose: bool = False,
    record_history: bool = True,
    history_note: str | None = None,
    analyzer_name: str = "default",
) -> tuple[list[EvalResult], EvalSummary]:
    """
    跑评估的便捷入口。

    v3 E5：默认自动记录到 SQLite 历史库（record_history=False 可关闭）

    返回：(results, summary)
    """
    ds = dataset or get_default_dataset()
    evaluator = RCAEvaluator(rca_func=rca_func)
    results = evaluator.evaluate_all(ds)
    summary = evaluator.summarize(results, ds)

    if verbose:
        logger.info(f"=== 评估汇总 ===")
        logger.info(f"Total: {summary.total}, Passed: {summary.passed}, Pass Rate: {summary.pass_rate:.1%}")
        logger.info(f"Avg Confidence: {summary.avg_confidence:.3f}")
        for dim, stats in summary.by_dimension.items():
            logger.info(f"  {dim}: {stats['rate']:.1%}")
        for cat, stats in summary.by_category.items():
            logger.info(f"  category={cat}: {stats['rate']:.1%} ({stats['passed']}/{stats['total']})")
        if summary.failing_ids:
            logger.warning(f"Failing IDs: {summary.failing_ids}")

    # v3 E5：自动记录到历史库（失败不抛错，best-effort）
    if record_history:
        try:
            from app.eval.history import record_eval_run
            record_eval_run(
                split=ds.eval_split or "all",
                analyzer=analyzer_name,
                summary=summary,
                config={"rca_func": rca_func.__name__ if rca_func else None},
                note=history_note,
            )
        except Exception as e:
            logger.debug(f"record_eval_run 失败（不影响主流程）: {e}")

    return results, summary
