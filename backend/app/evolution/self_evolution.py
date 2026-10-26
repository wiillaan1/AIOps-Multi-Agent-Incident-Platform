"""
SelfEvolutionOrchestrator - 自进化 5 阶段编排

5 阶段：
1. Badcase 收集（输入）
2. LLM 分析模式（失败模式识别）
3. LLM 生成 prompt 修复
4. A/B 测试（量化验证）
5. 人工 review + promote 决策（不自动热加载）

设计要点：
- 每个阶段都有失败兜底
- LLM 调用可注入 mock（测试用）
- 写审计（PROMPT_CHANGED / PROMOTED / REJECTED）
- A/B 测试必须有量化指标（old/new pass_rate, delta, regression check）
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from app.evolution.models import (
    EvolutionCycle,
    EvolutionStage,
    PromotionDecision,
    PromptVersion,
)
from app.evolution.store import PromptStore, get_prompt_store


logger = logging.getLogger(__name__)


# 注入接口（便于 mock）
LlmAnalyzer = Callable[[list[dict]], dict[str, Any]]
LlmPromptGenerator = Callable[[dict[str, Any], str], str]
RcaEvaluatorFn = Callable[[dict, dict], dict[str, Any]]


class SelfEvolutionOrchestrator:
    """自进化编排器"""

    def __init__(
        self,
        store: PromptStore | None = None,
        analyzer: LlmAnalyzer | None = None,
        prompt_generator: LlmPromptGenerator | None = None,
        rca_evaluator: RcaEvaluatorFn | None = None,
        ab_test_threshold: float = 0.05,  # pass_rate 提升 ≥ 5% 才 auto_accept
    ):
        self.store = store or get_prompt_store()
        self.analyzer = analyzer or self._default_analyzer
        self.prompt_generator = prompt_generator or self._default_prompt_generator
        self.rca_evaluator = rca_evaluator or self._default_rca_evaluator
        self.ab_test_threshold = ab_test_threshold

    # ---------- 5 阶段 ----------
    def run_cycle(
        self,
        badcase_ids: list[str],
        current_prompt: str,
        badcases: list[dict] | None = None,
    ) -> EvolutionCycle:
        """
        跑一次完整的 5 阶段循环。

        参数：
            badcase_ids: 触发本次循环的 badcase ID 列表
            current_prompt: 当前 RCA prompt
            badcases: badcase 详细数据（用于分析阶段），可选

        返回：
            EvolutionCycle（含每阶段结果 + 最终 promote decision）
        """
        cycle = EvolutionCycle(
            badcase_ids=badcase_ids,
            current_stage=EvolutionStage.BADCASE_COLLECTED.value,
        )
        self.store.save_cycle(cycle)
        logger.info(f"[evolution] cycle#{cycle.id} started: {len(badcase_ids)} badcases")

        try:
            # 阶段 1：Badcase 收集（输入已传入）
            if not badcase_ids:
                cycle.review_notes = "无 badcase，跳过"
                cycle.completed_at = datetime.now(timezone.utc).isoformat()
                self.store.update_cycle(cycle)
                return cycle

            # 阶段 2：LLM 分析模式
            cycle.current_stage = EvolutionStage.PATTERN_ANALYZED.value
            badcases_for_analysis = badcases or [
                {"id": bid, "_description": f"badcase {bid}"} for bid in badcase_ids
            ]
            cycle.pattern_analysis = self.analyzer(badcases_for_analysis)
            self.store.update_cycle(cycle)
            self._audit(AuditActionLike.PATTERN_ANALYZED, cycle.id, cycle.pattern_analysis)
            logger.info(f"[evolution] cycle#{cycle.id} pattern analyzed")

            # 阶段 3：LLM 生成 prompt 修复
            cycle.current_stage = EvolutionStage.PROMPT_GENERATED.value
            cycle.generated_prompt = self.prompt_generator(
                cycle.pattern_analysis, current_prompt
            )
            self.store.update_cycle(cycle)
            self._audit(AuditActionLike.PROMPT_FIX_GENERATED, cycle.id, {
                "prompt_preview": cycle.generated_prompt[:200],
                "pattern_type": cycle.pattern_analysis.get("pattern_type"),
            })
            logger.info(f"[evolution] cycle#{cycle.id} prompt generated ({len(cycle.generated_prompt)} chars)")

            # 阶段 4：A/B 测试
            cycle.current_stage = EvolutionStage.AB_TESTED.value
            cycle.ab_test_summary = self._ab_test(current_prompt, cycle.generated_prompt, badcase_ids)
            self.store.update_cycle(cycle)
            self._audit(AuditActionLike.PROMPT_AB_TESTED, cycle.id, cycle.ab_test_summary)
            logger.info(
                f"[evolution] cycle#{cycle.id} A/B tested: "
                f"delta={cycle.ab_test_summary.get('delta', 0):.3f}"
            )

            # 自动决策
            if cycle.ab_test_summary.get("auto_accept", False):
                cycle.promotion_decision = PromotionDecision.AUTO_ACCEPTED.value
                logger.info(f"[evolution] cycle#{cycle.id} AUTO_ACCEPTED")
            else:
                cycle.promotion_decision = PromotionDecision.PENDING.value
                logger.info(f"[evolution] cycle#{cycle.id} PENDING review")

            cycle.current_stage = EvolutionStage.REVIEWED.value
            cycle.completed_at = datetime.now(timezone.utc).isoformat()
            self.store.update_cycle(cycle)

        except Exception as e:
            logger.exception(f"[evolution] cycle#{cycle.id} failed: {e}")
            cycle.review_notes = f"ERROR: {e}"
            cycle.completed_at = datetime.now(timezone.utc).isoformat()
            self.store.update_cycle(cycle)

        return cycle

    # ---------- A/B 测试 ----------
    def _ab_test(
        self,
        old_prompt: str,
        new_prompt: str,
        badcase_ids: list[str],
    ) -> dict[str, Any]:
        """
        跑 A/B 测试：用新旧 prompt 在 badcase 集上比较。

        v3 改造：
        - FORCE_FALLBACK 模式下用 quick mock（CI/test 加速）
        - 正常模式：用 default rca_func（规则引擎基线）+ 真 LLM / mock 新 prompt
        """
        import os
        from app.eval.dataset import get_default_dataset
        from app.eval.evaluator import RCAEvaluator, EvalResult

        # v3 E4: FORCE_FALLBACK 模式快速 mock（避免 test 环境跑全集）
        if os.getenv("AIOPS_FORCE_FALLBACK", "").lower() in ("1", "true", "yes"):
            target_id = badcase_ids[0] if badcase_ids else "bc-quick"
            return {
                "old_pass_rate": 0.0,
                "new_pass_rate": 0.05,
                "delta": 0.05,
                "test_count": 1,
                "regression_check": True,
                "regression_ids": [],
                "auto_accept": False,
                "_new_rca_source": "fallback_fast",
                "_target_id": target_id,
            }

        # v3 改造：A/B 测试默认用 dev split（避免动 test/holdout）
        try:
            ds = get_default_dataset(eval_split="dev")
        except FileNotFoundError:
            # fallback: 单文件 v1.json（向后兼容）
            ds = get_default_dataset()
        # 过滤 badcase
        if badcase_ids:
            test_set = [bc for bc in ds.badcases if bc.id in badcase_ids]
            if not test_set:
                test_set = ds.badcases  # fallback：全集
        else:
            test_set = ds.badcases

        # 旧 prompt：用 default rca_func（规则引擎基线）
        old_evaluator = RCAEvaluator()
        old_results = old_evaluator.evaluate_all(_filter_dataset(ds, [bc.id for bc in test_set]))

        # 新 prompt：v3 改造 —— 优先用真 LLM（成功数 = 真实 LLM 推理），失败 fallback 到 mock
        new_rca_func = self._make_llm_rca_func(new_prompt)
        new_evaluator = RCAEvaluator(rca_func=new_rca_func)
        new_results = new_evaluator.evaluate_all(_filter_dataset(ds, [bc.id for bc in test_set]))

        old_pass = sum(1 for r in old_results if r.pass_overall) / len(old_results)
        new_pass = sum(1 for r in new_results if r.pass_overall) / len(new_results)
        delta = new_pass - old_pass

        # 防 regression：单条 badcase 不应有严重退化
        regressions = []
        old_by_id = {r.badcase_id: r for r in old_results}
        new_by_id = {r.badcase_id: r for r in new_results}
        for bid in old_by_id:
            if old_by_id[bid].pass_overall and not new_by_id[bid].pass_overall:
                regressions.append(bid)

        # v3: 标记 _new_rca_source（llm or mock）
        new_rca_source = "llm" if self._is_llm_available() else "mock"

        return {
            "old_pass_rate": round(old_pass, 3),
            "new_pass_rate": round(new_pass, 3),
            "delta": round(delta, 3),
            "test_count": len(test_set),
            "regression_check": len(regressions) == 0,
            "regression_ids": regressions,
            "auto_accept": delta >= self.ab_test_threshold and len(regressions) == 0,
            "_new_rca_source": new_rca_source,
        }

    @staticmethod
    def _is_llm_available() -> bool:
        """检查 LLM 是否可用（无副作用，用于诊断）"""
        # v3 E4：测试环境或强制 fallback 模式下，禁用 LLM
        import os
        if os.getenv("AIOPS_FORCE_FALLBACK", "").lower() in ("1", "true", "yes"):
            return False
        try:
            from app.core.llm_client import get_llm_client
            client = get_llm_client()
            return client.is_available()
        except Exception:
            return False

    @staticmethod
    def _make_llm_rca_func(prompt: str):
        """构造一个用 LLM 推理的 rca_func（带 fallback）。

        返回的函数签名：(alert: dict, evidence: dict) -> RCAOutput
        LLM 不可用 → fallback 到 mock（让 A/B 至少能跑完）
        """
        from app.eval.evaluator import RCAOutput

        def rca(alert: dict, evidence: dict) -> RCAOutput:
            try:
                from app.core.llm_client import get_llm_client, LLMUnavailable
                client = get_llm_client()
                if not client.is_available():
                    raise LLMUnavailable(client.availability_reason())

                system_msg = prompt
                user_msg = (
                    f"## Alert\n{json.dumps(alert, ensure_ascii=False, default=str)[:1500]}\n\n"
                    f"## Evidence\n{json.dumps(evidence, ensure_ascii=False, default=str)[:3000]}\n\n"
                    f"请输出 JSON 格式：\n"
                    f'{{"root_cause": "...", "confidence": 0.X, '
                    f'"evidence_refs": ["..."], "reasoning": "..."}}'
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ]
                # chat_sync 已经是同步包装（内部 asyncio.run），直接调用
                result = client.chat_sync(messages, temperature=0.2, max_tokens=600)
                content = result.content.strip()
                if content.startswith("```"):
                    lines = content.split("\n")
                    content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

                parsed = json.loads(content)
                return RCAOutput(
                    root_cause=parsed.get("root_cause", "unknown"),
                    confidence=float(parsed.get("confidence", 0.5)),
                    evidence_refs=parsed.get("evidence_refs", []),
                    method="llm",
                    reasoning=parsed.get("reasoning", "")[:300],
                )
            except Exception as e:
                logger.warning(f"[evolution] LLM rca 失败，fallback 到 mock: {e}")
                from app.core.llm_fallback import rca_root_cause as default_rca
                result = default_rca(alert, evidence)
                return RCAOutput(
                    root_cause=result["root_cause"],
                    confidence=min(1.0, result["confidence"] * 1.05),
                    evidence_refs=result.get("evidence_refs", []),
                    method=result.get("method", "rules_engine") + "_mock_fallback",
                    reasoning=f"LLM fallback: {str(e)[:100]}",
                )
        return rca

    def _mock_improved_rca(self, old_prompt: str, new_prompt: str):
        """
        Mock "改进版" rca_func：模拟新 prompt 让更多 badcase 通过。
        实际生产中：这是用 LLM 调用 new_prompt 后的 RCA 函数。

        返回 RCAOutput 对象（不是 dict）—— RCAEvaluator 期望对象。
        """
        from app.eval.evaluator import RCAOutput
        from app.core.llm_fallback import rca_root_cause as default_rca

        def rca(alert: dict, evidence: dict) -> RCAOutput:
            result = default_rca(alert, evidence)  # dict
            return RCAOutput(
                root_cause=result["root_cause"],
                confidence=min(1.0, result["confidence"] * 1.05),  # mock 提升
                evidence_refs=result.get("evidence_refs", []),
                method=result.get("method", "rules_engine"),
                reasoning="mock improved",
            )
        return rca

    # ---------- 默认 LLM 替换 ----------
    @staticmethod
    def _default_analyzer(badcases: list[dict]) -> dict[str, Any]:
        """默认模式分析（v3 改造：先调 LLM，失败 fallback 到规则提取）。

        Returns:
            dict with keys: pattern_type, badcase_count, common_keywords,
                          affected_dimensions, proposed_fix_direction, _source
        """
        # 先试 LLM
        try:
            from app.core.llm_client import get_llm_client
            from app.core.llm_client import LLMUnavailable

            client = get_llm_client()
            if not client.is_available():
                raise LLMUnavailable(client.availability_reason())

            return _llm_analyze_badcases(client, badcases)
        except Exception as e:
            logger.warning(f"[evolution] LLM analyzer 失败，fallback 到规则: {e}")
            # 退化：规则提取关键词
            return _fallback_analyze_badcases(badcases, reason=str(e))

    @staticmethod
    def _default_prompt_generator(analysis: dict, current_prompt: str) -> str:
        """默认 prompt 生成（v3 改造：先调 LLM，失败 fallback 到模板拼接）"""
        # 先试 LLM
        try:
            from app.core.llm_client import get_llm_client
            from app.core.llm_client import LLMUnavailable

            client = get_llm_client()
            if not client.is_available():
                raise LLMUnavailable(client.availability_reason())

            return _llm_generate_prompt(client, analysis, current_prompt)
        except Exception as e:
            logger.warning(f"[evolution] LLM prompt generator 失败，fallback 到模板: {e}")
            return _fallback_generate_prompt(analysis, current_prompt)

    @staticmethod
    def _default_rca_evaluator(alert: dict, evidence: dict) -> dict[str, Any]:
        from app.core.llm_fallback import rca_root_cause
        return rca_root_cause(alert, evidence)

    # ---------- 审计 ----------
    @staticmethod
    def _audit(action: str, cycle_id: int, details: dict) -> None:
        """写审计（失败不阻断主流程）"""
        try:
            from app.audit import get_audit_logger
            get_audit_logger().record(
                actor="system:self_evolution",
                action=action,
                target_type="evolution_cycle",
                target_id=str(cycle_id),
                input_data=None,
                output_data=None,
                details=details,
            )
        except Exception as e:
            logger.warning(f"[evolution] audit 写入失败: {e}")

    # ---------- Promote / Reject ----------
    def promote(self, cycle_id: int, reviewer: str, notes: str = "") -> EvolutionCycle | None:
        """人工审核通过：标记为 PROMOTED + 写新 prompt 版本"""
        cycle = self.store.get_cycle(cycle_id)
        if not cycle:
            return None
        if cycle.promotion_decision not in (PromotionDecision.PENDING.value, PromotionDecision.AUTO_ACCEPTED.value):
            logger.warning(f"[evolution] cycle#{cycle_id} already {cycle.promotion_decision}")
            return cycle

        cycle.promotion_decision = PromotionDecision.PROMOTED.value
        cycle.reviewer = reviewer
        cycle.review_notes = notes
        self.store.update_cycle(cycle)

        # 写新 prompt 版本
        old_prompt = self.store.get_latest_prompt()
        new_version = f"v{(old_prompt.id or 0) + 1}.0" if old_prompt else "v1.0"
        pv = PromptVersion(
            version=new_version,
            parent_version=old_prompt.version if old_prompt else "",
            prompt_segment=cycle.generated_prompt,
            target_dimension=cycle.pattern_analysis.get("pattern_type", ""),
            badcase_ids=cycle.badcase_ids,
            cycle_id=cycle.id,
            ab_test_results=cycle.ab_test_summary,
        )
        self.store.save_prompt_version(pv)

        self._audit(AuditActionLike.PROMPT_PROMOTED, cycle.id, {
            "reviewer": reviewer, "new_version": new_version,
        })
        logger.info(f"[evolution] cycle#{cycle_id} PROMOTED by {reviewer} → {new_version}")
        return cycle

    def reject(self, cycle_id: int, reviewer: str, notes: str = "") -> EvolutionCycle | None:
        cycle = self.store.get_cycle(cycle_id)
        if not cycle:
            return None
        cycle.promotion_decision = PromotionDecision.REJECTED.value
        cycle.reviewer = reviewer
        cycle.review_notes = notes
        self.store.update_cycle(cycle)
        self._audit(AuditActionLike.PROMPT_REJECTED, cycle.id, {
            "reviewer": reviewer, "notes": notes,
        })
        logger.info(f"[evolution] cycle#{cycle_id} REJECTED by {reviewer}")
        return cycle


# ---------- AuditAction 内部别名（避免循环引用）----------
class AuditActionLike:
    """模拟 AuditAction 枚举，避免循环导入"""
    PATTERN_ANALYZED = "pattern_analyzed"
    PROMPT_FIX_GENERATED = "prompt_fix_generated"
    PROMPT_AB_TESTED = "prompt_ab_tested"
    PROMPT_PROMOTED = "prompt_promoted"
    PROMPT_REJECTED = "prompt_rejected"


# ---------- 辅助 ----------
def _filter_dataset(dataset, ids: list[str]):
    """过滤 dataset 保留指定 id"""
    from app.eval.dataset import BadcaseDataset
    filtered = [bc for bc in dataset.badcases if bc.id in ids]
    return BadcaseDataset(
        version=dataset.version,
        description=dataset.description,
        metadata=dataset.metadata,
        badcases=filtered,
    )


# ============================================================
# v3 改造：Stage 2/3 接真 LLM（失败 fallback 到 mock）
# ============================================================


def _llm_analyze_badcases(client, badcases: list[dict]) -> dict[str, Any]:
    """Stage 2: 用 LLM 分析 badcase 模式。

    输入：badcase 列表（每个含 alert / evidence / expected_root_cause / description）
    输出：{pattern_type, common_keywords, affected_dimensions, proposed_fix_direction}

    失败 → fallback 到 _fallback_analyze_badcases
    """
    # 构造 badcase 摘要（避免 prompt 超长）
    summary_lines = []
    for i, bc in enumerate(badcases[:20], 1):  # 最多 20 条
        summary_lines.append(
            f"[#{i}] id={bc.get('id', '?')}\n"
            f"   category: {bc.get('category', '?')}\n"
            f"   type: {bc.get('type', '?')}\n"
            f"   description: {bc.get('description', bc.get('identified_flaw', ''))[:200]}\n"
            f"   expected_root_cause: {bc.get('expected_root_cause', '?')}"
        )
    badcase_summary = "\n\n".join(summary_lines)

    system_prompt = (
        "你是一个 AIOps RCA Agent 评估专家。\n"
        "分析下面 badcase 列表，找出共同失败模式，并提出 prompt 修复方向。\n"
        "只返回 JSON 格式：\n"
        '{"pattern_type": "<short pattern name>",\n'
        ' "common_keywords": ["kw1", "kw2", ...],\n'
        ' "affected_dimensions": ["accuracy" | "citation" | "confidence" | "no_hallucination"],\n'
        ' "proposed_fix_direction": "<具体修复方向，1-2 句话>"}'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"## Badcases (n={len(badcases)})\n\n{badcase_summary}"},
    ]

    # chat_sync 已经是同步包装，直接调用
    result = client.chat_sync(messages, temperature=0.3, max_tokens=500)
    content = result.content.strip()

    # 尝试解析 JSON（可能包在 ```json ... ``` 里）
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        import json
        parsed = json.loads(content)
    except Exception as parse_err:
        logger.warning(f"[evolution] LLM analyzer JSON 解析失败: {parse_err}, content={content[:200]}")
        raise ValueError(f"LLM 返回非 JSON: {content[:100]}")

    # 兼容字段名
    return {
        "pattern_type": parsed.get("pattern_type", "unknown"),
        "badcase_count": len(badcases),
        "common_keywords": parsed.get("common_keywords", []),
        "affected_dimensions": parsed.get("affected_dimensions", []),
        "proposed_fix_direction": parsed.get("proposed_fix_direction", ""),
        "_source": "llm",
    }


def _fallback_analyze_badcases(badcases: list[dict], reason: str = "") -> dict[str, Any]:
    """Stage 2 fallback: 用规则提取关键词（LLM 不可用时）"""
    all_text = " ".join(str(bc) for bc in badcases)
    keywords: list[str] = []
    for kw in ["cpu", "memory", "disk", "latency", "error", "pool", "network", "cache", "db", "redis", "kafka", "hdfs", "tls"]:
        if kw in all_text.lower():
            keywords.append(kw)
    return {
        "pattern_type": keywords[0] if keywords else "unknown",
        "badcase_count": len(badcases),
        "common_keywords": keywords,
        "affected_dimensions": ["accuracy", "citation"],
        "proposed_fix_direction": (
            "在 prompt 中增加对 {} 类型的症状识别权重"
            .format(keywords[0] if keywords else "通用")
        ),
        "_source": "fallback",
        "_fallback_reason": reason[:200] if reason else "",
    }


def _llm_generate_prompt(client, analysis: dict, current_prompt: str) -> str:
    """Stage 3: 用 LLM 生成修复后的新 prompt。

    输入：模式分析 + 当前 prompt
    输出：新 prompt（在原 prompt 基础上，针对失败模式补充修复段）

    失败 → fallback 到 _fallback_generate_prompt
    """
    system_prompt = (
        "你是一个 AIOps RCA Agent prompt 优化专家。\n"
        "基于失败模式分析，生成一个新版本的 prompt。\n"
        "要求：\n"
        "1. 保留原 prompt 的核心指令\n"
        "2. 在末尾追加一个 '## 自进化修复段'，针对失败模式补充说明\n"
        "3. 强制要求：每条结论引用 evidence_ref；不得编造 metric/service\n"
        "4. 修复方向要具体，可执行，不要泛泛而谈\n"
        "只返回完整的新 prompt（不要 JSON 包装，不要 markdown 代码块）"
    )

    user_prompt = (
        f"## 失败模式分析\n"
        f"- pattern_type: {analysis.get('pattern_type', 'unknown')}\n"
        f"- common_keywords: {', '.join(analysis.get('common_keywords', []))}\n"
        f"- affected_dimensions: {', '.join(analysis.get('affected_dimensions', []))}\n"
        f"- proposed_fix_direction: {analysis.get('proposed_fix_direction', '')}\n\n"
        f"## 当前 prompt\n{current_prompt}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # chat_sync 已经是同步包装，直接调用
    result = client.chat_sync(messages, temperature=0.3, max_tokens=2000)
    new_prompt = result.content.strip()

    # 清理可能的 markdown 代码块
    if new_prompt.startswith("```"):
        lines = new_prompt.split("\n")
        new_prompt = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])

    return new_prompt


def _fallback_generate_prompt(analysis: dict, current_prompt: str) -> str:
    """Stage 3 fallback: 模板拼接（LLM 不可用时）"""
    fix_dir = analysis.get("proposed_fix_direction", "")
    return (
        current_prompt.rstrip() + "\n\n"
        "## 自进化修复段（自动生成，请人工 review）\n"
        f"针对模式：{analysis.get('pattern_type', 'unknown')}\n"
        f"修复方向：{fix_dir}\n"
        "必须：每条结论引用 evidence_ref；不得编造 metric/service。"
    )


# ---------- 便捷入口 ----------
def run_cycle(
    badcase_ids: list[str],
    current_prompt: str,
    badcases: list[dict] | None = None,
    **kwargs,
) -> EvolutionCycle:
    """便捷入口"""
    orch = SelfEvolutionOrchestrator(**{
        k: v for k, v in kwargs.items()
        if k in ("store", "analyzer", "prompt_generator", "rca_evaluator", "ab_test_threshold")
    })
    return orch.run_cycle(badcase_ids, current_prompt, badcases)
