"""
AIOps Agent Platform - Runbook Service

基于 RCA 事件自动生成可发布的 Runbook 草案。

设计目标：
- 复用 RCA Agent 输出的全量证据（贝叶斯 / RAG / 日志 / 历史记忆）
- 输出结构化 dict + Markdown 文本，前端可直接渲染或保存为 .md 文件
- 日志证据（Loki 样本）作为 Runbook 的"关键证据片段"独立成段，便于复盘引用
- 不依赖外部 LLM，纯模板生成，确保可用性
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.models.events import AlertEvent, RCAEvent
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RunbookService:
    """基于 RCAEvent 自动生成 Runbook 草案的服务。"""

    # 根因 -> 默认回滚建议的兜底映射
    DEFAULT_ROLLBACK: dict[str, list[str]] = {
        "recent_deployment": ["kubectl rollout undo deployment/{service}"],
        "configuration_change": ["git revert config/{service}.yaml", "kubectl rollout restart deployment/{service}"],
        "resource_exhaustion": ["kubectl scale deployment/{service} --replicas={original_replicas}"],
        "dependency_failure": ["kubectl apply -f circuit-breaker/{service}.yaml"],
        "network_issue": ["检查 NetworkPolicy / iptables 规则"],
        "database_issue": ["DBA 介入，kill 长事务；必要时回滚 DDL"],
        "code_bug": ["回滚到上一个稳定版本（参照 recommended_actions）"],
        "traffic_spike": ["kubectl scale deployment/{service} --replicas={scaled_replicas}"],
        "hardware_failure": ["kubectl drain node/{node}; 等待节点替换"],
        "third_party_issue": ["启用 fallback / 关闭对第三方依赖的请求"],
        "unknown": ["保留现场，等待人工排查"],
    }

    def generate(
        self,
        rca_event: RCAEvent,
        alert: AlertEvent | None = None,
    ) -> dict[str, Any]:
        """
        生成 Runbook 草案。

        Args:
            rca_event: RCA Agent 的输出事件
            alert: 原始告警（用于补充 service / metric / value 等上下文）

        Returns:
            dict: 包含 markdown 文本与结构化字段
        """
        evidence = rca_event.evidence or {}
        log_evidence = evidence.get("log_evidence") or {}
        alert_service = alert.service if alert else "unknown"

        sections: list[dict[str, Any]] = []

        # 1. 概述
        sections.append(self._section_overview(rca_event, alert))

        # 2. 根因结论
        sections.append(self._section_root_cause(rca_event, evidence))

        # 3. 影响链路
        sections.append(self._section_impact_chain(rca_event))

        # 4. 证据汇总（含日志证据重点段）
        sections.append(self._section_evidence_summary(evidence))

        # 4a. 日志证据样本（独立成段，优先引用）
        log_section = self._section_log_evidence(log_evidence, alert_service)
        if log_section:
            sections.append(log_section)

        # 5. 历史记忆匹配
        if evidence.get("historical_memory_matches", 0) > 0:
            sections.append(self._section_historical_memory(evidence))

        # 6. 推荐操作
        sections.append(self._section_recommended_actions(rca_event))

        # 7. 回滚方案
        sections.append(self._section_rollback(rca_event, alert_service))

        # 8. 验证步骤
        sections.append(self._section_verification(alert))

        markdown = self._render_markdown(
            title=self._compose_title(rca_event, alert),
            sections=sections,
            metadata=self._compose_metadata(rca_event, alert),
        )

        draft = {
            "title": self._compose_title(rca_event, alert),
            "service": alert_service,
            "root_cause": rca_event.root_cause,
            "confidence": rca_event.confidence,
            "incident_id": rca_event.incident_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sections": sections,
            "markdown": markdown,
            "source": {
                "type": "auto_generated",
                "rca_evidence_keys": list(evidence.keys()),
                "log_evidence_available": bool(log_evidence.get("available")),
            },
        }

        logger.info(
            "Runbook generated",
            incident_id=rca_event.incident_id,
            root_cause=rca_event.root_cause,
            confidence=rca_event.confidence,
            log_evidence_used=bool(log_evidence.get("samples")),
        )

        return draft

    # ==================== Sections ====================

    @staticmethod
    def _section_overview(rca: RCAEvent, alert: AlertEvent | None) -> dict[str, Any]:
        service = alert.service if alert else "未知服务"
        metric = alert.metric if alert else ""
        value = alert.value if alert else None
        threshold = alert.threshold if alert else None
        severity = alert.severity.value if alert else ""

        text = (
            f"- **服务**: `{service}`\n"
            f"- **指标**: `{metric}` (当前 {value}, 阈值 {threshold})\n"
            f"- **严重级别**: `{severity}`\n"
            f"- **故障 ID**: `{rca.incident_id}`\n"
            f"- **置信度**: `{rca.confidence:.2%}`"
        )
        return {"heading": "概述", "kind": "overview", "text": text}

    @staticmethod
    def _section_root_cause(rca: RCAEvent, evidence: dict[str, Any]) -> dict[str, Any]:
        bayesian_top = evidence.get("bayesian_top_causes", []) or []
        top = bayesian_top[0] if bayesian_top else None
        text = (
            f"- **判定根因**: `{rca.root_cause or 'unknown'}`\n"
            f"- **置信度**: `{rca.confidence:.2%}`\n"
        )
        if top:
            text += (
                f"- **贝叶斯后验 Top-1**: `{top.get('cause')}` "
                f"(posterior={top.get('posterior')})\n"
            )
        if rca.contributing_factors:
            text += (
                f"- **贡献因子**: "
                + ", ".join(f"`{c}`" for c in rca.contributing_factors[:5])
                + "\n"
            )
        return {"heading": "根因结论", "kind": "root_cause", "text": text}

    @staticmethod
    def _section_impact_chain(rca: RCAEvent) -> dict[str, Any]:
        if not rca.impact_chain:
            return {
                "heading": "影响链路",
                "kind": "impact_chain",
                "text": "_暂无影响链路数据_",
            }
        items = "\n".join(f"- `{s}`" for s in rca.impact_chain)
        return {
            "heading": "影响链路",
            "kind": "impact_chain",
            "text": f"从根因服务向外传播：\n{items}",
        }

    @staticmethod
    def _section_evidence_summary(evidence: dict[str, Any]) -> dict[str, Any]:
        rag_matches = evidence.get("rag_matches", []) or []
        rag_text = (
            "\n".join(
                f"- `{m.get('id')}` ({m.get('category')})"
                for m in rag_matches[:3]
            )
            or "_无_"
        )
        bayesian_top = evidence.get("bayesian_top_causes", []) or []
        bayesian_text = (
            "\n".join(
                f"- `{b.get('cause')}` posterior={b.get('posterior')}"
                for b in bayesian_top[:3]
            )
            or "_无_"
        )

        log_evidence = evidence.get("log_evidence") or {}
        log_summary = RunbookService._format_log_summary(log_evidence)

        text = (
            "**贝叶斯后验 Top-3**\n"
            f"{bayesian_text}\n\n"
            "**RAG 知识库匹配**\n"
            f"{rag_text}\n\n"
            "**日志证据**\n"
            f"{log_summary}"
        )
        return {"heading": "证据汇总", "kind": "evidence_summary", "text": text}

    @staticmethod
    def _section_log_evidence(
        log_evidence: dict[str, Any],
        service: str,
    ) -> dict[str, Any] | None:
        """如果日志证据可用，单独成段，把样本日志作为 Runbook 的引用证据。"""
        if not log_evidence or not log_evidence.get("available"):
            return {
                "heading": "日志证据（Loki）",
                "kind": "log_evidence",
                "text": "_日志证据不可用：Loki 未配置或未采集到 ERROR 日志_",
                "available": False,
            }

        cause_hits = log_evidence.get("cause_hits", {}) or {}
        samples = log_evidence.get("samples", {}) or {}
        corroborated = log_evidence.get("corroborates_bayesian", False)
        diverged = log_evidence.get("diverges_from_bayesian", False)

        status_line = "✅ 与贝叶斯结论一致（佐证）" if corroborated else (
            "⚠️ 与贝叶斯结论分歧" if diverged else "ℹ️ 仅记录，不参与裁决"
        )

        cause_lines = (
            "\n".join(f"- `{c}` × {n}" for c, n in sorted(
                cause_hits.items(), key=lambda kv: -kv[1]
            ))
            or "_无命中_"
        )

        sample_lines: list[str] = []
        for cause, lines in samples.items():
            sample_lines.append(f"**根因签名 `{cause}`**")
            for line in lines[:3]:
                sample_lines.append(f"```\n{line}\n```")
        sample_text = "\n".join(sample_lines) or "_无样本_"

        text = (
            f"- **数据源**: Loki (`{service}`)\n"
            f"- **条目数**: {log_evidence.get('entries_count', 0)}\n"
            f"- **加成**: {log_evidence.get('confidence_boost', 0):.2%} "
            f"(effective={log_evidence.get('effective_boost', 0):.2%})\n"
            f"- **状态**: {status_line}\n\n"
            f"**根因签名投票**\n{cause_lines}\n\n"
            f"**关键样本日志**\n{sample_text}"
        )
        return {
            "heading": "日志证据（Loki）",
            "kind": "log_evidence",
            "text": text,
            "available": True,
            "entries_count": log_evidence.get("entries_count", 0),
            "samples_count": sum(len(v) for v in samples.values()),
        }

    @staticmethod
    def _section_historical_memory(evidence: dict[str, Any]) -> dict[str, Any]:
        matches = evidence.get("historical_memory_matches", 0)
        top_score = evidence.get("memory_top_score", 0)
        text = (
            f"匹配到 **{matches}** 条历史相似故障记忆，"
            f"最高相似度 `{top_score}`。"
        )
        return {
            "heading": "历史故障匹配",
            "kind": "historical_memory",
            "text": text,
        }

    @staticmethod
    def _section_recommended_actions(rca: RCAEvent) -> dict[str, Any]:
        if not rca.recommended_actions:
            return {
                "heading": "推荐操作",
                "kind": "actions",
                "text": "_暂无推荐操作_",
            }
        items = "\n".join(f"{i+1}. `{a}`" for i, a in enumerate(rca.recommended_actions))
        return {
            "heading": "推荐操作",
            "kind": "actions",
            "text": items,
            "actions": rca.recommended_actions,
        }

    @classmethod
    def _section_rollback(cls, rca: RCAEvent, service: str) -> dict[str, Any]:
        # 用根因的第一段作为分类键（截断到下划线前）
        root_key = (rca.root_cause or "unknown").split(" ")[0].split("(")[0]
        templates = cls.DEFAULT_ROLLBACK.get(root_key, cls.DEFAULT_ROLLBACK["unknown"])
        items = "\n".join(
            f"```bash\n{cls._safe_format(template, service=service)}\n```"
            for template in templates
        )
        return {
            "heading": "回滚方案",
            "kind": "rollback",
            "text": items,
            "service": service,
            "root_cause_key": root_key,
        }

    @staticmethod
    def _safe_format(template: str, **kwargs: Any) -> str:
        """
        模板里若出现未提供的占位符，原样保留并加 `<需人工确认>` 提示，
        避免运行期 KeyError 把整个 Runbook 生成搞挂。
        """
        import re

        result = template
        for match in re.findall(r"\{(\w+)\}", template):
            if match not in kwargs:
                result = result.replace(
                    "{" + match + "}",
                    f"<{match}: 需人工确认>",
                )
        try:
            return result.format(**kwargs)
        except KeyError:
            return re.sub(r"\{(\w+)\}", r"<\1: 需人工确认>", result)

    @staticmethod
    def _section_verification(alert: AlertEvent | None) -> dict[str, Any]:
        metric = alert.metric if alert else "<metric>"
        service = alert.service if alert else "<service>"
        items = "\n".join([
            f"1. 观察 `{metric}` 在 `{service}` 上的趋势是否回归阈值以下",
            "2. 抽样验证受影响接口的成功率与 P95 延迟",
            "3. 确认无新增相关告警（5 分钟窗口）",
            "4. 在值班群同步恢复结论并保留现场 30 分钟",
        ])
        return {"heading": "验证步骤", "kind": "verification", "text": items}

    # ==================== Markdown render ====================

    @staticmethod
    def _render_markdown(
        title: str,
        sections: list[dict[str, Any]],
        metadata: list[str],
    ) -> str:
        meta_block = "\n".join(metadata)
        body_blocks: list[str] = []
        for sec in sections:
            body_blocks.append(f"## {sec['heading']}\n\n{sec['text']}")
        return f"# {title}\n\n{meta_block}\n\n" + "\n\n".join(body_blocks) + "\n"

    @staticmethod
    def _compose_title(rca: RCAEvent, alert: AlertEvent | None) -> str:
        service = alert.service if alert else "未知服务"
        metric = alert.metric if alert else ""
        cause = rca.root_cause or "unknown"
        return f"[Runbook] {service} {metric} 根因={cause}"

    @staticmethod
    def _compose_metadata(rca: RCAEvent, alert: AlertEvent | None) -> list[str]:
        lines = [
            f"- **生成时间**: {datetime.now(timezone.utc).isoformat()}",
            f"- **故障 ID**: `{rca.incident_id}`",
            f"- **置信度**: `{rca.confidence:.2%}`",
        ]
        if alert:
            lines.append(f"- **告警事件 ID**: `{alert.event_id}`")
        return lines

    @staticmethod
    def _format_log_summary(log_evidence: dict[str, Any]) -> str:
        if not log_evidence:
            return "_日志证据不可用_"
        if not log_evidence.get("available"):
            return "_Loki 不可达，未生成日志证据_"
        cause_hits = log_evidence.get("cause_hits") or {}
        if not cause_hits:
            return "未匹配到任何根因签名"
        items = "\n".join(
            f"- `{c}` × {n}"
            for c, n in sorted(cause_hits.items(), key=lambda kv: -kv[1])
        )
        return f"命中签名:\n{items}"


# 单例便于复用
runbook_service = RunbookService()