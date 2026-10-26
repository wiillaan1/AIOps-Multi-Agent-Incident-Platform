"""Test RCAInput._reflection_context + _synthesize_analysis weight shift (W7)."""

from app.agents.rca_agent import BayesianNode, RCAInput, RCAAgent
from app.models.events import AlertEvent, SeverityLevel


def _alert():
    return AlertEvent(
        service="order-service",
        metric="cpu_usage_percent",
        value=95.0,
        threshold=80.0,
        severity=SeverityLevel.HIGH,
        labels={"tier": "critical"},
        annotations={},
    )


def _bayes():
    return [
        BayesianNode(name="resource_exhaustion", prior=0.2, likelihood=0.3,
                     posterior=0.15, evidence_strength=0.2),
        BayesianNode(name="recent_deployment", prior=0.18, likelihood=0.25,
                     posterior=0.13, evidence_strength=0.2),
    ]


def test_synthesize_default_weights_when_no_reflection_context():
    agent = RCAAgent()
    rca = RCAInput(alert=_alert(), incident_id="INC-1")
    root, conf, evi = agent._synthesize_analysis(
        bayesian_results=_bayes(), rag_results=[], impact_chain=[], alert=_alert(),
        memory_results=[], log_evidence={"available": False, "entries_count": 0},
    )
    # 默认权重：log_evidence 写入 evidence 的 effective_boost 字段（计算后再读取）
    assert isinstance(root, str)
    assert isinstance(conf, float)
    # _reflection_context 字段已存在，且默认 None
    assert rca._reflection_context is None


def test_synthesize_boosts_log_weight_when_reflection_context_set():
    """带 _reflection_context={"evidence_boost":"log"} 时 log effective_boost 比例上升。"""
    agent = RCAAgent()
    log_evi = {
        "available": True,
        "entries_count": 5,
        "boosted_cause": "resource_exhaustion",
        "cause_hits": {"resource_exhaustion": 3},
        "confidence_boost": 0.10,
        "samples": {"resource_exhaustion": ["oom sample"]},
        "source": "loki",
    }
    rca = RCAInput(
        alert=_alert(),
        incident_id="INC-2",
        _reflection_context={"evidence_boost": "log", "bayesian_penalty": 0.15},
    )
    root, conf, evi = agent._synthesize_analysis(
        bayesian_results=_bayes(), rag_results=[], impact_chain=[], alert=_alert(),
        memory_results=[], log_evidence=log_evi,
        reflection_context=rca._reflection_context,
    )
    # _synthesize_analysis 会读 evidence["log_evidence"]["effective_boost"]
    assert evi["log_evidence"]["effective_boost"] > 0.10


def test_rca_input_accepts_reflection_context_via_alias():
    """RCAInput 应支持 _reflection_context 别名传入（W7 重跑管道使用）。"""
    rca = RCAInput(
        alert=_alert(),
        incident_id="INC-3",
        _reflection_context={"evidence_boost": "log"},
    )
    assert rca._reflection_context == {"evidence_boost": "log"}