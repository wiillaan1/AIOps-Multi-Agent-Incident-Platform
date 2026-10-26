"""Bounded, evidence-first orchestration for root-cause investigation.

The engine only prepares a HealAgent dry-run.  It never executes remediation;
execution is represented solely by a separately accepted trusted receipt.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from app.agents.heal_agent import HealInput
from app.agents.rca_agent import RCAInput
from app.models.agent import AgentExecutionContext
from app.models.events import RCAEvent
from app.models.incident import Incident
from app.models.investigation import (
    EvidenceRecord,
    EvidenceSource,
    HealExecutionReceipt,
    InvestigationRun,
    InvestigationState,
    RecoveryOutcome,
    RecoveryVerification,
    RootCauseHypothesis,
    RootCauseVerificationProfile,
    VerificationStatus,
)
from app.services.assertion_evaluator import (
    QueryContext,
    QueryResult,
    evaluate_assertion,
    evaluate_profile,
    render_query,
)
from app.services.recovery_verifier import RecoverySample, RecoveryVerifier


QueryAdapter = Callable[..., Awaitable[QueryResult | dict[str, Any]]]
Clock = Callable[[], datetime]


class InvestigationLoopEngine:
    """Run at most three RCA/verification rounds and two heal attempts."""

    def __init__(
        self,
        store: Any,
        profiles: Any,
        rca_agent: Any,
        heal_agent: Any,
        metric_query: QueryAdapter,
        log_query: QueryAdapter,
        clock: Clock | None = None,
    ) -> None:
        self.store = store
        self.profiles = profiles
        self.rca_agent = rca_agent
        self.heal_agent = heal_agent
        self.metric_query = metric_query
        self.log_query = log_query
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.recovery_verifier = RecoveryVerifier()

    async def investigate(self, incident: Incident) -> InvestigationRun:
        """Perform deterministic RCA/profile verification up to the third round."""
        if incident.alert_event is None:
            return self._escalate(incident.incident_id, "incident_alert_missing")

        run = self.store.get_or_create(incident.incident_id)
        if run.state in {
            InvestigationState.AWAITING_EXECUTION,
            InvestigationState.RECOVERY_OBSERVING,
            InvestigationState.RECOVERED,
        }:
            return self._snapshot(incident.incident_id)

        rejected = list(run.audit_summary.get("rejected_root_causes", []))
        reflection_context = run.audit_summary.get("reflection_context")
        while run.rca_round < 3:
            run.rca_round += 1
            run.state = (
                InvestigationState.INVESTIGATING
                if run.rca_round == 1
                else InvestigationState.REPLANNING
            )
            rca_event = await self._run_rca(incident, reflection_context)
            if rca_event is None or rca_event.root_cause not in self._standard_causes():
                return self._escalate(incident.incident_id, "rca_unavailable_or_invalid")

            profile = self._profile_or_none(rca_event.root_cause)
            hypothesis = RootCauseHypothesis(
                root_cause=rca_event.root_cause,
                confidence=rca_event.confidence,
                rca_round=run.rca_round,
                profile_version=profile.version if profile else None,
                summary={"rca_event": rca_event.model_dump(mode="json")},
                created_at=self.clock(),
            )
            self.store.record_hypothesis(incident.incident_id, hypothesis)

            if profile is None:
                return self._escalate(incident.incident_id, "profile_missing")
            if profile.manual_only:
                run.verification_status = VerificationStatus.MANUAL_ONLY
                return self._escalate(incident.incident_id, "profile_manual_only")

            decision, baseline = await self._evaluate_profile(incident, profile)
            run.verification_status = decision.status
            hypothesis.evidence_refs = list(run.evidence_refs)
            if decision.status is VerificationStatus.CONFIRMED:
                return await self._prepare_dry_run(incident, rca_event, profile, baseline)

            rejected.append(rca_event.root_cause)
            reflection_context = {
                "rejected_root_causes": sorted(set(rejected)),
                "verification_status": decision.status.value,
                "next_strategies": profile.next_strategies[decision.status.value],
                "unavailable_assertion_ids": list(decision.unavailable_assertion_ids),
            }
            run.audit_summary.update(
                {
                    "rejected_root_causes": reflection_context["rejected_root_causes"],
                    "reflection_context": reflection_context,
                }
            )

        return self._escalate(incident.incident_id, "verification_round_limit")

    async def accept_execution_receipt(
        self, incident: Incident, receipt: HealExecutionReceipt
    ) -> InvestigationRun:
        """Persist a receipt and begin observation only for a succeeded receipt."""
        if receipt.incident_id != incident.incident_id:
            raise ValueError("receipt incident_id must match incident")
        run = self.store.get_or_create(incident.incident_id)
        mismatch = self._dry_run_receipt_mismatch(run, receipt)
        if mismatch is not None:
            run.state = InvestigationState.HEAL_FAILED
            run.audit_summary["receipt_rejected_reason"] = mismatch
            return self._snapshot(incident.incident_id)
        accepted, persisted = self.store.record_receipt(receipt)
        run.audit_summary["receipt_accepted"] = accepted
        if persisted.status == "succeeded":
            run.state = InvestigationState.RECOVERY_OBSERVING
        else:
            run.state = InvestigationState.HEAL_FAILED
            suggestion = self.suggest_plan_b(incident)
            if suggestion is not None:
                run.audit_summary["plan_b_suggestion"] = suggestion
        return self._snapshot(incident.incident_id)

    async def verify_recovery(
        self, incident: Incident, samples: list[RecoverySample] | None = None
    ) -> InvestigationRun:
        """Evaluate supplied observations after exactly one succeeded receipt."""
        run = self.store.get_or_create(incident.incident_id)
        receipt = next((item for item in reversed(run.receipts) if item.status == "succeeded"), None)
        if receipt is None:
            raise ValueError("recovery verification requires a succeeded receipt")
        profile = self._latest_profile(run)
        if profile is None:
            return self._escalate(incident.incident_id, "profile_missing_for_recovery")

        verification = self.recovery_verifier.verify(
            profile,
            receipt,
            dict(run.audit_summary.get("recovery_baseline", {})),
            samples or [],
        )
        run.recovery_verifications.append(verification)
        if verification.outcome is RecoveryOutcome.RECOVERED:
            run.state = InvestigationState.RECOVERED
        elif verification.outcome is RecoveryOutcome.FAILED:
            run.state = InvestigationState.HEAL_FAILED
            suggestion = self.suggest_plan_b(incident)
            if suggestion is not None:
                run.audit_summary["plan_b_suggestion"] = suggestion
        elif not run.audit_summary.get("extra_observation_used", False):
            run.audit_summary["extra_observation_used"] = True
            run.state = InvestigationState.RECOVERY_OBSERVING
        else:
            run.state = InvestigationState.MANUAL_ESCALATION
            run.audit_summary["manual_escalation_reason"] = "recovery_observation_limit"
        return self._snapshot(incident.incident_id)

    def suggest_plan_b(self, incident: Incident) -> dict[str, Any] | None:
        """Return a different playbook candidate; this method never executes it."""
        run = self.store.get_run(incident.incident_id)
        if run is None:
            return None
        dry_run = run.audit_summary.get("dry_run", {})
        original = dry_run.get("playbook_id")
        rca_data = run.audit_summary.get("rca_event")
        if not rca_data or not hasattr(self.heal_agent, "plan_candidates"):
            return None
        try:
            candidates = self.heal_agent.plan_candidates(RCAEvent.model_validate(rca_data), top_k=3)
        except Exception:
            return None
        for candidate in candidates:
            candidate_id = candidate.get("playbook_id") or candidate.get("id")
            if candidate_id and candidate_id != original:
                return dict(candidate)
        return None

    async def _run_rca(
        self, incident: Incident, reflection_context: dict[str, Any] | None
    ) -> RCAEvent | None:
        try:
            result = await self.rca_agent.execute(
                RCAInput(
                    alert=incident.alert_event,
                    incident_id=incident.incident_id,
                    reflection_context=reflection_context,
                ),
                AgentExecutionContext(incident_id=incident.incident_id),
            )
        except Exception:
            return None
        if isinstance(result, RCAEvent):
            return result
        if not getattr(result, "success", False):
            return None
        payload = getattr(result, "output_data", {}).get("rca_event")
        try:
            return RCAEvent.model_validate(payload)
        except Exception:
            return None

    async def _evaluate_profile(
        self, incident: Incident, profile: RootCauseVerificationProfile
    ) -> tuple[Any, dict[str, Any]]:
        alert = incident.alert_event
        context = QueryContext(
            service=alert.service,
            target=alert.labels.get("target", alert.service),
            metric=alert.metric,
            environment=alert.labels.get("environment", incident.environment),
            labels=alert.labels,
        )
        evaluations = []
        baseline = {"threshold": alert.threshold, "log_hits": None}
        for rule in [*profile.assertions, *profile.counter_evidence]:
            try:
                query = render_query(rule.query_template, context)
                adapter = self.metric_query if rule.source is EvidenceSource.PROMETHEUS else self.log_query
                result = await adapter(query, rule=rule, context=context)
                if not isinstance(result, QueryResult):
                    result = QueryResult(**result)
            except Exception as exc:
                result = QueryResult(
                    source=rule.source,
                    available=False,
                    value=None,
                    hits=0,
                    summary={},
                    error=str(exc)[:200],
                )
                query = "render_or_query_failed"
            evaluation = evaluate_assertion(rule, result)
            evaluations.append(evaluation)
            record = EvidenceRecord(
                incident_id=incident.incident_id,
                assertion_id=rule.id,
                source=rule.source,
                outcome=evaluation.outcome,
                observed_at=self.clock(),
                query_summary={"query": query, "source": rule.source.value},
                summary=result.summary,
                score=rule.weight if evaluation.outcome.value in {"pass", "counter"} else None,
            )
            self.store.record_evidence(incident.incident_id, record)
            if rule.source is EvidenceSource.LOKI and result.available:
                baseline["log_hits"] = result.hits
        return evaluate_profile(profile, evaluations), baseline

    async def _prepare_dry_run(
        self,
        incident: Incident,
        rca_event: RCAEvent,
        profile: RootCauseVerificationProfile,
        baseline: dict[str, Any],
    ) -> InvestigationRun:
        run = self.store.get_or_create(incident.incident_id)
        if run.heal_attempt >= 2:
            return self._escalate(incident.incident_id, "heal_attempt_limit")
        try:
            result = await self.heal_agent.execute(
                HealInput(rca_event=rca_event, incident_id=incident.incident_id, dry_run=True),
                AgentExecutionContext(incident_id=incident.incident_id),
            )
        except Exception:
            return self._escalate(incident.incident_id, "heal_dry_run_failed")
        if not getattr(result, "success", False):
            return self._escalate(incident.incident_id, "heal_dry_run_failed")
        heal_event = getattr(result, "output_data", {}).get("heal_event", {})
        run.heal_attempt += 1
        run.state = InvestigationState.AWAITING_EXECUTION
        run.audit_summary.update(
            {
                "rca_event": rca_event.model_dump(mode="json"),
                "profile_version": profile.version,
                "recovery_baseline": baseline,
                "dry_run": {
                    "action_id": heal_event.get("action_id"),
                    "playbook_id": heal_event.get("action_category"),
                    "target_resource": heal_event.get("target_resource"),
                    "executable": heal_event.get("dry_run_result", {}).get("all_executable"),
                },
            }
        )
        return self._snapshot(incident.incident_id)

    @staticmethod
    def _dry_run_receipt_mismatch(
        run: InvestigationRun, receipt: HealExecutionReceipt
    ) -> str | None:
        """Bind every receipt to the exact pending dry-run before observation."""
        dry_run = run.audit_summary.get("dry_run")
        if not isinstance(dry_run, dict):
            return "dry_run_missing"
        for field in ("action_id", "playbook_id", "target_resource"):
            if not dry_run.get(field) or dry_run[field] != getattr(receipt, field):
                return f"dry_run_{field}_mismatch"
        return None

    def _profile_or_none(self, root_cause: str) -> RootCauseVerificationProfile | None:
        try:
            profile = self.profiles.get(root_cause)
            if profile is None:
                return None
            return RootCauseVerificationProfile.model_validate(profile.model_dump())
        except Exception:
            return None

    def _latest_profile(self, run: InvestigationRun) -> RootCauseVerificationProfile | None:
        if not run.hypotheses:
            return None
        return self._profile_or_none(run.hypotheses[-1].root_cause)

    def _escalate(self, incident_id: str, reason: str) -> InvestigationRun:
        run = self.store.get_or_create(incident_id)
        run.state = InvestigationState.MANUAL_ESCALATION
        run.audit_summary["manual_escalation_reason"] = reason
        return self._snapshot(incident_id)

    def _snapshot(self, incident_id: str) -> InvestigationRun:
        run = self.store.get_run(incident_id)
        assert run is not None
        return run

    @staticmethod
    def _standard_causes() -> set[str]:
        return {
            "recent_deployment", "configuration_change", "resource_exhaustion",
            "dependency_failure", "network_issue", "database_issue", "code_bug",
            "traffic_spike", "hardware_failure", "third_party_issue", "memory_leak", "unknown",
        }
