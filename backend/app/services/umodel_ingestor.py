"""Register existing entities in InvestigationGraph (W8 Task 1.3)."""

from __future__ import annotations

from typing import Any, Protocol

from app.models.incident import Incident
from app.models.umodel import (
    EntitySet,
    EntitySetLink,
    EntityState,
    EntityType,
    InvestigationGraph,
    LinkType,
    entity_id,
)
from app.services.umodel_store import InvestigationGraphStore


class RunbookLike(Protocol):
    """Describe the published-runbook interface used by the ingestor."""

    def list_published_runbooks(self) -> list[dict[str, Any]]:
        """Return published runbooks as entity metadata dictionaries."""
        ...


class UModelIngestor:
    """Register incident-related service, workload, and runbook entities."""

    def __init__(
        self,
        store: InvestigationGraphStore,
        runbook_service: RunbookLike | None = None,
    ) -> None:
        """Initialize the ingestor with graph storage and optional runbooks."""
        self.store = store
        self.runbook_service = runbook_service

    def ingest_incident(
        self,
        incident: Incident,
        tenant: str = "default",
    ) -> InvestigationGraph:
        """Register available incident entities and persist the resulting graph."""
        case_id = getattr(incident, "id", incident.incident_id)
        graph = self.store.load(tenant, case_id)
        context = incident.context or {}

        service_name = context.get("service")
        if service_name:
            service_key = entity_id(EntityType.SERVICE, service_name)
            graph.add_entity(
                EntitySet(
                    entity_type=EntityType.SERVICE,
                    entity_id=service_name,
                    state=EntityState.WARNING,
                    attributes={
                        "incident_id": case_id,
                        "severity": incident.severity,
                    },
                )
            )

            pod_name = context.get("pod")
            if pod_name:
                pod_key = entity_id(EntityType.POD, pod_name)
                graph.add_entity(
                    EntitySet(
                        entity_type=EntityType.POD,
                        entity_id=pod_name,
                        state=EntityState.UNKNOWN,
                    )
                )
                graph.add_link(
                    EntitySetLink(
                        src=pod_key,
                        dst=service_key,
                        link_type=LinkType.HOSTED_ON,
                    )
                )

            db_name = context.get("db")
            if db_name:
                db_key = entity_id(EntityType.DB, db_name)
                graph.add_entity(
                    EntitySet(
                        entity_type=EntityType.DB,
                        entity_id=db_name,
                        state=EntityState.UNKNOWN,
                    )
                )
                graph.add_link(
                    EntitySetLink(
                        src=service_key,
                        dst=db_key,
                        link_type=LinkType.READS,
                    )
                )

            topic_name = context.get("topic")
            if topic_name:
                topic_key = entity_id(EntityType.TOPIC, topic_name)
                graph.add_entity(
                    EntitySet(
                        entity_type=EntityType.TOPIC,
                        entity_id=topic_name,
                        state=EntityState.UNKNOWN,
                    )
                )
                graph.add_link(
                    EntitySetLink(
                        src=service_key,
                        dst=topic_key,
                        link_type=LinkType.WRITES,
                    )
                )

            if self.runbook_service is not None:
                self._link_runbooks_to_service(graph, service_key)

        self.store.save(tenant, case_id, graph)
        return graph

    def ingest_runbook(
        self,
        runbook_id: str,
        tenant: str = "default",
    ) -> EntitySet:
        """Create a runbook entity for later graph registration."""
        return EntitySet(entity_type=EntityType.RUNBOOK, entity_id=runbook_id)

    def _link_runbooks_to_service(
        self,
        graph: InvestigationGraph,
        service_key: str,
    ) -> None:
        """Register published runbooks that target the selected service."""
        if self.runbook_service is None:
            return
        try:
            runbooks = self.runbook_service.list_published_runbooks()
        except Exception:
            return

        for runbook in runbooks:
            runbook_id = runbook.get("runbook_id")
            runbook_service = runbook.get("service_name")
            if not runbook_id or not runbook_service:
                continue
            if entity_id(EntityType.SERVICE, runbook_service) != service_key:
                continue

            runbook_key = entity_id(EntityType.RUNBOOK, runbook_id)
            if runbook_key not in graph.entities:
                graph.add_entity(
                    EntitySet(
                        entity_type=EntityType.RUNBOOK,
                        entity_id=runbook_id,
                        attributes={"title": runbook.get("title", "")},
                    )
                )
            graph.add_link(
                EntitySetLink(
                    src=service_key,
                    dst=runbook_key,
                    link_type=LinkType.DEPENDS_ON,
                )
            )
