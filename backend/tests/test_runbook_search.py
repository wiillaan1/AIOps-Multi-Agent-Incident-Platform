"""
Tests for Runbook Search Service + /runbooks/search endpoint.

ChromaDB 在测试环境可能未装；这些测试用 in-memory 模拟来覆盖 service 逻辑。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import _incident_service
from app.services.runbook_search_service import (
    RunbookSearchService,
    _build_search_document,
    _doc_id_for,
)
from app.services.runbook_version_store import RunbookVersion


# ============================ Helpers ============================


class FakeCollection:
    def __init__(self) -> None:
        self._docs: dict[str, dict] = {}

    def upsert(self, *, ids, documents, metadatas):
        for i, doc_id in enumerate(ids):
            self._docs[doc_id] = {
                "document": documents[i],
                "metadata": metadatas[i],
            }

    def delete(self, *, ids):
        for doc_id in ids:
            self._docs.pop(doc_id, None)

    def query(self, *, query_texts, n_results, where=None):
        # 极简语义匹配：关键词与 document 子串匹配
        q = query_texts[0].lower()
        candidates = []
        for doc_id, payload in self._docs.items():
            meta = payload["metadata"]
            if where:
                skip = False
                for k, v in where.items():
                    if isinstance(v, dict) and "$gte" in v:
                        if not (meta.get(k, 0) >= v["$gte"]):
                            skip = True
                            break
                    elif meta.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            doc = payload["document"]
            distance = 0.0 if q in doc.lower() else 1.0
            candidates.append((doc_id, payload, distance))
        # 按距离排序
        candidates.sort(key=lambda t: t[2])
        top = candidates[: n_results]
        return {
            "ids": [[c[0] for c in top]],
            "documents": [[c[1]["document"] for c in top]],
            "metadatas": [[c[1]["metadata"] for c in top]],
            "distances": [[c[2] for c in top]],
        }

    def count(self) -> int:
        return len(self._docs)


class FakeClient:
    def get_or_create_collection(self, *, name, metadata=None):
        return FakeCollection()


@pytest.fixture
def fake_search_service(monkeypatch):
    """用 fake Chroma collection 替换真实的初始化路径。"""
    service = RunbookSearchService(persist_directory="/tmp/fake-chroma")
    shared_collection = FakeCollection()
    shared_client = FakeClient()
    # 缓存 collection 让 get_or_create_collection 每次返回同一个
    cache: dict[str, FakeCollection] = {}

    def _fake_get_or_create(name, metadata=None):
        if name not in cache:
            cache[name] = FakeCollection()
        return cache[name]

    shared_client.get_or_create_collection = _fake_get_or_create

    def _fake_init(self):
        if self._collection is None:
            self._client = shared_client
            self._collection = shared_client.get_or_create_collection(
                name=self.collection_name
            )
        return True

    monkeypatch.setattr(RunbookSearchService, "_ensure_initialized", _fake_init)
    return service


def _make_version(
    *,
    incident_id: str = "INC-1",
    version_number: int = 1,
    title: str = "[Runbook] order-service high_cpu",
    root_cause: str = "resource_exhaustion",
    confidence: float = 0.8,
    service: str = "order-service",
    change_note: str = "",
    markdown: str | None = None,
) -> RunbookVersion:
    if markdown is None:
        markdown = f"# {title}\n\n服务: {service}\n根因: {root_cause}\n"
    return RunbookVersion(
        version_id=f"rbv-{incident_id}-{version_number}",
        incident_id=incident_id,
        version_number=version_number,
        status="published",
        title=title,
        root_cause=root_cause,
        confidence=confidence,
        service=service,
        markdown=markdown,
        change_note=change_note,
        created_at="2026-07-17T10:00:00+00:00",
        updated_at="2026-07-17T10:00:00+00:00",
    )


# ============================ Unit tests ============================


def test_doc_id_format():
    v = _make_version(incident_id="INC-X", version_number=3)
    assert _doc_id_for(v) == "INC-X::v3"


def test_build_search_document_includes_key_fields():
    v = _make_version(
        title="T", root_cause="rc", confidence=0.6, service="svc"
    )
    doc = _build_search_document(v)
    assert "T" in doc
    assert "服务: svc" in doc
    assert "根因: rc" in doc
    assert "60.00%" in doc


def test_index_version_upserts(fake_search_service):
    v = _make_version()
    assert fake_search_service.index_version(v) is True
    stats = fake_search_service.stats()
    assert stats["available"] is True
    assert stats["count"] == 1


def test_index_idempotent(fake_search_service):
    v = _make_version()
    fake_search_service.index_version(v)
    fake_search_service.index_version(v)
    assert fake_search_service.stats()["count"] == 1


def test_search_returns_scored_hits(fake_search_service):
    fake_search_service.index_version(_make_version(
        title="order-service 高 CPU 怎么自愈",
        markdown="# order-service 高 CPU\n## 自愈: scale_up",
        service="order-service",
    ))
    hits = fake_search_service.search("order-service 高 CPU")
    assert len(hits) == 1
    assert hits[0]["service"] == "order-service"
    assert hits[0]["score"] == 1.0


def test_search_with_service_filter(fake_search_service):
    fake_search_service.index_version(_make_version(incident_id="A", service="order-service"))
    fake_search_service.index_version(_make_version(incident_id="B", service="payment-service"))

    hits = fake_search_service.search("CPU", service="order-service")
    assert len(hits) == 1
    assert hits[0]["service"] == "order-service"


def test_search_with_status_filter(fake_search_service):
    fake_search_service.index_version(_make_version(incident_id="A"))
    archived = _make_version(incident_id="B", version_number=2)
    archived.status = "archived"
    fake_search_service.index_version(archived)

    hits = fake_search_service.search("CPU", status="published")
    assert len(hits) == 1
    assert hits[0]["status"] == "published"


def test_search_with_min_confidence(fake_search_service):
    fake_search_service.index_version(_make_version(incident_id="A", confidence=0.5))
    fake_search_service.index_version(_make_version(incident_id="B", confidence=0.9))

    hits = fake_search_service.search("CPU", min_confidence=0.7)
    assert len(hits) == 1
    assert hits[0]["incident_id"] == "B"


def test_search_empty_query_returns_empty(fake_search_service):
    fake_search_service.index_version(_make_version())
    assert fake_search_service.search("") == []
    assert fake_search_service.search("   ") == []


def test_remove_version_drops_doc(fake_search_service):
    fake_search_service.index_version(_make_version(incident_id="X"))
    assert fake_search_service.stats()["count"] == 1
    assert fake_search_service.remove_version("X", 1) is True
    assert fake_search_service.stats()["count"] == 0


# ============================ API endpoint tests ============================


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _seed_incident_with_runbook(runbook: dict) -> str:
    resp = TestClient(app).post(
        "/api/v1/incidents/trigger",
        json={
            "service": "order-service",
            "metric": "cpu_usage_percent",
            "severity": "high",
            "value": 95.0,
            "threshold": 80.0,
            "source": "test",
        },
    )
    assert resp.status_code in (201, 202), resp.text
    iid = resp.json()["incident_id"]
    _incident_service._incidents[iid].context["runbook_draft"] = runbook
    return iid


def test_api_search_validates_query_required(client: TestClient):
    resp = client.get("/api/v1/runbooks/search")
    assert resp.status_code == 422  # FastAPI Query 校验失败


def test_api_search_returns_results(client: TestClient, monkeypatch):
    """注入 fake search service 到 endpoints，验证返回结构。"""
    from app.api import routes

    fake_svc = RunbookSearchService(persist_directory="/tmp/fake-api")
    cache: dict[str, FakeCollection] = {}
    shared_client = FakeClient()

    def _get_or_create(name, metadata=None):
        if name not in cache:
            cache[name] = FakeCollection()
        return cache[name]

    shared_client.get_or_create_collection = _get_or_create

    def _fake_init(self):
        if self._collection is None:
            self._client = shared_client
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            # 预置一条
            self._collection.upsert(
                ids=["INC-1::v1"],
                documents=["order-service high_cpu"],
                metadatas=[{
                    "incident_id": "INC-1",
                    "version_number": 1,
                    "status": "published",
                    "service": "order-service",
                    "root_cause": "resource_exhaustion",
                    "confidence": 0.8,
                    "title": "[Runbook] order high_cpu",
                    "change_note": "",
                    "published_by": "tester",
                    "created_at": "2026-07-17",
                }],
            )
        return True

    monkeypatch.setattr(RunbookSearchService, "_ensure_initialized", _fake_init)
    monkeypatch.setattr(routes, "runbook_search_service", fake_svc, raising=False)

    resp = client.get("/api/v1/runbooks/search?q=order-service+CPU")
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "order-service CPU"
    assert body["count"] == 1
    assert body["hits"][0]["service"] == "order-service"


def test_api_search_with_filters(client: TestClient, monkeypatch):
    from app.api import routes

    fake_svc = RunbookSearchService(persist_directory="/tmp/fake-api2")
    cache: dict[str, FakeCollection] = {}
    shared_client = FakeClient()

    def _get_or_create(name, metadata=None):
        if name not in cache:
            cache[name] = FakeCollection()
        return cache[name]

    shared_client.get_or_create_collection = _get_or_create

    def _fake_init(self):
        if self._collection is None:
            self._client = shared_client
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
            self._collection.upsert(
                ids=["INC-A::v1"],
                documents=["order high_cpu"],
                metadatas=[{
                    "incident_id": "INC-A", "version_number": 1, "status": "published",
                    "service": "order-service", "root_cause": "rc",
                    "confidence": 0.9, "title": "t", "change_note": "",
                    "published_by": "", "created_at": "",
                }],
            )
        return True

    monkeypatch.setattr(RunbookSearchService, "_ensure_initialized", _fake_init)
    monkeypatch.setattr(routes, "runbook_search_service", fake_svc, raising=False)

    resp = client.get(
        "/api/v1/runbooks/search?q=cpu&service=order-service&min_confidence=0.8"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["filters"]["service"] == "order-service"


def test_api_stats_endpoint(client: TestClient, monkeypatch):
    from app.api import routes

    fake_svc = RunbookSearchService(persist_directory="/tmp/fake-stats")
    cache: dict[str, FakeCollection] = {}
    shared_client = FakeClient()

    def _get_or_create(name, metadata=None):
        if name not in cache:
            cache[name] = FakeCollection()
        return cache[name]

    shared_client.get_or_create_collection = _get_or_create

    def _fake_init(self):
        if self._collection is None:
            self._client = shared_client
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name
            )
        return True

    monkeypatch.setattr(RunbookSearchService, "_ensure_initialized", _fake_init)
    monkeypatch.setattr(routes, "runbook_search_service", fake_svc, raising=False)

    resp = client.get("/api/v1/runbooks/stats")
    assert resp.status_code == 200
    assert resp.json()["available"] is True
    assert "count" in resp.json()