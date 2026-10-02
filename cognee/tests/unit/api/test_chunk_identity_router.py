"""Bounded, authorization-first exact identity lookup; no retrieval or LLM fallback."""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.exceptions import CogneeApiError
from cognee.modules.users.exceptions import PermissionDeniedError

router_module = importlib.import_module("cognee.api.v1.search.routers.get_search_router")


@pytest.fixture
def lookup(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset = SimpleNamespace(id=uuid4(), owner_id=uuid4())
    permission = AsyncMock(return_value=[dataset])
    retrieve = AsyncMock(return_value=[])
    engine = AsyncMock(return_value=SimpleNamespace(vector=SimpleNamespace(retrieve=retrieve)))
    events = []

    @asynccontextmanager
    async def scope(dataset_id, owner_id):
        events.append(("enter", dataset_id, owner_id))
        try:
            yield
        finally:
            events.append(("exit", dataset_id, owner_id))

    monkeypatch.setattr(router_module, "get_specific_user_permission_datasets", permission)
    monkeypatch.setattr(router_module, "set_database_global_context_variables", scope)
    monkeypatch.setattr(router_module, "get_unified_engine", engine)
    app = FastAPI()
    app.include_router(router_module.get_search_router(), prefix="/api/v1/search")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: user

    @app.exception_handler(CogneeApiError)
    async def cognee_error(_request, error):
        return JSONResponse(status_code=error.status_code, content={"detail": "denied"})

    return SimpleNamespace(
        client=TestClient(app, raise_server_exceptions=False),
        app=app,
        user=user,
        dataset=dataset,
        permission=permission,
        retrieve=retrieve,
        engine=engine,
        events=events,
    )


@pytest.mark.parametrize("camel_case", [False, True])
def test_returns_only_requested_present_ids_in_order_and_scopes_to_owner(lookup, camel_case):
    first, second, missing, unexpected = [uuid4() for _ in range(4)]
    lookup.retrieve.return_value = [
        SimpleNamespace(id=str(second)),
        SimpleNamespace(id=unexpected),
        SimpleNamespace(id=first),
        SimpleNamespace(id=first),
    ]
    dataset_key, chunks_key = (
        ("datasetId", "chunkIds") if camel_case else ("dataset_id", "chunk_ids")
    )
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={
            dataset_key: str(lookup.dataset.id),
            chunks_key: list(map(str, [first, missing, second, first])),
        },
    )
    assert response.status_code == 200
    assert response.json() == [str(first), str(second)]
    lookup.permission.assert_awaited_once_with(lookup.user.id, "read", [lookup.dataset.id])
    lookup.retrieve.assert_awaited_once_with(
        "DocumentChunk_text", list(map(str, [first, missing, second]))
    )
    assert lookup.events == [
        ("enter", lookup.dataset.id, lookup.dataset.owner_id),
        ("exit", lookup.dataset.id, lookup.dataset.owner_id),
    ]


def test_unknown_or_foreign_ids_are_not_verified(lookup):
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={"datasetId": str(lookup.dataset.id), "chunkIds": [str(uuid4())]},
    )
    assert response.status_code == 200
    assert response.json() == []


def test_permission_denial_happens_before_scope_or_backend(lookup):
    lookup.permission.side_effect = PermissionDeniedError(log=False)
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={"datasetId": str(uuid4()), "chunkIds": [str(uuid4())]},
    )
    assert response.status_code == 403
    lookup.engine.assert_not_awaited()
    lookup.retrieve.assert_not_awaited()
    assert lookup.events == []


def test_unauthenticated_request_never_checks_dataset_or_backend(lookup):
    def unauthenticated():
        raise HTTPException(status_code=401, detail="Authentication required")

    lookup.app.dependency_overrides[router_module.get_authenticated_user] = unauthenticated
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={"datasetId": str(lookup.dataset.id), "chunkIds": [str(uuid4())]},
    )
    assert response.status_code == 401
    lookup.permission.assert_not_awaited()
    lookup.engine.assert_not_awaited()


@pytest.mark.parametrize("chunk_ids", [[], [str(uuid4())] * 101, ["not-a-uuid"]])
def test_invalid_or_unbounded_requests_never_touch_storage(lookup, chunk_ids):
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={"datasetId": str(lookup.dataset.id), "chunkIds": chunk_ids},
    )
    assert response.status_code == 422
    lookup.permission.assert_not_awaited()
    lookup.engine.assert_not_awaited()


def test_backend_failure_is_not_a_successful_empty_verification(lookup):
    lookup.retrieve.side_effect = RuntimeError("backend unavailable")
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={"datasetId": str(lookup.dataset.id), "chunkIds": [str(uuid4())]},
    )
    assert response.status_code == 500
    assert lookup.events[-1] == ("exit", lookup.dataset.id, lookup.dataset.owner_id)


def test_exactly_one_hundred_ids_are_accepted(lookup):
    chunk_ids = [uuid4() for _ in range(100)]
    lookup.retrieve.return_value = [SimpleNamespace(id=chunk_id) for chunk_id in chunk_ids]
    response = lookup.client.post(
        "/api/v1/search/chunk-identities",
        json={"datasetId": str(lookup.dataset.id), "chunkIds": list(map(str, chunk_ids))},
    )
    assert response.status_code == 200
    assert response.json() == list(map(str, chunk_ids))


def test_openapi_declares_bounded_uuid_wire_contract(lookup):
    schemas = lookup.app.openapi()["components"]["schemas"]
    fields = schemas["ChunkIdentityPayloadDTO"]["properties"]
    assert fields["datasetId"]["format"] == "uuid"
    assert fields["chunkIds"]["minItems"] == 1
    assert fields["chunkIds"]["maxItems"] == 100
    assert fields["chunkIds"]["items"]["format"] == "uuid"
