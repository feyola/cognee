import types
from uuid import uuid4, uuid5, NAMESPACE_OID

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.search.types import SearchType


def _make_user(user_id: str = "u1", tenant_id=None):
    return types.SimpleNamespace(id=user_id, tenant_id=tenant_id)


def _make_dataset(*, name="ds", tenant_id="t1", dataset_id=None, owner_id=None):
    return types.SimpleNamespace(
        id=dataset_id or uuid5(NAMESPACE_OID, name),
        name=name,
        tenant_id=uuid5(NAMESPACE_OID, tenant_id),
        owner_id=owner_id or uuid4(),
    )


@pytest.fixture
def api_search_mod():
    import importlib

    return importlib.import_module("cognee.api.v1.search.search")


@pytest.mark.asyncio
async def test_api_graph_search_passes_feedback_influence_to_search_function(
    monkeypatch, api_search_mod
):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["feedback_influence"] == 0.4
        return ["ok"]

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    out = await api_search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        dataset_ids=[dataset.id],
        feedback_influence=0.4,
    )

    assert out == ["ok"]


@pytest.mark.asyncio
async def test_api_graph_search_omits_unspecified_triplet_penalty(monkeypatch, api_search_mod):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["triplet_distance_penalty"] is None
        return ["ok"]

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    out = await api_search_mod.search(
        query_text="q",
        query_type=SearchType.GRAPH_COMPLETION,
        user=user,
        dataset_ids=[dataset.id],
    )

    assert out == ["ok"]


@pytest.mark.asyncio
async def test_api_code_search_merges_code_query_into_retriever_config(monkeypatch, api_search_mod):
    user = _make_user()
    dataset = _make_dataset()

    async def dummy_set_session_user_context_variable(_user):
        return None

    async def dummy_search_function(**kwargs):
        assert kwargs["retriever_specific_config"] == {
            "existing": True,
            "operation": "find_path",
            "target": "PaymentStore",
        }
        return [{"found": True}]

    monkeypatch.setattr(
        api_search_mod,
        "set_session_user_context_variable",
        dummy_set_session_user_context_variable,
    )
    monkeypatch.setattr(api_search_mod, "search_function", dummy_search_function)

    result = await api_search_mod.search(
        query_text="CheckoutService",
        query_type=SearchType.CODE,
        user=user,
        dataset_ids=[dataset.id],
        retriever_specific_config={"existing": True},
        code_query={"operation": "find_path", "target": "PaymentStore"},
    )

    assert result == [{"found": True}]


@pytest.mark.asyncio
async def test_api_code_query_rejects_non_code_search(api_search_mod):
    with pytest.raises(CogneeValidationError, match="code_query requires"):
        await api_search_mod.search(
            query_text="CheckoutService",
            query_type=SearchType.CHUNKS,
            user=_make_user(),
            code_query={"operation": "explore"},
        )


@pytest.mark.asyncio
async def test_chunk_identity_endpoint_returns_only_live_vector_ids(monkeypatch):
    import importlib
    from contextlib import asynccontextmanager

    from cognee.api.v1.search.routers.get_search_router import ChunkIdentityPayloadDTO

    router_module = importlib.import_module(
        "cognee.api.v1.search.routers.get_search_router"
    )

    requested = [uuid4(), uuid4()]
    dataset_id = uuid4()
    owner_id = uuid4()

    class Vector:
        async def retrieve(self, collection, ids):
            assert collection == "DocumentChunk_text"
            assert ids == [str(value) for value in requested]
            return [types.SimpleNamespace(id=requested[0])]

    async def fake_engine():
        return types.SimpleNamespace(vector=Vector())

    async def fake_permissions(user_id, permission, dataset_ids):
        assert user_id == _make_user().id
        assert permission == "read"
        assert dataset_ids == [dataset_id]
        return [types.SimpleNamespace(id=dataset_id, owner_id=owner_id)]

    @asynccontextmanager
    async def fake_context(selected_dataset_id, selected_owner_id):
        assert selected_dataset_id == dataset_id
        assert selected_owner_id == owner_id
        yield

    monkeypatch.setattr(router_module, "get_unified_engine", fake_engine)
    monkeypatch.setattr(
        router_module, "get_specific_user_permission_datasets", fake_permissions
    )
    monkeypatch.setattr(router_module, "set_database_global_context_variables", fake_context)
    router = router_module.get_search_router()
    endpoint = next(
        route.endpoint for route in router.routes if route.path == "/chunk-identities"
    )

    result = await endpoint(
        ChunkIdentityPayloadDTO(dataset_id=dataset_id, chunk_ids=requested),
        user=_make_user(),
    )

    assert result == [requested[0]]
