"""Regression test: the cognify router binds its arguments by name.

``cognify()`` gained ``data_ids`` as its second parameter while the router still
passed ``(datasets, user, ...)`` positionally, so the authenticated ``User`` was
bound to ``data_ids`` and every ``POST /api/v1/cognify`` failed with
``'User' object is not iterable``. The fake below binds the router's call
against the real signature, so any future reordering fails here.
"""

import importlib
import inspect
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

cognify_router_module = importlib.import_module("cognee.api.v1.cognify.routers.get_cognify_router")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(cognify_router_module.get_cognify_router(), prefix="/api/v1/cognify")
    user = SimpleNamespace(
        id=str(uuid.uuid4()), email="default@example.com", is_active=True, tenant_id=None
    )

    async def override_get_authenticated_user():
        return user

    cognify_router_module.send_telemetry = lambda *a, **k: None
    app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    with TestClient(app) as test_client:
        yield test_client, user


def test_cognify_router_passes_user_as_user(client, monkeypatch):
    test_client, user = client

    import cognee.api.v1.cognify as cognify_pkg

    real_signature = inspect.signature(cognify_pkg.cognify)
    bound = {}

    async def fake_cognify(*args, **kwargs):
        bound.update(real_signature.bind(*args, **kwargs).arguments)
        return {}

    monkeypatch.setattr(cognify_pkg, "cognify", fake_cognify)

    response = test_client.post("/api/v1/cognify", json={"datasets": ["ds"]})

    assert response.status_code == 200, response.text
    assert bound["user"] is user
    assert bound["datasets"] == ["ds"]
    assert bound.get("data_ids") is None
