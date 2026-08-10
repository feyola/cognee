from importlib import import_module

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute


visualize_module = import_module("cognee.api.v1.users.routers.get_visualize_router")


@pytest.mark.asyncio
async def test_visualize_rejects_missing_user_by_default(monkeypatch):
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_LOCAL_VISUALIZE", raising=False)

    with pytest.raises(HTTPException) as error:
        await visualize_module.get_visualize_user(None)

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_visualize_default_user_requires_explicit_opt_in(monkeypatch):
    default_user = object()

    async def get_default_user():
        return default_user

    monkeypatch.setenv("ALLOW_UNAUTHENTICATED_LOCAL_VISUALIZE", "true")
    monkeypatch.setattr(visualize_module, "get_default_user", get_default_user)

    assert await visualize_module.get_visualize_user(None) is default_user


def test_every_visualize_route_uses_the_posture_aware_dependency():
    routes = visualize_module.get_visualize_router().routes

    assert routes
    for route in routes:
        assert isinstance(route, APIRoute)
        dependency_calls = {dependency.call for dependency in route.dependant.dependencies}
        assert visualize_module.get_visualize_user in dependency_calls, route.path
