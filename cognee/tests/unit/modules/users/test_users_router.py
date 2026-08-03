from fastapi.routing import APIRoute

from cognee.api.v1.users.routers.get_users_router import get_users_router
from cognee.modules.users.methods import get_authenticated_user


def test_get_users_router_prefers_posture_aware_me_route():
    me_routes = [
        route
        for route in get_users_router().routes
        if isinstance(route, APIRoute) and route.path == "/me" and "GET" in route.methods
    ]

    assert len(me_routes) == 1
    assert me_routes[0].include_in_schema is False
    assert me_routes[0].dependant.dependencies[0].call is get_authenticated_user
