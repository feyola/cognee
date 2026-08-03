from fastapi import APIRouter, Depends

from cognee.modules.users.get_fastapi_users import get_fastapi_users
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models.User import User, UserRead, UserUpdate


def get_users_router():
    router = APIRouter()

    # FastAPI Users wires GET /me directly to its mandatory current-user
    # dependency. Register Cognee's posture-aware version first so trusted
    # deployments resolve the shared default user, while authenticated
    # deployments keep enforcing a real session. The stock route remains in
    # the schema and provides the other /me methods unchanged.
    @router.get("/me", response_model=UserRead, include_in_schema=False)
    async def get_current_user(user: User = Depends(get_authenticated_user)):
        return user

    router.include_router(get_fastapi_users().get_users_router(UserRead, UserUpdate))
    return router
