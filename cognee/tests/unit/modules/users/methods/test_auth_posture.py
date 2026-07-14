import importlib


auth_module = importlib.import_module(
    "cognee.modules.users.methods.get_authenticated_user"
)


def test_explicitly_disabled_auth_keeps_backend_access_control(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")
    monkeypatch.setenv("REQUIRE_AUTHENTICATION", "false")

    require_authentication, backend_access_control, reason = auth_module._resolve_auth_posture()

    assert require_authentication is False
    assert backend_access_control is True
    assert reason == "explicit REQUIRE_AUTHENTICATION"


def test_backend_access_control_requires_auth_by_default(monkeypatch):
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "true")
    monkeypatch.delenv("REQUIRE_AUTHENTICATION", raising=False)

    require_authentication, backend_access_control, reason = auth_module._resolve_auth_posture()

    assert require_authentication is True
    assert backend_access_control is True
    assert reason == "inherited from ENABLE_BACKEND_ACCESS_CONTROL"
