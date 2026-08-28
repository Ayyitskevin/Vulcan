"""Contract snapshot for the public ``/openapi.json`` surface.

Pins structure, never the full blob: the exact set of public paths and
methods, plus the error status codes declared per route. A full-JSON
snapshot would break on every harmless schema tweak; these assertions break
only when the contract itself changes — the case that needs an intentional
decision and a README update, per the documented v1 contract.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

# The complete public surface: no endpoint may exist that is not the
# documented gateway contract (README "API contract"), and no documented
# endpoint may silently disappear.
EXPECTED_PUBLIC_SURFACE: dict[str, list[str]] = {
    "/healthz": ["get"],
    "/v1/models": ["get"],
    "/v1/models/{model_id}": ["get"],
    "/v1/capabilities": ["get"],
    "/v1/chat/completions": ["post"],
    "/v1/embeddings": ["post"],
    "/v1/usage": ["get"],
}


def _openapi(client: TestClient) -> dict[str, object]:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    return response.json()


def test_openapi_surface_is_exactly_the_documented_contract(client: TestClient) -> None:
    schema = _openapi(client)
    surface = {
        path: sorted(method for method in item if method in {"get", "post"})
        for path, item in schema["paths"].items()
    }
    assert surface == EXPECTED_PUBLIC_SURFACE


def test_every_route_declares_a_success_shape(client: TestClient) -> None:
    schema = _openapi(client)
    for path, item in schema["paths"].items():
        for method, operation in item.items():
            assert "200" in operation["responses"], f"{method.upper()} {path} lost its 200 shape"


def test_v1_routes_declare_the_core_error_contract(client: TestClient) -> None:
    # 404 unknown alias, 422 contract violation, 503 provider/credential
    # failure: the stable error envelope the README pins for /v1 traffic.
    schema = _openapi(client)
    for path, item in schema["paths"].items():
        if not path.startswith("/v1"):
            continue
        for method, operation in item.items():
            declared = set(operation["responses"])
            for code in ("404", "422", "503"):
                assert code in declared, f"{method.upper()} {path} no longer declares {code}"


def test_healthz_declares_invalid_host(client: TestClient) -> None:
    schema = _openapi(client)
    assert "400" in schema["paths"]["/healthz"]["get"]["responses"]


def test_no_route_declares_an_auth_failure(client: TestClient) -> None:
    # 401 appears nowhere: Vulcan deliberately has no authentication layer
    # (README "What Vulcan does not do"); the loopback+Host boundary is the
    # only admission control. If a future phase adds auth, this test is the
    # flag that the contract changed on purpose.
    schema = _openapi(client)
    declared = {
        code
        for item in schema["paths"].values()
        for operation in item.values()
        for code in operation["responses"]
    }
    assert "401" not in declared
