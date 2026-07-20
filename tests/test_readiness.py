"""Live-but-honest readiness/discovery reconciliation tests.

Ollama HTTP is mocked only at the transport seam. Assertions drive the shipped
gateway and API entry points — never a reimplemented parser.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    GatewayConfig,
    ModelConfig,
    OllamaProviderConfig,
    ServerConfig,
)
from vulcan.gateway import Gateway
from vulcan.providers.ollama import OllamaProvider
from vulcan.readiness import RuntimeProbe, reconcile_configured_models
from vulcan.registry import ConfiguredModel, ModelRegistry

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def _configured(*pairs: tuple[str, str]) -> tuple[ConfiguredModel, ...]:
    return tuple(
        ConfiguredModel(
            id=public_id,
            runtime_name=runtime_name,
            capabilities=frozenset({Capability.CHAT}),
            description=None,
        )
        for public_id, runtime_name in pairs
    )


def test_reconcile_marks_present_and_missing_runtime_names() -> None:
    report = reconcile_configured_models(
        _configured(("public-a", "runtime-a"), ("public-b", "runtime-b")),
        RuntimeProbe(
            live=True,
            provider_availability="available",
            runtime_names=frozenset({"runtime-a", "other"}),
        ),
    )

    assert report.live is True
    assert report.source == "configuration"
    assert report.availability == "available"
    by_id = {item.model_id: item.availability for item in report.models}
    assert by_id == {"public-a": "available", "public-b": "unavailable"}
    # Never invents IDs from the runtime-only name "other"
    assert "other" not in by_id


def test_reconcile_probe_failure_never_fakes_available() -> None:
    configured = _configured(("public-a", "runtime-a"))
    for probe in (
        RuntimeProbe(live=False, provider_availability="unavailable", runtime_names=None),
        RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None),
    ):
        report = reconcile_configured_models(configured, probe)
        assert report.live is False
        assert report.availability == probe.provider_availability
        assert report.models[0].availability == "unchecked"


def test_reconcile_deterministic_style_available_without_live_inventory() -> None:
    report = reconcile_configured_models(
        _configured(("public-a", "runtime-a")),
        RuntimeProbe(live=False, provider_availability="available", runtime_names=None),
    )
    assert report.live is False
    assert report.availability == "available"
    assert report.models[0].availability == "available"


def _ollama_gateway(handler: MockHandler) -> Gateway:
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OllamaProvider(
        OllamaProviderConfig(kind="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0),
        client=client,
    )
    registry = ModelRegistry(
        (
            ModelConfig(
                id="public-chat",
                runtime_name="runtime-chat",
                capabilities=frozenset({Capability.CHAT}),
                description="Configured chat model",
            ),
            ModelConfig(
                id="public-missing",
                runtime_name="runtime-missing",
                capabilities=frozenset({Capability.CHAT}),
            ),
        )
    )
    return Gateway(registry, provider)


def test_gateway_readiness_uses_ollama_tags_without_inventing_models() -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        assert request.method == "GET"
        return httpx.Response(
            200,
            json={"models": [{"name": "runtime-chat"}, {"name": "only-on-runtime"}]},
        )

    gateway = _ollama_gateway(handler)
    try:
        report = asyncio.run(gateway.readiness())
    finally:
        asyncio.run(gateway.provider.aclose())

    assert seen == ["/api/tags"]
    assert report.live is True
    by_id = {item.model_id: item.availability for item in report.models}
    assert by_id == {"public-chat": "available", "public-missing": "unavailable"}
    assert "only-on-runtime" not in by_id


@pytest.mark.parametrize(
    ("status", "body", "expected_provider", "expected_model", "expected_live"),
    [
        (500, {"error": "boom"}, "unavailable", "unchecked", False),
        (200, {"models": "not-a-list"}, "unchecked", "unchecked", False),
        # Missing models key → empty inventory (successful probe, no names).
        (200, {"not": "models"}, "available", "unavailable", True),
    ],
)
def test_gateway_readiness_fail_loud_on_bad_tags(
    status: int,
    body: dict[str, Any],
    expected_provider: str,
    expected_model: str,
    expected_live: bool,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    gateway = _ollama_gateway(handler)
    try:
        report = asyncio.run(gateway.readiness())
    finally:
        asyncio.run(gateway.provider.aclose())

    assert report.live is expected_live
    assert report.provider_availability == expected_provider
    assert {item.availability for item in report.models} == {expected_model}


def test_gateway_readiness_transport_and_timeout_are_honest() -> None:
    async def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://127.0.0.1/"))

    gateway = _ollama_gateway(down)
    try:
        report = asyncio.run(gateway.readiness())
    finally:
        asyncio.run(gateway.provider.aclose())
    assert report.live is False
    assert report.provider_availability == "unavailable"
    assert report.models[0].availability == "unchecked"

    async def timed_out(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=httpx.Request("GET", "http://127.0.0.1/"))

    gateway = _ollama_gateway(timed_out)
    try:
        report = asyncio.run(gateway.readiness())
    finally:
        asyncio.run(gateway.provider.aclose())
    assert report.live is False
    assert report.provider_availability == "unchecked"
    assert report.models[0].availability == "unchecked"


def test_api_models_and_health_surface_live_reconciliation() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "runtime-chat"}]})

    client_http = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OllamaProvider(
        OllamaProviderConfig(kind="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0),
        client=client_http,
    )
    config = GatewayConfig(
        schema_version=1,
        server=ServerConfig(host="127.0.0.1", port=8140, log_level="INFO"),
        provider=OllamaProviderConfig(
            kind="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
        ),
        models=(
            ModelConfig(
                id="public-chat",
                runtime_name="runtime-chat",
                capabilities=frozenset({Capability.CHAT}),
            ),
            ModelConfig(
                id="public-missing",
                runtime_name="runtime-missing",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    with TestClient(create_app(config, provider=provider), base_url="http://127.0.0.1") as client:
        health = client.get("/healthz")
        models = client.get("/v1/models")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["provider"] == {"kind": "ollama", "availability": "available"}
    assert health.json()["models_configured"] == 2

    assert models.status_code == 200
    body = models.json()
    assert body["discovery"] == {
        "source": "configuration",
        "live": True,
        "availability": "available",
    }
    by_id = {row["id"]: row["availability"] for row in body["data"]}
    assert by_id == {"public-chat": "available", "public-missing": "unavailable"}
    assert "runtime-chat" not in models.text
    assert "runtime-missing" not in models.text


def test_api_models_stay_unchecked_when_ollama_is_down() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://127.0.0.1/"))

    client_http = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OllamaProvider(
        OllamaProviderConfig(kind="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0),
        client=client_http,
    )
    config = GatewayConfig(
        schema_version=1,
        server=ServerConfig(host="127.0.0.1", port=8140, log_level="INFO"),
        provider=OllamaProviderConfig(
            kind="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
        ),
        models=(
            ModelConfig(
                id="public-chat",
                runtime_name="runtime-chat",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    with TestClient(create_app(config, provider=provider), base_url="http://127.0.0.1") as client:
        health = client.get("/healthz").json()
        models = client.get("/v1/models").json()

    assert health["status"] == "ok"  # process liveness independent of provider
    assert health["provider"]["availability"] == "unavailable"
    assert models["discovery"] == {
        "source": "configuration",
        "live": False,
        "availability": "unavailable",
    }
    assert models["data"][0]["availability"] == "unchecked"
