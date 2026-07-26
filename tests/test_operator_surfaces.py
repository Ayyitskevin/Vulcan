"""Operator surfaces: configurable probe TTL, model retrieve, readiness logs."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Literal

import httpx
import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    GatewayConfig,
    ModelConfig,
    OllamaProviderConfig,
    ReadinessConfig,
    ServerConfig,
)
from vulcan.gateway import Gateway
from vulcan.observability import SafeJsonFormatter
from vulcan.providers.ollama import OllamaProvider
from vulcan.readiness import RuntimeProbe
from vulcan.registry import ModelRegistry

PROVIDER_ID = "local-ollama"


def _ollama_app(handler, *, ttl: float = 5.0) -> tuple[TestClient, httpx.AsyncClient]:
    client_http = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OllamaProvider(
        PROVIDER_ID,
        OllamaProviderConfig(type="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0),
        client=client_http,
    )
    config = GatewayConfig(
        schema_version=2,
        server=ServerConfig(host="127.0.0.1", port=8140, log_level="INFO"),
        providers={
            PROVIDER_ID: OllamaProviderConfig(
                type="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
            )
        },
        readiness=ReadinessConfig(probe_ttl_seconds=ttl),
        models=(
            ModelConfig(
                id="public-chat",
                provider=PROVIDER_ID,
                provider_model="runtime-chat",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    return (
        TestClient(
            create_app(config, providers={PROVIDER_ID: provider}), base_url="http://127.0.0.1"
        ),
        client_http,
    )


def test_zero_ttl_probes_every_call() -> None:
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, json={"models": [{"name": "runtime-chat"}]})

    client, _ = _ollama_app(handler, ttl=0.0)
    with client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/models/public-chat").status_code == 200
    assert hits == 3


def test_get_model_shares_probe_cache_and_refresh() -> None:
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, json={"models": [{"name": "runtime-chat"}]})

    client, _ = _ollama_app(handler, ttl=5.0)
    with client:
        assert client.get("/v1/models/public-chat").json()["availability"] == "available"
        assert client.get("/v1/models/public-chat").json()["availability"] == "available"
        assert hits == 1
        assert (
            client.get("/v1/models/public-chat", params={"refresh": "true"}).json()["availability"]
            == "available"
        )
        assert hits == 2


def test_readiness_logs_probe_and_reuse_without_runtime_names(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class CountingProvider:
        provider_type: Literal["ollama"] = "ollama"
        provider_id = PROVIDER_ID

        def __init__(self) -> None:
            self.calls = 0

        async def discover_runtime(self) -> RuntimeProbe:
            self.calls += 1
            return RuntimeProbe(
                live=True,
                provider_availability="available",
                runtime_names=frozenset({"runtime-chat", "secret-runtime-name"}),
            )

        async def chat(self, request):  # pragma: no cover
            raise AssertionError("chat not used")

        async def aclose(self) -> None:
            return None

    provider = CountingProvider()
    gateway = Gateway(
        ModelRegistry(
            (
                ModelConfig(
                    id="public-chat",
                    provider=PROVIDER_ID,
                    provider_model="runtime-chat",
                    capabilities=frozenset({Capability.CHAT}),
                ),
            )
        ),
        {PROVIDER_ID: provider},
        readiness_ttl_seconds=5.0,
    )
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter())
    log = logging.getLogger("vulcan.gateway")
    log.handlers = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False

    asyncio.run(gateway.readiness())
    asyncio.run(gateway.readiness())
    handler.flush()
    lines = [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]
    events = [row["event"] for row in lines]
    assert events == ["readiness_probed", "readiness_reused"]
    assert lines[0]["providers_configured"] == 1
    assert lines[0]["models_available"] == 1
    assert lines[0]["forced"] is False
    assert lines[0]["reused"] is False
    assert lines[1]["reused"] is True
    serialized = stream.getvalue()
    assert "secret-runtime-name" not in serialized
    assert "runtime-chat" not in serialized
    assert provider.calls == 1


def test_create_app_honors_config_probe_ttl() -> None:
    hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal hits
        hits += 1
        return httpx.Response(200, json={"models": [{"name": "runtime-chat"}]})

    client, _ = _ollama_app(handler, ttl=0.0)
    with client:
        client.get("/healthz")
        client.get("/healthz")
    assert hits == 2
