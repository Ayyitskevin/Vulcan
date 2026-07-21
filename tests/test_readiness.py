"""Live-but-honest readiness/discovery reconciliation tests.

Ollama HTTP is mocked only at the transport seam. Assertions drive the shipped
gateway and API entry points — never a reimplemented parser.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, Literal

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
from vulcan.errors import ModelUnavailableError
from vulcan.gateway import Gateway
from vulcan.providers.base import ProviderChatRequest, ProviderChatResult
from vulcan.providers.ollama import OllamaProvider
from vulcan.readiness import (
    READINESS_PROBE_TTL_SECONDS,
    RuntimeProbe,
    reconcile_configured_models,
    runtime_name_matches,
)
from vulcan.registry import ConfiguredModel, ModelRegistry
from vulcan.schemas import ChatCompletionRequest, ChatMessage, MessageRole

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


def test_runtime_name_matches_exact_and_unambiguous_tagged_only() -> None:
    live = frozenset({"llama3:latest", "mistral:7b", "exact-name"})
    assert runtime_name_matches("exact-name", live) is True
    assert runtime_name_matches("llama3:latest", live) is True
    # Untagged config matches the single tagged form
    assert runtime_name_matches("llama3", live) is True
    assert runtime_name_matches("mistral", live) is True
    assert runtime_name_matches("missing", live) is False
    assert runtime_name_matches("", live) is False
    # Multi-tag collision must not claim available
    collision = frozenset({"llama3:latest", "llama3:8b"})
    assert runtime_name_matches("llama3", collision) is False
    # Tagged config never uses the untagged expansion rule
    assert runtime_name_matches("llama3:8b", frozenset({"llama3:latest"})) is False


def test_reconcile_uses_tagged_match_without_inventing_ids() -> None:
    report = reconcile_configured_models(
        _configured(("public-short", "llama3"), ("public-exact", "mistral:7b")),
        RuntimeProbe(
            live=True,
            provider_availability="available",
            runtime_names=frozenset({"llama3:latest", "mistral:7b", "only-live"}),
        ),
    )
    by_id = {item.model_id: item.availability for item in report.models}
    assert by_id == {"public-short": "available", "public-exact": "available"}
    assert "only-live" not in by_id

    ambiguous = reconcile_configured_models(
        _configured(("public-short", "llama3")),
        RuntimeProbe(
            live=True,
            provider_availability="available",
            runtime_names=frozenset({"llama3:latest", "llama3:8b"}),
        ),
    )
    assert ambiguous.models[0].availability == "unavailable"


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
    assert health.json()["provider"] == {"kind": "ollama", "availability": "unchecked"}
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
    assert health["provider"]["availability"] == "unchecked"
    assert models["discovery"] == {
        "source": "configuration",
        "live": False,
        "availability": "unavailable",
    }
    assert models["data"][0]["availability"] == "unchecked"


# ── Hardening: bounded probe reuse + chat preflight ──────────────────────────


class _CountingClock:
    def __init__(self, start: float = 1_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _LiveInventoryProvider:
    """Records discover/chat calls; returns a fixed live name set."""

    kind: Literal["ollama"] = "ollama"

    def __init__(self, names: frozenset[str]) -> None:
        self.names = names
        self.discover_calls = 0
        self.chat_calls: list[ProviderChatRequest] = []

    async def discover_runtime(self) -> RuntimeProbe:
        self.discover_calls += 1
        return RuntimeProbe(live=True, provider_availability="available", runtime_names=self.names)

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.chat_calls.append(request)
        return ProviderChatResult(content="should-not-matter", finish_reason="stop")

    async def aclose(self) -> None:
        return None


def test_readiness_reuses_probe_within_ttl_and_reprobes_after() -> None:
    assert READINESS_PROBE_TTL_SECONDS == 5.0
    clock = _CountingClock(start=100.0)
    provider = _LiveInventoryProvider(frozenset({"runtime-chat"}))
    gateway = Gateway(
        ModelRegistry(
            (
                ModelConfig(
                    id="public-chat",
                    runtime_name="runtime-chat",
                    capabilities=frozenset({Capability.CHAT}),
                ),
            )
        ),
        provider,
        clock=clock,
        readiness_ttl_seconds=READINESS_PROBE_TTL_SECONDS,
    )

    first = asyncio.run(gateway.readiness())
    second = asyncio.run(gateway.readiness())
    assert provider.discover_calls == 1
    assert first is second  # same cached report object within TTL

    clock.advance(READINESS_PROBE_TTL_SECONDS - 0.001)
    asyncio.run(gateway.readiness())
    assert provider.discover_calls == 1

    clock.advance(0.002)
    third = asyncio.run(gateway.readiness())
    assert provider.discover_calls == 2
    assert third is not first

    # force always probes even inside the bound
    asyncio.run(gateway.readiness(force=True))
    assert provider.discover_calls == 3


def test_api_health_and_models_share_one_tags_probe_within_ttl() -> None:
    tags_hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal tags_hits
        assert request.url.path == "/api/tags"
        tags_hits += 1
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
        ),
    )
    with TestClient(create_app(config, provider=provider), base_url="http://127.0.0.1") as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert client.get("/healthz").status_code == 200
    assert tags_hits == 1


def test_chat_preflight_skips_provider_when_live_list_proved_missing() -> None:
    provider = _LiveInventoryProvider(frozenset({"other-runtime"}))
    gateway = Gateway(
        ModelRegistry(
            (
                ModelConfig(
                    id="public-chat",
                    runtime_name="runtime-chat",
                    capabilities=frozenset({Capability.CHAT}),
                ),
            )
        ),
        provider,
    )
    request = ChatCompletionRequest(
        model="public-chat",
        messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
    )

    with pytest.raises(ModelUnavailableError) as caught:
        asyncio.run(gateway.chat(request))

    assert caught.value.code == "model_unavailable"
    assert caught.value.details == {"model": "public-chat"}
    assert provider.chat_calls == []
    assert provider.discover_calls == 1


def test_chat_still_calls_provider_when_probe_is_unchecked() -> None:
    class DownProvider:
        kind: Literal["ollama"] = "ollama"

        def __init__(self) -> None:
            self.chat_calls = 0
            self.discover_calls = 0

        async def discover_runtime(self) -> RuntimeProbe:
            self.discover_calls += 1
            return RuntimeProbe(live=False, provider_availability="unavailable", runtime_names=None)

        async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
            del request
            self.chat_calls += 1
            return ProviderChatResult(content="recovered", finish_reason="stop")

        async def aclose(self) -> None:
            return None

    provider = DownProvider()
    gateway = Gateway(
        ModelRegistry(
            (
                ModelConfig(
                    id="public-chat",
                    runtime_name="runtime-chat",
                    capabilities=frozenset({Capability.CHAT}),
                ),
            )
        ),
        provider,
    )
    response = asyncio.run(
        gateway.chat(
            ChatCompletionRequest(
                model="public-chat",
                messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
            )
        )
    )
    assert response.choices[0].message.content == "recovered"
    assert provider.discover_calls == 1
    assert provider.chat_calls == 1


def test_chat_available_model_still_reaches_provider() -> None:
    provider = _LiveInventoryProvider(frozenset({"runtime-chat"}))
    gateway = Gateway(
        ModelRegistry(
            (
                ModelConfig(
                    id="public-chat",
                    runtime_name="runtime-chat",
                    capabilities=frozenset({Capability.CHAT}),
                ),
            )
        ),
        provider,
    )
    response = asyncio.run(
        gateway.chat(
            ChatCompletionRequest(
                model="public-chat",
                messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
            )
        )
    )
    assert response.choices[0].message.content == "should-not-matter"
    assert len(provider.chat_calls) == 1
    assert provider.discover_calls == 1


def test_api_chat_returns_model_unavailable_without_chat_post() -> None:
    posts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        posts.append(f"{request.method} {request.url.path}")
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "only-other"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"role": "assistant", "content": "nope"}})
        return httpx.Response(404)

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
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public-chat",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "model_unavailable"
    assert body["error"]["details"] == {"model": "public-chat"}
    assert posts == ["GET /api/tags"]
    assert "POST /api/chat" not in posts


# ── Ops honesty: force refresh + cache invalidation ──────────────────────────


def test_api_refresh_forces_second_tags_get_inside_ttl() -> None:
    tags_hits = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal tags_hits
        assert request.url.path == "/api/tags"
        tags_hits += 1
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
        ),
    )
    with TestClient(create_app(config, provider=provider), base_url="http://127.0.0.1") as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/models").status_code == 200
        assert tags_hits == 1
        assert client.get("/healthz", params={"refresh": "true"}).status_code == 200
        assert tags_hits == 1
        assert client.get("/v1/models", params={"refresh": "true"}).status_code == 200
        assert tags_hits == 2
        # Without force, reuse holds again
        assert client.get("/v1/models").status_code == 200
        assert tags_hits == 2


def test_provider_model_unavailable_invalidates_readiness_cache() -> None:
    """Stale 'available' must not stick after provider proves model gone."""

    class FlipInventoryProvider:
        kind: Literal["ollama"] = "ollama"

        def __init__(self) -> None:
            self.discover_calls = 0
            self.chat_calls = 0
            self.names = frozenset({"runtime-chat"})

        async def discover_runtime(self) -> RuntimeProbe:
            self.discover_calls += 1
            return RuntimeProbe(
                live=True,
                provider_availability="available",
                runtime_names=self.names,
            )

        async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
            del request
            self.chat_calls += 1
            # Provider proves absence after preflight thought it was available.
            raise ModelUnavailableError("public-chat")

        async def aclose(self) -> None:
            return None

    provider = FlipInventoryProvider()
    gateway = Gateway(
        ModelRegistry(
            (
                ModelConfig(
                    id="public-chat",
                    runtime_name="runtime-chat",
                    capabilities=frozenset({Capability.CHAT}),
                ),
            )
        ),
        provider,
    )
    # Warm cache: model available
    first = asyncio.run(gateway.readiness())
    assert first.models[0].availability == "available"
    assert provider.discover_calls == 1

    with pytest.raises(ModelUnavailableError):
        asyncio.run(
            gateway.chat(
                ChatCompletionRequest(
                    model="public-chat",
                    messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
                )
            )
        )
    assert provider.chat_calls == 1

    # After model_unavailable, inventory is invalid; next readiness re-probes.
    # Simulate runtime now missing the model on the re-probe.
    provider.names = frozenset({"other-only"})
    second = asyncio.run(gateway.readiness())
    assert provider.discover_calls == 2
    assert second.models[0].availability == "unavailable"


def test_api_models_tagged_runtime_match_is_available() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200,
            json={"models": [{"name": "llama3:latest"}, {"name": "other:tag"}]},
        )

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
                runtime_name="llama3",
                capabilities=frozenset({Capability.CHAT}),
            ),
            ModelConfig(
                id="public-ambiguous",
                runtime_name="other",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    # second model: only one other:tag → also unambiguous available
    with TestClient(create_app(config, provider=provider), base_url="http://127.0.0.1") as client:
        body = client.get("/v1/models").json()
    by_id = {row["id"]: row["availability"] for row in body["data"]}
    assert by_id == {"public-chat": "available", "public-ambiguous": "available"}
    assert "llama3" not in body  # runtime names never leaked as model ids
