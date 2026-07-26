"""End-to-end multi-provider routing through the real app and adapters.

Hosted HTTP is mocked at the transport seam only; requests flow through the
shipped config → factory → gateway → adapter path.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import GatewayConfig
from vulcan.providers.anthropic import AnthropicProvider
from vulcan.providers.base import Provider
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.factory import build_providers
from vulcan.providers.ollama import OllamaProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider

OPENAI_KEY_ENV = "VULCAN_ROUTE_TEST_OPENAI_KEY"
ANTHROPIC_KEY_ENV = "VULCAN_ROUTE_TEST_ANTHROPIC_KEY"
OPENAI_KEY_SENTINEL = "sk-openai-route-secret-11aa"
ANTHROPIC_KEY_SENTINEL = "sk-ant-route-secret-22bb"
UPSTREAM_BODY_SENTINEL = "upstream-error-body-must-not-escape-9d41"
PROMPT_SENTINEL = "routing-prompt-must-not-escape-7c55"


def _document() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "providers": {
            "local-ollama": {
                "type": "ollama",
                "base_url": "http://127.0.0.1:11434",
                "timeout_seconds": 1.0,
            },
            "openai": {
                "type": "openai_compatible",
                "base_url": "https://api.openai-mock.example/v1",
                "api_key_env": OPENAI_KEY_ENV,
                "timeout_seconds": 1.0,
            },
            "anthropic": {
                "type": "anthropic",
                "base_url": "https://api.anthropic-mock.example",
                "api_key_env": ANTHROPIC_KEY_ENV,
                "timeout_seconds": 1.0,
                "default_max_tokens": 512,
            },
            "det": {
                "type": "deterministic",
                "response_text": "canned",
            },
        },
        "models": [
            {
                "id": "local-chat",
                "provider": "local-ollama",
                "provider_model": "runtime-chat",
                "capabilities": ["chat"],
            },
            {
                "id": "openai-chat",
                "provider": "openai",
                "provider_model": "openai-native",
                "capabilities": ["chat"],
            },
            {
                "id": "anthropic-chat",
                "provider": "anthropic",
                "provider_model": "anthropic-native",
                "capabilities": ["chat"],
            },
            {
                "id": "canned-chat",
                "provider": "det",
                "provider_model": "canned",
                "capabilities": ["chat"],
            },
        ],
    }


def _mocked_providers(
    config: GatewayConfig,
    handlers: dict[str, Any],
) -> dict[str, Provider]:
    """Build the real adapters with per-provider mock transports."""

    providers: dict[str, Provider] = {}
    for provider_id, provider_config in config.providers.items():
        if provider_config.type == "deterministic":
            providers[provider_id] = DeterministicProvider(provider_id, provider_config)
            continue
        client = httpx.AsyncClient(
            base_url=provider_config.base_url,
            transport=httpx.MockTransport(handlers[provider_id]),
            trust_env=False,
        )
        if provider_config.type == "ollama":
            providers[provider_id] = OllamaProvider(provider_id, provider_config, client=client)
        elif provider_config.type == "anthropic":
            providers[provider_id] = AnthropicProvider(provider_id, provider_config, client=client)
        else:
            providers[provider_id] = OpenAICompatibleProvider(
                provider_id, provider_config, client=client
            )
    return providers


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, OPENAI_KEY_SENTINEL)
    monkeypatch.setenv(ANTHROPIC_KEY_ENV, ANTHROPIC_KEY_SENTINEL)


def _chat(client: TestClient, model: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": PROMPT_SENTINEL}]},
    )


def test_build_providers_constructs_every_configured_instance() -> None:
    config = GatewayConfig.model_validate(_document())

    providers = build_providers(config)

    assert set(providers) == {"local-ollama", "openai", "anthropic", "det"}
    assert type(providers["local-ollama"]) is OllamaProvider
    assert type(providers["openai"]) is OpenAICompatibleProvider
    assert type(providers["anthropic"]) is AnthropicProvider
    assert type(providers["det"]) is DeterministicProvider
    assert all(providers[key].provider_id == key for key in providers)


def test_each_alias_reaches_exactly_its_own_upstream() -> None:
    calls: dict[str, list[str]] = {"local-ollama": [], "openai": [], "anthropic": []}

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        calls["local-ollama"].append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "runtime-chat"}]})
        assert json.loads(request.content)["model"] == "runtime-chat"
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "from ollama"}, "done": True},
        )

    async def openai_handler(request: httpx.Request) -> httpx.Response:
        calls["openai"].append(request.url.path)
        assert request.headers["Authorization"] == f"Bearer {OPENAI_KEY_SENTINEL}"
        assert json.loads(request.content)["model"] == "openai-native"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "from openai"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async def anthropic_handler(request: httpx.Request) -> httpx.Response:
        calls["anthropic"].append(request.url.path)
        assert request.headers["x-api-key"] == ANTHROPIC_KEY_SENTINEL
        assert json.loads(request.content)["model"] == "anthropic-native"
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": "from anthropic"}],
                "stop_reason": "end_turn",
            },
        )

    config = GatewayConfig.model_validate(_document())
    providers = _mocked_providers(
        config,
        {
            "local-ollama": ollama_handler,
            "openai": openai_handler,
            "anthropic": anthropic_handler,
        },
    )
    with TestClient(create_app(config, providers=providers), base_url="http://127.0.0.1") as client:
        openai_reply = _chat(client, "openai-chat")
        anthropic_reply = _chat(client, "anthropic-chat")
        local_reply = _chat(client, "local-chat")
        canned_reply = _chat(client, "canned-chat")

    assert openai_reply.status_code == 200
    assert openai_reply.json()["provider"] == "openai"
    assert openai_reply.json()["choices"][0]["message"]["content"] == "from openai"

    assert anthropic_reply.status_code == 200
    assert anthropic_reply.json()["provider"] == "anthropic"
    assert anthropic_reply.json()["choices"][0]["message"]["content"] == "from anthropic"

    assert local_reply.status_code == 200
    assert local_reply.json()["provider"] == "local-ollama"
    assert local_reply.json()["choices"][0]["message"]["content"] == "from ollama"

    assert canned_reply.status_code == 200
    assert canned_reply.json()["provider"] == "det"

    # Exactly one chat call per hosted provider; no cross-provider traffic.
    assert calls["openai"] == ["/v1/chat/completions"]
    assert calls["anthropic"] == ["/v1/messages"]
    assert calls["local-ollama"].count("/api/chat") == 1


def test_hosted_failure_never_falls_back_to_another_provider() -> None:
    calls: dict[str, int] = {"openai": 0, "anthropic": 0, "local-ollama": 0}

    async def failing_openai(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            calls["openai"] += 1
        return httpx.Response(429, json={"error": {"message": UPSTREAM_BODY_SENTINEL}})

    async def anthropic_handler(request: httpx.Request) -> httpx.Response:
        calls["anthropic"] += 1
        return httpx.Response(200, json={"role": "assistant", "content": []})

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            calls["local-ollama"] += 1
        return httpx.Response(200, json={"models": []})

    config = GatewayConfig.model_validate(_document())
    providers = _mocked_providers(
        config,
        {
            "local-ollama": ollama_handler,
            "openai": failing_openai,
            "anthropic": anthropic_handler,
        },
    )
    with TestClient(create_app(config, providers=providers), base_url="http://127.0.0.1") as client:
        response = _chat(client, "openai-chat")

    assert response.status_code == 429
    body = response.json()
    assert body["error"]["code"] == "provider_rate_limited"
    assert body["error"]["retryable"] is True
    assert body["error"]["details"] == {"provider": "openai"}
    assert calls == {"openai": 1, "anthropic": 0, "local-ollama": 0}
    assert UPSTREAM_BODY_SENTINEL not in response.text
    assert PROMPT_SENTINEL not in response.text


def test_missing_credential_fails_only_that_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)

    async def openai_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call may happen without a credential")

    async def anthropic_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": "anthropic still works"}],
                "stop_reason": "end_turn",
            },
        )

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    config = GatewayConfig.model_validate(_document())
    providers = _mocked_providers(
        config,
        {
            "local-ollama": ollama_handler,
            "openai": openai_handler,
            "anthropic": anthropic_handler,
        },
    )
    with TestClient(create_app(config, providers=providers), base_url="http://127.0.0.1") as client:
        failing = _chat(client, "openai-chat")
        working = _chat(client, "anthropic-chat")

    assert failing.status_code == 503
    assert failing.json()["error"]["code"] == "missing_credential"
    assert failing.json()["error"]["details"] == {
        "api_key_env": OPENAI_KEY_ENV,
        "provider": "openai",
    }
    assert working.status_code == 200
    assert working.json()["choices"][0]["message"]["content"] == "anthropic still works"


def test_health_and_models_report_hosted_providers_as_unchecked_without_probes() -> None:
    upstream_hits: list[str] = []

    async def hosted_guard(request: httpx.Request) -> httpx.Response:
        upstream_hits.append(str(request.url))
        raise AssertionError("hosted providers must never be probed for readiness")

    async def ollama_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(200, json={"models": [{"name": "runtime-chat"}]})

    config = GatewayConfig.model_validate(_document())
    providers = _mocked_providers(
        config,
        {
            "local-ollama": ollama_handler,
            "openai": hosted_guard,
            "anthropic": hosted_guard,
        },
    )
    with TestClient(create_app(config, providers=providers), base_url="http://127.0.0.1") as client:
        health = client.get("/healthz").json()
        models = client.get("/v1/models").json()

    assert upstream_hits == []
    by_id = {entry["id"]: entry for entry in health["providers"]}
    assert by_id["local-ollama"]["availability"] == "available"
    assert by_id["openai"] == {
        "id": "openai",
        "type": "openai_compatible",
        "availability": "unchecked",
    }
    assert by_id["anthropic"] == {
        "id": "anthropic",
        "type": "anthropic",
        "availability": "unchecked",
    }
    assert by_id["det"]["availability"] == "available"

    by_model = {row["id"]: row for row in models["data"]}
    assert by_model["local-chat"]["availability"] == "available"
    assert by_model["openai-chat"]["availability"] == "unchecked"
    assert by_model["anthropic-chat"]["availability"] == "unchecked"
    assert by_model["canned-chat"]["availability"] == "available"


def test_hosted_provider_failure_logs_and_errors_stay_content_safe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def failing_openai(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": UPSTREAM_BODY_SENTINEL}})

    async def quiet(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    config = GatewayConfig.model_validate(_document())
    providers = _mocked_providers(
        config,
        {"local-ollama": quiet, "openai": failing_openai, "anthropic": quiet},
    )
    with (
        caplog.at_level("DEBUG"),
        TestClient(create_app(config, providers=providers), base_url="http://127.0.0.1") as client,
    ):
        response = _chat(client, "openai-chat")

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "provider_auth_failed"
    for sentinel in (
        UPSTREAM_BODY_SENTINEL,
        PROMPT_SENTINEL,
        OPENAI_KEY_SENTINEL,
        ANTHROPIC_KEY_SENTINEL,
        "Bearer ",
    ):
        assert sentinel not in response.text
        assert sentinel not in caplog.text
