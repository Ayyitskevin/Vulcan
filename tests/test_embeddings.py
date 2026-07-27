"""Embeddings coverage: adapters, gateway routing, config policy, HTTP contract.

Every upstream exchange goes through ``httpx.MockTransport``; no test contacts
a real API. Credentials and inputs are synthetic sentinels so the leak
assertions are meaningful.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from vulcan.errors import (
    MissingCredentialError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from vulcan.gateway import Gateway
from vulcan.providers.base import (
    ProviderEmbeddingRequest,
    ProviderEmbeddingResult,
    ProviderEmbeddingUsage,
)
from vulcan.providers.deterministic import DETERMINISTIC_EMBEDDING, DeterministicProvider
from vulcan.providers.ollama import OllamaProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider
from vulcan.readiness import RuntimeProbe
from vulcan.registry import ModelRegistry
from vulcan.schemas import EmbeddingsRequest

OPENAI_KEY_ENV = "VULCAN_EMBED_TEST_OPENAI_KEY"
OPENAI_KEY_SENTINEL = "sk-embed-openai-secret-71bd"
BODY_SENTINEL = "embed-upstream-body-must-not-escape-33fa"
INPUT_SENTINEL = "embed-input-must-not-escape-84c1"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, OPENAI_KEY_SENTINEL)


def _embed_request(*inputs: str) -> ProviderEmbeddingRequest:
    return ProviderEmbeddingRequest(
        provider_model="native-embed",
        inputs=inputs or (INPUT_SENTINEL,),
    )


def _run(provider: Any, request: ProviderEmbeddingRequest | None = None) -> ProviderEmbeddingResult:
    async def go() -> ProviderEmbeddingResult:
        try:
            return await provider.embed(request or _embed_request())
        finally:
            await provider.aclose()

    return asyncio.run(go())


# ── OpenAI-compatible adapter ────────────────────────────────────────────────


def _compat_provider(handler: Any) -> OpenAICompatibleProvider:
    config = OpenAICompatibleProviderConfig(
        type="openai_compatible",
        base_url="https://api.compat-mock.example/v1",
        api_key_env=OPENAI_KEY_ENV,
        timeout_seconds=1.0,
    )
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return OpenAICompatibleProvider("compat", config, client=client)


def test_compat_embed_posts_batch_with_bearer_auth_and_parses_vectors() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            },
        )

    result = _run(_compat_provider(handler), _embed_request("first", "second"))

    assert str(captured[0].url) == "https://api.compat-mock.example/v1/embeddings"
    assert captured[0].headers["Authorization"] == f"Bearer {OPENAI_KEY_SENTINEL}"
    assert json.loads(captured[0].content) == {
        "model": "native-embed",
        "input": ["first", "second"],
    }
    assert result == ProviderEmbeddingResult(
        vectors=((0.1, 0.2), (0.3, 0.4)),
        usage=ProviderEmbeddingUsage(prompt_tokens=9, total_tokens=9),
    )


def test_compat_embed_reorders_records_by_index() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 2, "embedding": [3.0]},
                    {"index": 0, "embedding": [1.0]},
                    {"index": 1, "embedding": [2.0]},
                ]
            },
        )

    result = _run(_compat_provider(handler), _embed_request("a", "b", "c"))

    assert result.vectors == ((1.0,), (2.0,), (3.0,))


@pytest.mark.parametrize(
    "data",
    [
        # Indices that do not form 0..n-1 would silently misalign inputs.
        [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [2.0]}],
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
        [{"index": 0, "embedding": [1.0]}, {"embedding": [2.0]}],
    ],
)
def test_compat_embed_rejects_inconsistent_indices(data: list[dict[str, Any]]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": data})

    with pytest.raises(ProviderProtocolError):
        _run(_compat_provider(handler), _embed_request("a", "b"))


def test_compat_embed_accepts_records_without_indices_in_order() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]})

    assert _run(_compat_provider(handler), _embed_request("a", "b")).vectors == ((1.0,), (2.0,))


@pytest.mark.parametrize(
    "body",
    [
        {"data": []},
        {},
        {"data": [{"embedding": "not-a-list"}]},
        {"data": [{"embedding": [1.0, "x"]}]},
        {"data": [{"embedding": [1.0], "index": "0"}]},
        {"data": [{"embedding": [1.0]}], "usage": {"prompt_tokens": -1}},
    ],
)
def test_compat_embed_invalid_shapes_are_protocol_errors(body: dict[str, Any]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        _run(_compat_provider(handler))


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_compat_embed_rejects_non_finite_floats(literal: str) -> None:
    """Python's JSON parser accepts these; a vector must never carry one."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=f'{{"data": [{{"index": 0, "embedding": [1.0, {literal}]}}]}}'.encode(),
            headers={"content-type": "application/json"},
        )

    with pytest.raises(ProviderProtocolError):
        _run(_compat_provider(handler))


def test_compat_embed_usage_defaults_total_to_prompt_when_absent() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0]}], "usage": {"prompt_tokens": 4}},
        )

    assert _run(_compat_provider(handler)).usage == ProviderEmbeddingUsage(
        prompt_tokens=4, total_tokens=4
    )


def test_compat_embed_without_usage_never_invents_counts() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})

    assert _run(_compat_provider(handler)).usage is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderAuthError),
        (404, ModelUnavailableError),
        (429, ProviderRateLimitError),
        (503, ProviderUnavailableError),
    ],
)
def test_compat_embed_status_normalization(status: int, expected: type[Exception]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": BODY_SENTINEL}})

    with pytest.raises(expected) as caught:
        _run(_compat_provider(handler))

    assert BODY_SENTINEL not in f"{caught.value!r} {getattr(caught.value, 'details', None)!r}"


def test_compat_embed_timeout_and_transport_failures_are_normalized() -> None:
    async def timed_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    with pytest.raises(ProviderTimeoutError):
        _run(_compat_provider(timed_out))

    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        _run(_compat_provider(refused))


def test_compat_embed_missing_credential_fails_before_any_http_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call may happen without a credential")

    with pytest.raises(MissingCredentialError) as caught:
        _run(_compat_provider(handler))

    assert caught.value.details == {"api_key_env": OPENAI_KEY_ENV}


# ── Ollama adapter ───────────────────────────────────────────────────────────


def _ollama_provider(handler: Any) -> OllamaProvider:
    config = OllamaProviderConfig(
        type="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
    )
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return OllamaProvider("local-ollama", config, client=client)


def test_ollama_embed_posts_api_embed_and_parses_vectors() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={"embeddings": [[0.5, 0.25], [0.75, 1.0]], "prompt_eval_count": 6},
        )

    result = _run(_ollama_provider(handler), _embed_request("one", "two"))

    assert str(captured[0].url) == "http://127.0.0.1:11434/api/embed"
    assert json.loads(captured[0].content) == {"model": "native-embed", "input": ["one", "two"]}
    assert result.vectors == ((0.5, 0.25), (0.75, 1.0))
    # Embeddings have no completion tokens, so total mirrors prompt.
    assert result.usage == ProviderEmbeddingUsage(prompt_tokens=6, total_tokens=6)


def test_ollama_embed_missing_model_maps_to_model_unavailable() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'native-embed' not found"})

    with pytest.raises(ModelUnavailableError):
        _run(_ollama_provider(handler))


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": []},
        {},
        {"embeddings": [[1.0], "not-a-vector"]},
        {"embeddings": [[1.0]], "prompt_eval_count": -1},
    ],
)
def test_ollama_embed_invalid_shapes_are_protocol_errors(body: dict[str, Any]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        _run(_ollama_provider(handler))


def test_ollama_embed_rejects_non_finite_floats() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"embeddings": [[1.0, NaN]]}',
            headers={"content-type": "application/json"},
        )

    with pytest.raises(ProviderProtocolError):
        _run(_ollama_provider(handler))


# ── Deterministic adapter ────────────────────────────────────────────────────


def test_deterministic_embed_returns_one_fixed_vector_per_input() -> None:
    provider = DeterministicProvider(
        "det", DeterministicProviderConfig(type="deterministic", response_text="unused")
    )

    result = _run(provider, _embed_request("a", "b", "c"))

    assert result.vectors == (DETERMINISTIC_EMBEDDING,) * 3
    assert result.usage is None


# ── Configuration policy ─────────────────────────────────────────────────────


def _document(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": 2,
        "providers": {
            "det": {"type": "deterministic", "response_text": "canned"},
            "anthropic": {
                "type": "anthropic",
                "api_key_env": "VULCAN_EMBED_TEST_ANTHROPIC_KEY",
                "timeout_seconds": 1.0,
            },
        },
        "models": [
            {
                "id": "chat-alias",
                "provider": "det",
                "provider_model": "native-chat",
                "capabilities": ["chat"],
            }
        ],
    }
    document.update(overrides)
    return document


def _error_types(error: ValidationError) -> set[str]:
    return {item["type"] for item in error.errors(include_url=False, include_input=False)}


def test_config_rejects_embeddings_capability_on_anthropic_providers() -> None:
    document = _document()
    document["models"].append(
        {
            "id": "embed-alias",
            "provider": "anthropic",
            "provider_model": "native-embed",
            "capabilities": ["embeddings"],
        }
    )

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(document)

    assert "anthropic_embeddings_unsupported" in _error_types(raised.value)


def test_config_rejects_mixed_chat_and_embeddings_on_anthropic_providers() -> None:
    document = _document()
    document["models"].append(
        {
            "id": "embed-alias",
            "provider": "anthropic",
            "provider_model": "native-embed",
            "capabilities": ["chat", "embeddings"],
        }
    )

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(document)

    assert "anthropic_embeddings_unsupported" in _error_types(raised.value)


def test_config_allows_chat_only_models_on_anthropic_providers() -> None:
    document = _document()
    document["models"].append(
        {
            "id": "anthropic-chat",
            "provider": "anthropic",
            "provider_model": "native-chat",
            "capabilities": ["chat"],
        }
    )

    config = GatewayConfig.model_validate(document)

    assert config.models[1].provider == "anthropic"


# ── Request schema bounds ────────────────────────────────────────────────────


def test_embeddings_request_accepts_one_string_or_a_list() -> None:
    single = EmbeddingsRequest.model_validate({"model": "alias", "input": "solo"})
    many = EmbeddingsRequest.model_validate({"model": "alias", "input": ["a", "b"]})

    assert single.inputs == ("solo",)
    assert many.inputs == ("a", "b")


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "alias", "input": []},
        {"model": "alias", "input": ["  "]},
        {"model": "alias", "input": ""},
        {"model": "alias", "input": ["ok", ""]},
        {"model": "alias", "input": ["x"] * 65},
        {"model": "alias", "input": "y" * 8193},
        {"model": "alias", "input": ["z" * 8192] * 9},
        {"model": "alias", "input": 5},
        {"model": "alias", "input": [1, 2]},
        {"model": "alias"},
        {"model": "alias", "input": "ok", "unexpected": "field"},
    ],
)
def test_embeddings_request_rejects_out_of_contract_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingsRequest.model_validate(payload)


def test_embeddings_request_accepts_exact_boundaries() -> None:
    at_limit = EmbeddingsRequest.model_validate({"model": "alias", "input": ["a"] * 64})
    longest = EmbeddingsRequest.model_validate({"model": "alias", "input": "b" * 8192})

    assert len(at_limit.inputs) == 64
    assert len(longest.inputs[0]) == 8192


# ── Gateway routing ──────────────────────────────────────────────────────────


class _StubEmbedProvider:
    provider_type: Literal["deterministic"] = "deterministic"

    def __init__(
        self,
        provider_id: str = "stub",
        result: ProviderEmbeddingResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.result = result or ProviderEmbeddingResult(vectors=((1.0, 2.0),), usage=None)
        self.failure = failure
        self.calls: list[ProviderEmbeddingRequest] = []

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.result

    async def chat(self, request: Any) -> Any:  # pragma: no cover - unused here
        raise AssertionError("chat must not be called for an embeddings request")

    async def discover_runtime(self) -> RuntimeProbe:
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        return None


def _registry(capabilities: frozenset[Capability], provider_id: str = "stub") -> ModelRegistry:
    return ModelRegistry(
        (
            ModelConfig(
                id="embed-alias",
                provider=provider_id,
                provider_model="native-embed",
                capabilities=capabilities,
            ),
        )
    )


def _embed(gateway: Gateway, *inputs: str) -> Any:
    request = EmbeddingsRequest.model_validate(
        {"model": "embed-alias", "input": list(inputs) or [INPUT_SENTINEL]}
    )
    return asyncio.run(gateway.embed(request, request_id="req-embed"))


def test_gateway_embed_maps_alias_to_provider_and_preserves_order() -> None:
    provider = _StubEmbedProvider(
        result=ProviderEmbeddingResult(
            vectors=((1.0,), (2.0,), (3.0,)),
            usage=ProviderEmbeddingUsage(prompt_tokens=12, total_tokens=12),
        )
    )
    gateway = Gateway(_registry(frozenset({Capability.EMBEDDINGS})), {"stub": provider})

    response = _embed(gateway, "a", "b", "c")

    assert provider.calls == [
        ProviderEmbeddingRequest(provider_model="native-embed", inputs=("a", "b", "c"))
    ]
    assert response.model_dump(mode="json") == {
        "object": "list",
        "model": "embed-alias",
        "provider": "stub",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [1.0]},
            {"object": "embedding", "index": 1, "embedding": [2.0]},
            {"object": "embedding", "index": 2, "embedding": [3.0]},
        ],
        "usage": {"prompt_tokens": 12, "total_tokens": 12},
    }


def test_gateway_embed_requires_the_embeddings_capability() -> None:
    provider = _StubEmbedProvider()
    gateway = Gateway(_registry(frozenset({Capability.CHAT})), {"stub": provider})

    with pytest.raises(UnsupportedCapabilityError) as caught:
        _embed(gateway)

    assert caught.value.details == {"capability": "embeddings", "model": "embed-alias"}
    assert provider.calls == []


def test_gateway_embed_rejects_a_vector_count_mismatch() -> None:
    provider = _StubEmbedProvider(
        result=ProviderEmbeddingResult(vectors=((1.0,),), usage=None)  # one vector, two inputs
    )
    gateway = Gateway(_registry(frozenset({Capability.EMBEDDINGS})), {"stub": provider})

    with pytest.raises(ProviderProtocolError):
        _embed(gateway, "a", "b")


def test_gateway_embed_annotates_failures_with_the_selected_provider() -> None:
    provider = _StubEmbedProvider("hosted", failure=ProviderRateLimitError())
    gateway = Gateway(_registry(frozenset({Capability.EMBEDDINGS}), "hosted"), {"hosted": provider})

    with pytest.raises(ProviderRateLimitError) as caught:
        _embed(gateway)

    assert caught.value.details == {"provider": "hosted"}


def test_gateway_embed_never_falls_back_to_another_provider() -> None:
    failing = _StubEmbedProvider("failing", failure=ProviderUnavailableError())
    healthy = _StubEmbedProvider("healthy")
    registry = ModelRegistry(
        (
            ModelConfig(
                id="embed-alias",
                provider="failing",
                provider_model="native-embed",
                capabilities=frozenset({Capability.EMBEDDINGS}),
            ),
            ModelConfig(
                id="other-embed",
                provider="healthy",
                provider_model="other-embed",
                capabilities=frozenset({Capability.EMBEDDINGS}),
            ),
        )
    )
    gateway = Gateway(registry, {"failing": failing, "healthy": healthy})

    with pytest.raises(ProviderUnavailableError):
        _embed(gateway)

    assert len(failing.calls) == 1
    assert healthy.calls == []


def test_gateway_embed_logs_counts_without_input_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _StubEmbedProvider(
        result=ProviderEmbeddingResult(vectors=((1.0, 2.0, 3.0),), usage=None)
    )
    gateway = Gateway(_registry(frozenset({Capability.EMBEDDINGS})), {"stub": provider})

    with caplog.at_level("INFO", logger="vulcan.gateway"):
        _embed(gateway, INPUT_SENTINEL)

    completed = next(record for record in caplog.records if record.msg == "embeddings_completed")
    metadata = completed.metadata  # type: ignore[attr-defined]
    assert metadata["input_count"] == 1
    assert metadata["input_chars"] == len(INPUT_SENTINEL)
    assert metadata["vector_count"] == 1
    assert metadata["dimensions"] == 3
    assert metadata["provider"] == "stub"
    assert metadata["request_id"] == "req-embed"
    assert INPUT_SENTINEL not in caplog.text


# ── HTTP contract ────────────────────────────────────────────────────────────


def _app_config() -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        providers={
            "det": DeterministicProviderConfig(type="deterministic", response_text="canned")
        },
        models=(
            ModelConfig(
                id="embed-alias",
                provider="det",
                provider_model="native-embed",
                capabilities=frozenset({Capability.CHAT, Capability.EMBEDDINGS}),
            ),
            ModelConfig(
                id="chat-only",
                provider="det",
                provider_model="native-chat",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )


def test_http_embeddings_returns_one_record_per_input() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed-alias", "input": [INPUT_SENTINEL, "second"]},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "embed-alias"
    assert body["provider"] == "det"
    assert [record["index"] for record in body["data"]] == [0, 1]
    assert all(record["object"] == "embedding" for record in body["data"])
    assert body["data"][0]["embedding"] == list(DETERMINISTIC_EMBEDDING)
    assert body["usage"] is None
    assert "native-embed" not in response.text


def test_http_embeddings_accepts_a_single_string_input() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        response = client.post("/v1/embeddings", json={"model": "embed-alias", "input": "solo"})

    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_http_embeddings_rejects_a_chat_only_alias() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        response = client.post("/v1/embeddings", json={"model": "chat-only", "input": "x"})

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "unsupported_capability",
        "message": "The selected model or endpoint does not support this capability.",
        "retryable": False,
        "details": {"capability": "embeddings", "model": "chat-only"},
        "validation": None,
    }


def test_http_embeddings_unknown_alias_is_model_not_found() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        response = client.post("/v1/embeddings", json={"model": "missing", "input": "x"})

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_http_embeddings_validation_errors_are_sanitized() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed-alias", "input": [INPUT_SENTINEL], "unexpected": "value"},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_request"
    assert body["error"]["validation"] == [{"path": "body.unexpected", "reason": "extra_forbidden"}]
    assert INPUT_SENTINEL not in response.text


def test_http_embeddings_rejects_non_loopback_host_before_routing() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed-alias", "input": "x"},
            headers={"host": "example.com"},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_host"


def test_http_embeddings_never_leaks_inputs_keys_or_upstream_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert INPUT_SENTINEL in request.content.decode("utf-8")
        return httpx.Response(429, json={"error": {"message": BODY_SENTINEL}})

    config = GatewayConfig(
        schema_version=2,
        providers={
            "compat": OpenAICompatibleProviderConfig(
                type="openai_compatible",
                base_url="https://api.compat-mock.example/v1",
                api_key_env=OPENAI_KEY_ENV,
                timeout_seconds=1.0,
            )
        },
        models=(
            ModelConfig(
                id="embed-alias",
                provider="compat",
                provider_model="native-embed",
                capabilities=frozenset({Capability.EMBEDDINGS}),
            ),
            ModelConfig(
                id="chat-alias",
                provider="compat",
                provider_model="native-chat",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    with (
        caplog.at_level("DEBUG"),
        TestClient(
            create_app(config, providers={"compat": _compat_provider(handler)}),
            base_url="http://127.0.0.1",
        ) as client,
    ):
        response = client.post(
            "/v1/embeddings",
            json={"model": "embed-alias", "input": INPUT_SENTINEL},
        )

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "provider_rate_limited"
    assert response.json()["error"]["details"] == {"provider": "compat"}
    for sentinel in (INPUT_SENTINEL, OPENAI_KEY_SENTINEL, BODY_SENTINEL, "Bearer ", "native-embed"):
        assert sentinel not in response.text
        assert sentinel not in caplog.text
