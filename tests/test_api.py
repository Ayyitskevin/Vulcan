"""Black-box coverage for Vulcan's stable HTTP contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
)
from vulcan.errors import (
    ConfigurationError,
    MissingCredentialError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    VulcanError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderTokenUsage,
    StreamDelta,
    StreamEnd,
)

PROMPT_SENTINEL = "prompt-must-not-escape-6f024f37"
REPLY_SENTINEL = "deterministic-reply-1cdbecf8"
RUNTIME_SENTINEL = "private-runtime-name-8675309"


class ResponseLike(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...

    def json(self) -> Any: ...


class RecordingProvider:
    """No-I/O provider used to prove validation happens before provider invocation."""

    provider_type: Literal["deterministic"] = "deterministic"

    def __init__(
        self,
        *,
        provider_id: str = "test-provider",
        result: ProviderChatResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.result = result or ProviderChatResult(content=REPLY_SENTINEL, finish_reason="stop")
        self.failure = failure
        self.calls: list[ProviderChatRequest] = []
        self.discover_calls = 0
        self.closed = False

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.result

    async def chat_stream(self, request: ProviderChatRequest):
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        yield StreamDelta(text=self.result.content)
        yield StreamEnd(finish_reason=self.result.finish_reason, usage=self.result.usage)

    async def discover_runtime(self):
        from vulcan.readiness import RuntimeProbe

        self.discover_calls += 1
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        self.closed = True


def _config(*, response_text: str = REPLY_SENTINEL) -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        providers={
            "test-provider": DeterministicProviderConfig(
                type="deterministic",
                response_text=response_text,
            )
        },
        models=(
            ModelConfig(
                id="public-chat",
                provider="test-provider",
                provider_model=RUNTIME_SENTINEL,
                capabilities=frozenset({Capability.CHAT}),
                description="Configured chat model",
                class_="code",
            ),
            ModelConfig(
                id="public-embed",
                provider="test-provider",
                provider_model="private-embedding-runtime",
                capabilities=frozenset({Capability.EMBEDDINGS}),
                description=None,
            ),
        ),
    )


def _valid_chat(*, model: str = "public-chat", stream: bool = False) -> dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
        "temperature": 0.25,
        "max_tokens": 17,
        "stream": stream,
    }


def _client(
    app: FastAPI,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    return TestClient(
        app,
        base_url="http://127.0.0.1",
        raise_server_exceptions=raise_server_exceptions,
    )


def _assert_request_id(response: ResponseLike, *, error_body: bool = False) -> str:
    request_id = response.headers["X-Request-ID"]
    assert str(UUID(request_id)) == request_id
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    if error_body:
        assert response.json()["request_id"] == request_id
    return request_id


def _expected_error(
    *,
    request_id: str,
    code: str,
    message: str,
    retryable: bool,
    details: dict[str, str | int | bool] | None = None,
    validation: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
            "details": details,
            "validation": validation,
        },
        "request_id": request_id,
    }


def test_healthz_reports_liveness_and_per_provider_readiness() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "vulcan",
        "api_version": "v1",
        "providers": [
            {"id": "test-provider", "type": "deterministic", "availability": "available"}
        ],
        "models_configured": 2,
    }
    _assert_request_id(response)


def test_models_are_exactly_configuration_driven_and_never_claim_loaded_state() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "discovery": {"source": "configuration"},
        "data": [
            {
                "id": "public-chat",
                "object": "model",
                "provider": "test-provider",
                "provider_type": "deterministic",
                "capabilities": ["chat"],
                "availability": "available",
                "description": "Configured chat model",
                "class": "code",
            },
            {
                "id": "public-embed",
                "object": "model",
                "provider": "test-provider",
                "provider_type": "deterministic",
                "capabilities": ["embeddings"],
                "availability": "available",
                "description": None,
                "class": None,
            },
        ],
    }
    assert RUNTIME_SENTINEL not in response.text
    assert "private-embedding-runtime" not in response.text
    _assert_request_id(response)


def test_capabilities_state_the_small_callable_contract_exactly() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "api_version": "v1",
        "model_discovery": "configuration",
        "callable_capabilities": ["chat", "embeddings"],
        "chat_completions": {
            "supported": True,
            "streaming": True,
            "message_roles": ["system", "user", "assistant"],
        },
        "embeddings": {
            "supported": True,
            "max_inputs": 64,
            "max_input_characters": 8192,
        },
    }
    _assert_request_id(response)


def test_deterministic_chat_response_is_stable_and_explicit() -> None:
    app = create_app(
        _config(),
        clock=lambda: 1_784_550_000.9,
        id_factory=lambda: "chatcmpl-fixed",
    )
    with _client(app) as client:
        response = client.post("/v1/chat/completions", json=_valid_chat())

    assert response.status_code == 200
    assert response.json() == {
        "id": "chatcmpl-fixed",
        "object": "chat.completion",
        "created": 1_784_550_000,
        "model": "public-chat",
        "provider": "test-provider",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": REPLY_SENTINEL},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }
    _assert_request_id(response)


def test_chat_maps_public_model_to_provider_model_and_preserves_provider_usage() -> None:
    provider = RecordingProvider(
        result=ProviderChatResult(
            content="mapped reply",
            finish_reason="length",
            usage=ProviderTokenUsage(prompt_tokens=8, completion_tokens=3),
        )
    )
    app = create_app(
        _config(),
        providers={"test-provider": provider},
        clock=lambda: 123.0,
        id_factory=lambda: "chatcmpl-mapped",
    )
    payload = {
        "model": "public-chat",
        "messages": [
            {"role": "system", "content": "Stay concise."},
            {"role": "user", "content": "Map me."},
        ],
        "temperature": 0.5,
        "max_tokens": 9,
    }

    with _client(app) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 200
    assert provider.calls == [
        ProviderChatRequest(
            provider_model=RUNTIME_SENTINEL,
            messages=(
                # These are provider-bound values, not public discovery values.
                ProviderMessage(role="system", content="Stay concise."),
                ProviderMessage(role="user", content="Map me."),
            ),
            temperature=0.5,
            max_tokens=9,
        )
    ]
    assert [(message.role, message.content) for message in provider.calls[0].messages] == [
        ("system", "Stay concise."),
        ("user", "Map me."),
    ]
    assert response.json()["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
    }
    assert provider.closed is True


def test_openapi_contains_only_the_documented_application_routes() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert {
        path: sorted(method for method in item if method in {"get", "post"})
        for path, item in schema["paths"].items()
    } == {
        "/healthz": ["get"],
        "/v1/models": ["get"],
        "/v1/models/{model_id}": ["get"],
        "/v1/capabilities": ["get"],
        "/v1/chat/completions": ["post"],
        "/v1/embeddings": ["post"],
        "/v1/usage": ["get"],
    }
    assert schema["info"]["license"] == {"name": "AGPL-3.0-only"}
    _assert_request_id(response)


def test_get_model_returns_configured_model_with_readiness() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/models/public-chat")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "public-chat"
    assert body["object"] == "model"
    assert body["provider"] == "test-provider"
    assert body["provider_type"] == "deterministic"
    assert body["capabilities"] == ["chat"]
    assert body["availability"] == "available"
    assert body["class"] == "code"
    assert RUNTIME_SENTINEL not in response.text
    _assert_request_id(response)


def test_get_model_unknown_id_is_model_not_found() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/models/does-not-exist")

    assert response.status_code == 404
    request_id = _assert_request_id(response, error_body=True)
    assert response.json()["error"]["code"] == "model_not_found"
    assert response.json()["error"]["details"] == {"model": "does-not-exist"}
    assert response.json()["request_id"] == request_id


def test_interactive_docs_are_disabled_without_removing_the_openapi_contract() -> None:
    with _client(create_app(_config())) as client:
        docs = client.get("/docs")
        redoc = client.get("/redoc")

    assert docs.status_code == 404
    assert redoc.status_code == 404
    assert "cdn.jsdelivr.net" not in docs.text
    assert "cdn.jsdelivr.net" not in redoc.text
    _assert_request_id(docs, error_body=True)
    _assert_request_id(redoc, error_body=True)


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "LOCALHOST:8140",
        "127.0.0.1",
        "127.99.42.7:65535",
        "[::1]",
        "[::1]:8140",
    ],
)
def test_loopback_host_headers_are_accepted(host: str) -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/healthz", headers={"host": host})

    assert response.status_code == 200
    _assert_request_id(response)


@pytest.mark.parametrize(
    "host",
    [
        "",
        "example.com",
        "localhost.example.com",
        "127.0.0.1.example.com",
        "2130706433",
        "0x7f000001",
        "::1",
        "[::1].example.com",
        "[::1]:0",
        "[::1]:65536",
        "127.0.0.1:not-a-port",
        " localhost",
    ],
)
def test_non_loopback_or_ambiguous_host_headers_are_rejected(host: str) -> None:
    provider = RecordingProvider()
    with _client(create_app(_config(), providers={"test-provider": provider})) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_valid_chat(),
            headers={"host": host},
        )

    assert response.status_code == 400
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code="invalid_host",
        message="The Host header must identify a loopback address.",
        retryable=False,
    )
    assert provider.calls == []
    assert PROMPT_SENTINEL not in response.text


def test_request_validation_uses_a_sanitized_stable_envelope() -> None:
    payload = _valid_chat()
    payload["unexpected"] = "invalid-value-must-not-escape-64f769b2"
    with _client(create_app(_config())) as client:
        response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code="invalid_request",
        message="The request does not match the Vulcan v1 contract.",
        retryable=False,
        validation=[{"path": "body.unexpected", "reason": "extra_forbidden"}],
    )
    assert PROMPT_SENTINEL not in response.text
    assert "invalid-value-must-not-escape-64f769b2" not in response.text


def test_unknown_model_is_normalized_before_provider_invocation() -> None:
    provider = RecordingProvider()
    with _client(create_app(_config(), providers={"test-provider": provider})) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_valid_chat(model="missing-model"),
        )

    assert response.status_code == 404
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code="model_not_found",
        message="The selected model is not configured.",
        retryable=False,
        details={"model": "missing-model"},
    )
    assert provider.calls == []
    assert PROMPT_SENTINEL not in response.text


def test_unsupported_capability_is_normalized_before_provider_invocation() -> None:
    model, capability = "public-embed", "chat"
    provider = RecordingProvider()
    with _client(create_app(_config(), providers={"test-provider": provider})) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_valid_chat(model=model),
        )

    assert response.status_code == 422
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code="unsupported_capability",
        message="The selected model or endpoint does not support this capability.",
        retryable=False,
        details={"capability": capability, "model": model},
    )
    assert provider.calls == []
    assert PROMPT_SENTINEL not in response.text


@pytest.mark.parametrize(
    ("failure", "status", "code", "message", "retryable", "details"),
    [
        pytest.param(
            ProviderUnavailableError(),
            503,
            "provider_unavailable",
            "The selected provider is unavailable.",
            True,
            {"provider": "test-provider"},
            id="provider-unavailable",
        ),
        pytest.param(
            ModelUnavailableError(),
            503,
            "model_unavailable",
            "The configured model is unavailable in the selected provider.",
            False,
            {"provider": "test-provider"},
            id="model-unavailable",
        ),
        pytest.param(
            MissingCredentialError("VULCAN_TEST_API_KEY"),
            503,
            "missing_credential",
            "The selected provider's credential environment variable is not usable.",
            False,
            {"api_key_env": "VULCAN_TEST_API_KEY", "provider": "test-provider"},
            id="missing-credential",
        ),
        pytest.param(
            ProviderAuthError(),
            502,
            "provider_auth_failed",
            "The selected provider rejected the configured credential.",
            False,
            {"provider": "test-provider"},
            id="provider-auth-failed",
        ),
        pytest.param(
            ProviderRateLimitError(),
            429,
            "provider_rate_limited",
            "The selected provider rate limited this request.",
            True,
            {"provider": "test-provider"},
            id="provider-rate-limited",
        ),
        pytest.param(
            ProviderTimeoutError(),
            504,
            "provider_timeout",
            "The selected provider timed out.",
            True,
            {"provider": "test-provider"},
            id="provider-timeout",
        ),
        pytest.param(
            ProviderError(),
            502,
            "provider_error",
            "The selected provider rejected or failed the request.",
            True,
            {"provider": "test-provider"},
            id="provider-error",
        ),
        pytest.param(
            ProviderError(retryable=False),
            502,
            "provider_error",
            "The selected provider rejected or failed the request.",
            False,
            {"provider": "test-provider"},
            id="provider-rejection",
        ),
        pytest.param(
            ProviderProtocolError(),
            502,
            "provider_protocol_error",
            "The selected provider returned an invalid response.",
            False,
            {"provider": "test-provider"},
            id="provider-protocol",
        ),
        pytest.param(
            ConfigurationError(),
            500,
            "configuration_error",
            "Vulcan is not configured to complete this request.",
            False,
            {"provider": "test-provider"},
            id="provider-misconfigured",
        ),
    ],
)
def test_provider_failures_share_one_normalized_error_contract(
    failure: VulcanError,
    status: int,
    code: str,
    message: str,
    retryable: bool,
    details: dict[str, str | int | bool],
) -> None:
    provider = RecordingProvider(failure=failure)
    with _client(create_app(_config(), providers={"test-provider": provider})) as client:
        response = client.post("/v1/chat/completions", json=_valid_chat())

    assert response.status_code == status
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code=code,
        message=message,
        retryable=retryable,
        details=details,
    )
    assert len(provider.calls) == 1
    assert PROMPT_SENTINEL not in response.text


def test_unknown_route_uses_the_same_request_scoped_error_boundary() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/not-a-real-route")

    assert response.status_code == 404
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code="not_found",
        message="The requested endpoint does not exist.",
        retryable=False,
    )


def test_unexpected_provider_failure_is_sanitized_at_the_http_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    exception_sentinel = "unexpected-provider-detail-must-not-escape"
    provider = RecordingProvider(failure=RuntimeError(exception_sentinel))
    with _client(
        create_app(_config(), providers={"test-provider": provider}),
        raise_server_exceptions=False,
    ) as client:
        response = client.post("/v1/chat/completions", json=_valid_chat())

    assert response.status_code == 500
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code="internal_error",
        message="Vulcan could not complete the request.",
        retryable=False,
    )
    assert exception_sentinel not in response.text
    assert PROMPT_SENTINEL not in response.text
    assert exception_sentinel not in caplog.text
    assert PROMPT_SENTINEL not in caplog.text
    # The safe diagnostic survives: the class name is logged (under a key the
    # formatter does not reserve or redact), never the exception message.
    internal_records = [record for record in caplog.records if record.msg == "internal_error"]
    assert internal_records
    metadata = internal_records[0].metadata  # type: ignore[attr-defined]
    assert metadata["exception_type"] == "RuntimeError"
    from vulcan.observability import redact_metadata

    assert redact_metadata(metadata)["exception_type"] == "RuntimeError"
