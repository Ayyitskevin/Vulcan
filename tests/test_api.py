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
    ModelUnavailableError,
    ProviderError,
    ProviderProtocolError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    VulcanError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderTokenUsage,
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

    kind: Literal["deterministic"] = "deterministic"

    def __init__(
        self,
        *,
        result: ProviderChatResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.result = result or ProviderChatResult(content=REPLY_SENTINEL, finish_reason="stop")
        self.failure = failure
        self.calls: list[ProviderChatRequest] = []
        self.closed = False

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.result

    async def aclose(self) -> None:
        self.closed = True


def _config(*, response_text: str = REPLY_SENTINEL) -> GatewayConfig:
    return GatewayConfig(
        schema_version=1,
        provider=DeterministicProviderConfig(
            kind="deterministic",
            response_text=response_text,
        ),
        models=(
            ModelConfig(
                id="public-chat",
                runtime_name=RUNTIME_SENTINEL,
                capabilities=frozenset({Capability.CHAT}),
                description="Configured chat model",
            ),
            ModelConfig(
                id="public-embed",
                runtime_name="private-embedding-runtime",
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


def test_healthz_reports_liveness_and_deterministic_provider_readiness() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "vulcan",
        "api_version": "v1",
        "provider": {"kind": "deterministic", "availability": "available"},
        "models_configured": 2,
    }
    _assert_request_id(response)


def test_models_are_exactly_configuration_driven_and_never_claim_loaded_state() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "discovery": {
            "source": "configuration",
            "live": False,
            "availability": "available",
        },
        "data": [
            {
                "id": "public-chat",
                "object": "model",
                "provider": "deterministic",
                "capabilities": ["chat"],
                "availability": "available",
                "description": "Configured chat model",
            },
            {
                "id": "public-embed",
                "object": "model",
                "provider": "deterministic",
                "capabilities": ["embeddings"],
                "availability": "available",
                "description": None,
            },
        ],
    }
    assert RUNTIME_SENTINEL not in response.text
    assert "private-embedding-runtime" not in response.text
    _assert_request_id(response)


def test_capabilities_state_the_small_non_streaming_contract_exactly() -> None:
    with _client(create_app(_config())) as client:
        response = client.get("/v1/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "api_version": "v1",
        "model_discovery": "configuration",
        "callable_capabilities": ["chat"],
        "chat_completions": {
            "supported": True,
            "streaming": False,
            "message_roles": ["system", "user", "assistant"],
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
        "provider": "deterministic",
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


def test_chat_maps_public_model_to_runtime_name_and_preserves_provider_usage() -> None:
    provider = RecordingProvider(
        result=ProviderChatResult(
            content="mapped reply",
            finish_reason="length",
            usage=ProviderTokenUsage(prompt_tokens=8, completion_tokens=3),
        )
    )
    app = create_app(
        _config(),
        provider=provider,
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
            runtime_model=RUNTIME_SENTINEL,
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


def test_openapi_contains_only_the_four_documented_application_routes() -> None:
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
        "/v1/capabilities": ["get"],
        "/v1/chat/completions": ["post"],
    }
    assert schema["info"]["license"] == {"name": "AGPL-3.0-only"}
    _assert_request_id(response)


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
    with _client(create_app(_config(), provider=provider)) as client:
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
    with _client(create_app(_config(), provider=provider)) as client:
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


@pytest.mark.parametrize(
    ("model", "stream", "capability"),
    [
        pytest.param("public-embed", False, "chat", id="model-lacks-chat"),
        pytest.param("public-chat", True, "streaming", id="streaming-not-supported"),
    ],
)
def test_unsupported_capability_is_normalized_before_provider_invocation(
    model: str,
    stream: bool,
    capability: str,
) -> None:
    provider = RecordingProvider()
    with _client(create_app(_config(), provider=provider)) as client:
        response = client.post(
            "/v1/chat/completions",
            json=_valid_chat(model=model, stream=stream),
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
    ("failure", "status", "code", "message", "retryable"),
    [
        pytest.param(
            ProviderUnavailableError(),
            503,
            "provider_unavailable",
            "The configured local provider is unavailable.",
            True,
            id="provider-unavailable",
        ),
        pytest.param(
            ModelUnavailableError(),
            503,
            "model_unavailable",
            "The configured model is unavailable in the local provider.",
            False,
            id="model-unavailable",
        ),
        pytest.param(
            ProviderTimeoutError(),
            504,
            "provider_timeout",
            "The configured local provider timed out.",
            True,
            id="provider-timeout",
        ),
        pytest.param(
            ProviderError(),
            502,
            "provider_error",
            "The configured local provider rejected or failed the request.",
            True,
            id="provider-error",
        ),
        pytest.param(
            ProviderError(retryable=False),
            502,
            "provider_error",
            "The configured local provider rejected or failed the request.",
            False,
            id="provider-rejection",
        ),
        pytest.param(
            ProviderProtocolError(),
            502,
            "provider_protocol_error",
            "The configured local provider returned an invalid response.",
            False,
            id="provider-protocol",
        ),
        pytest.param(
            ConfigurationError(),
            500,
            "configuration_error",
            "Vulcan is not configured to complete this request.",
            False,
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
) -> None:
    provider = RecordingProvider(failure=failure)
    with _client(create_app(_config(), provider=provider)) as client:
        response = client.post("/v1/chat/completions", json=_valid_chat())

    assert response.status_code == status
    request_id = _assert_request_id(response, error_body=True)
    assert response.json() == _expected_error(
        request_id=request_id,
        code=code,
        message=message,
        retryable=retryable,
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
        create_app(_config(), provider=provider),
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
