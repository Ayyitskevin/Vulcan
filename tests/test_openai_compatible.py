"""OpenAI-compatible adapter tests over mocked transports only.

No test contacts a real API: every HTTP exchange goes through
``httpx.MockTransport`` and credentials are synthetic sentinels.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any

import httpx
import pytest

from vulcan.config import OpenAICompatibleProviderConfig
from vulcan.errors import (
    MissingCredentialError,
    ModelUnavailableError,
    ProviderAuthError,
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderTokenUsage,
)
from vulcan.providers.openai_compatible import OpenAICompatibleProvider

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]

KEY_ENV = "VULCAN_TEST_COMPAT_API_KEY"
KEY_SENTINEL = "sk-compat-secret-3c1de9a2-must-not-leak"
BODY_SENTINEL = "upstream-body-must-not-escape-51f0c377"
PROMPT_SENTINEL = "compat-prompt-e5b7f215"


def _config(**overrides: Any) -> OpenAICompatibleProviderConfig:
    fields: dict[str, Any] = {
        "type": "openai_compatible",
        "base_url": "https://api.example-vendor.com/v1",
        "api_key_env": KEY_ENV,
        "timeout_seconds": 1.0,
    }
    fields.update(overrides)
    return OpenAICompatibleProviderConfig(**fields)


def _chat_request(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model="vendor-native-model",
        messages=(
            ProviderMessage(role="system", content="Be concise."),
            ProviderMessage(role="user", content=PROMPT_SENTINEL),
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _invoke(
    handler: MockHandler,
    *,
    config: OpenAICompatibleProviderConfig | None = None,
    request: ProviderChatRequest | None = None,
) -> ProviderChatResult:
    resolved = config or _config()
    client = httpx.AsyncClient(
        base_url=resolved.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OpenAICompatibleProvider("compat", resolved, client=client)

    async def run() -> ProviderChatResult:
        try:
            return await provider.chat(request or _chat_request())
        finally:
            await provider.aclose()

    return asyncio.run(run())


def _success_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def _credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_SENTINEL)


def test_posts_exact_chat_completions_request_with_bearer_auth() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    result = _invoke(handler, request=_chat_request(temperature=0.25, max_tokens=37))

    assert len(captured) == 1
    sent = captured[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.example-vendor.com/v1/chat/completions"
    assert sent.headers["Authorization"] == f"Bearer {KEY_SENTINEL}"
    assert json.loads(sent.content) == {
        "model": "vendor-native-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": PROMPT_SENTINEL},
        ],
        "stream": False,
        "temperature": 0.25,
        "max_tokens": 37,
    }
    assert result == ProviderChatResult(
        content="Hello.",
        finish_reason="stop",
        usage=ProviderTokenUsage(prompt_tokens=12, completion_tokens=4),
    )


def test_omits_sampling_fields_when_client_did_not_set_them() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(handler)

    payload = json.loads(captured[0].content)
    assert "temperature" not in payload
    assert "max_tokens" not in payload
    assert "max_completion_tokens" not in payload


def test_configurable_max_completion_tokens_field() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(
        handler,
        config=_config(max_tokens_field="max_completion_tokens"),
        request=_chat_request(max_tokens=99),
    )

    payload = json.loads(captured[0].content)
    assert payload["max_completion_tokens"] == 99
    assert "max_tokens" not in payload


def test_vendor_base_url_path_is_preserved() -> None:
    """Z.AI-style deep paths must survive URL joining."""

    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(handler, config=_config(base_url="https://api.example-vendor.com/api/paas/v4"))

    assert str(captured[0].url) == "https://api.example-vendor.com/api/paas/v4/chat/completions"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("stop", "stop"), ("length", "length"), ("tool_calls", None), (None, None)],
)
def test_finish_reason_mapping(finish_reason: str | None, expected: str | None) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        body = _success_body()
        body["choices"][0]["finish_reason"] = finish_reason
        return httpx.Response(200, json=body)

    assert _invoke(handler).finish_reason == expected


@pytest.mark.parametrize(
    "usage",
    [None, {"prompt_tokens": 9}, {"completion_tokens": 2}, {}],
)
def test_incomplete_usage_is_never_invented(usage: dict[str, int] | None) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(usage=usage))

    assert _invoke(handler).usage is None


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "completion_tokens": 4},
        {"prompt_tokens": 4, "completion_tokens": -1},
    ],
)
def test_negative_usage_is_a_protocol_error(usage: dict[str, int]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(usage=usage))

    with pytest.raises(ProviderProtocolError):
        _invoke(handler)


def test_missing_credential_fails_before_any_http_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP request may be sent without a credential")

    with pytest.raises(MissingCredentialError) as caught:
        _invoke(handler)

    assert caught.value.details == {"api_key_env": KEY_ENV}


@pytest.mark.parametrize("unusable", ["", "   ", "with space", "line\nbreak"])
def test_unusable_credential_values_fail_closed(
    monkeypatch: pytest.MonkeyPatch, unusable: str
) -> None:
    monkeypatch.setenv(KEY_ENV, unusable)

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP request may be sent with an unusable credential")

    with pytest.raises(MissingCredentialError) as caught:
        _invoke(handler)

    assert caught.value.details == {"api_key_env": KEY_ENV}
    if unusable.strip():
        assert unusable not in repr(caught.value.details)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderAuthError),
        (403, ProviderAuthError),
        (404, ModelUnavailableError),
        (429, ProviderRateLimitError),
        (503, ProviderUnavailableError),
        (529, ProviderUnavailableError),
    ],
)
def test_hosted_status_normalization(status: int, expected: type[Exception]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": BODY_SENTINEL}})

    with pytest.raises(expected):
        _invoke(handler)


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (408, True), (409, False), (500, True), (502, True)],
)
def test_other_statuses_map_to_provider_error_with_honest_retryability(
    status: int, retryable: bool
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": BODY_SENTINEL}})

    with pytest.raises(ProviderError) as caught:
        _invoke(handler)

    assert type(caught.value) is ProviderError
    assert caught.value.retryable is retryable


def test_timeout_and_transport_failures_are_normalized() -> None:
    async def timed_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    with pytest.raises(ProviderTimeoutError):
        _invoke(timed_out)

    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        _invoke(refused)


def test_malformed_json_is_a_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"{not json", headers={"content-type": "application/json"}
        )

    with pytest.raises(ProviderProtocolError):
        _invoke(handler)


@pytest.mark.parametrize(
    "body",
    [
        {"choices": []},
        {"choices": [{"message": {"role": "user", "content": "wrong role"}}]},
        {"choices": [{"message": {"role": "assistant", "content": None}}]},
        {"choices": [{"message": {"role": "assistant", "content": 123}}]},
        {"choices": [{"message": {"content": "missing role"}}]},
        {"choices": "not-a-list"},
        {},
        {
            "choices": [{"message": {"role": "assistant", "content": "x"}}],
            "usage": {"prompt_tokens": "1", "completion_tokens": 2},
        },
    ],
)
def test_invalid_response_shapes_are_protocol_errors(body: dict[str, Any]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        _invoke(handler)


def test_unknown_vendor_extension_fields_are_ignored() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        body = _success_body(system_fingerprint="fp_x", vendor_extension={"nested": True})
        body["choices"][0]["message"]["reasoning_content"] = "vendor extra"
        return httpx.Response(200, json=body)

    result = _invoke(handler)

    assert result.content == "Hello."


@pytest.mark.parametrize("status", [401, 429, 500])
def test_upstream_bodies_and_credentials_never_appear_in_raised_errors(status: int) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": BODY_SENTINEL}})

    with pytest.raises(Exception) as caught:
        _invoke(handler)

    rendered = f"{caught.value!r} {caught.value} {getattr(caught.value, 'details', None)!r}"
    assert BODY_SENTINEL not in rendered
    assert KEY_SENTINEL not in rendered


def test_discover_runtime_reports_unchecked_without_network() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("readiness must not probe hosted providers")

    config = _config()
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OpenAICompatibleProvider("compat", config, client=client)

    async def run():
        try:
            return await provider.discover_runtime()
        finally:
            await provider.aclose()

    probe = asyncio.run(run())

    assert probe.live is False
    assert probe.provider_availability == "unchecked"
    assert probe.runtime_names is None
