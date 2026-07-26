"""Anthropic Messages adapter tests over mocked transports only.

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

from vulcan.config import AnthropicProviderConfig
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
from vulcan.providers.anthropic import ANTHROPIC_VERSION, AnthropicProvider
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderTokenUsage,
)

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]

KEY_ENV = "VULCAN_TEST_ANTHROPIC_API_KEY"
KEY_SENTINEL = "sk-ant-secret-9b2ac1d4-must-not-leak"
BODY_SENTINEL = "anthropic-body-must-not-escape-77e0b911"
PROMPT_SENTINEL = "anthropic-prompt-2f8f4e60"


def _config(**overrides: Any) -> AnthropicProviderConfig:
    fields: dict[str, Any] = {
        "type": "anthropic",
        "api_key_env": KEY_ENV,
        "timeout_seconds": 1.0,
    }
    fields.update(overrides)
    return AnthropicProviderConfig(**fields)


def _chat_request(
    messages: tuple[ProviderMessage, ...] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model="example-anthropic-model",
        messages=messages
        or (
            ProviderMessage(role="system", content="Be concise."),
            ProviderMessage(role="user", content=PROMPT_SENTINEL),
        ),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _invoke(
    handler: MockHandler,
    *,
    config: AnthropicProviderConfig | None = None,
    request: ProviderChatRequest | None = None,
) -> ProviderChatResult:
    resolved = config or _config()
    client = httpx.AsyncClient(
        base_url=resolved.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = AnthropicProvider("anthropic", resolved, client=client)

    async def run() -> ProviderChatResult:
        try:
            return await provider.chat(request or _chat_request())
        finally:
            await provider.aclose()

    return asyncio.run(run())


def _success_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "msg_upstream",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello."}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def _credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_SENTINEL)


def test_posts_exact_messages_request_with_api_key_and_version_headers() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    result = _invoke(handler, request=_chat_request(temperature=0.5, max_tokens=64))

    assert len(captured) == 1
    sent = captured[0]
    assert sent.method == "POST"
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["x-api-key"] == KEY_SENTINEL
    assert sent.headers["anthropic-version"] == ANTHROPIC_VERSION
    assert "authorization" not in sent.headers
    assert json.loads(sent.content) == {
        "model": "example-anthropic-model",
        "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
        "max_tokens": 64,
        "system": "Be concise.",
        "temperature": 0.5,
    }
    assert result == ProviderChatResult(
        content="Hello.",
        finish_reason="stop",
        usage=ProviderTokenUsage(prompt_tokens=12, completion_tokens=4),
    )


def test_missing_max_tokens_uses_the_configured_default() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(handler, config=_config(default_max_tokens=1234))

    payload = json.loads(captured[0].content)
    assert payload["max_tokens"] == 1234
    assert "temperature" not in payload


def test_system_messages_from_any_position_are_concatenated_in_order() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(
        handler,
        request=_chat_request(
            (
                ProviderMessage(role="system", content="First rule."),
                ProviderMessage(role="user", content="Question one."),
                ProviderMessage(role="system", content="Second rule."),
                ProviderMessage(role="assistant", content="Answer one."),
                ProviderMessage(role="user", content="Question two."),
            )
        ),
    )

    payload = json.loads(captured[0].content)
    assert payload["system"] == "First rule.\n\nSecond rule."
    assert payload["messages"] == [
        {"role": "user", "content": "Question one."},
        {"role": "assistant", "content": "Answer one."},
        {"role": "user", "content": "Question two."},
    ]


def test_consecutive_same_role_turns_are_merged_for_strict_alternation() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(
        handler,
        request=_chat_request(
            (
                ProviderMessage(role="user", content="Part one."),
                ProviderMessage(role="user", content="Part two."),
                ProviderMessage(role="assistant", content="Reply."),
                ProviderMessage(role="user", content="Follow-up."),
            )
        ),
    )

    payload = json.loads(captured[0].content)
    assert payload["messages"] == [
        {"role": "user", "content": "Part one.\n\nPart two."},
        {"role": "assistant", "content": "Reply."},
        {"role": "user", "content": "Follow-up."},
    ]


def test_assistant_first_conversation_is_rejected_locally_without_io() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("assistant-led conversations must be rejected before HTTP")

    with pytest.raises(UnsupportedCapabilityError) as caught:
        _invoke(
            handler,
            request=_chat_request(
                (
                    ProviderMessage(role="assistant", content="I begin."),
                    ProviderMessage(role="user", content="You begin?"),
                )
            ),
        )

    assert caught.value.details == {"capability": "assistant_first_conversation"}


def test_temperature_above_one_is_rejected_locally_without_io() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("out-of-range temperature must be rejected before HTTP")

    with pytest.raises(UnsupportedCapabilityError) as caught:
        _invoke(handler, request=_chat_request(temperature=1.5))

    assert caught.value.details == {"capability": "temperature_above_one"}


def test_temperature_at_most_one_passes_through() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_success_body())

    _invoke(handler, request=_chat_request(temperature=1.0))

    assert json.loads(captured[0].content)["temperature"] == 1.0


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("tool_use", None),
        (None, None),
    ],
)
def test_stop_reason_mapping(stop_reason: str | None, expected: str | None) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(stop_reason=stop_reason))

    assert _invoke(handler).finish_reason == expected


def test_multiple_text_blocks_are_concatenated() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(
                content=[
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ]
            ),
        )

    assert _invoke(handler).content == "Part one. Part two."


def test_empty_content_with_max_tokens_is_an_empty_reply() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(content=[], stop_reason="max_tokens"))

    result = _invoke(handler)

    assert result.content == ""
    assert result.finish_reason == "length"


def test_non_text_content_block_is_a_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_body(
                content=[
                    {"type": "text", "text": "before"},
                    {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
                ]
            ),
        )

    with pytest.raises(ProviderProtocolError):
        _invoke(handler)


@pytest.mark.parametrize(
    "usage",
    [None, {"input_tokens": 9}, {"output_tokens": 2}, {}],
)
def test_incomplete_usage_is_never_invented(usage: dict[str, int] | None) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(usage=usage))

    assert _invoke(handler).usage is None


def test_missing_credential_fails_before_any_http_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP request may be sent without a credential")

    with pytest.raises(MissingCredentialError) as caught:
        _invoke(handler)

    assert caught.value.details == {"api_key_env": KEY_ENV}


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


def test_timeout_and_transport_failures_are_normalized() -> None:
    async def timed_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    with pytest.raises(ProviderTimeoutError):
        _invoke(timed_out)

    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        _invoke(refused)


@pytest.mark.parametrize(
    "body",
    [
        {"role": "user", "content": [{"type": "text", "text": "wrong role"}]},
        {"role": "assistant", "content": "not-a-list"},
        {"role": "assistant", "content": [{"type": "text", "text": 123}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": "1", "output_tokens": 2},
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": -1, "output_tokens": 2},
        },
    ],
)
def test_invalid_response_shapes_are_protocol_errors(body: dict[str, Any]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        _invoke(handler)


def test_malformed_json_is_a_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=b"{not json", headers={"content-type": "application/json"}
        )

    with pytest.raises(ProviderProtocolError):
        _invoke(handler)


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
    provider = AnthropicProvider("anthropic", config, client=client)

    async def run():
        try:
            return await provider.discover_runtime()
        finally:
            await provider.aclose()

    probe = asyncio.run(run())

    assert probe.live is False
    assert probe.provider_availability == "unchecked"
    assert probe.runtime_names is None
