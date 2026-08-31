"""Provider selection and adapter contract tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import cast

import httpx
import pytest

from vulcan.config import (
    DeterministicProviderConfig,
    OllamaProviderConfig,
    ProviderConfig,
)
from vulcan.errors import (
    ModelUnavailableError,
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
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.factory import build_provider
from vulcan.providers.ollama import OllamaProvider

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def _chat_request(
    *,
    provider_model: str = "runtime-model",
    temperature: float | None = None,
    max_tokens: int | None = None,
    keep_alive: str | None = None,
    think: bool | None = None,
) -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model=provider_model,
        messages=(
            ProviderMessage(role="system", content="Be concise."),
            ProviderMessage(role="user", content="Say hello."),
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        think=think,
        keep_alive=keep_alive,
    )


def _ollama_config() -> OllamaProviderConfig:
    return OllamaProviderConfig(
        type="ollama",
        base_url="http://127.0.0.1:11434",
        timeout_seconds=1.0,
    )


async def _invoke_ollama(
    handler: MockHandler,
    *,
    request: ProviderChatRequest | None = None,
) -> ProviderChatResult:
    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OllamaProvider("local-ollama", _ollama_config(), client=client)
    try:
        return await provider.chat(request or _chat_request())
    finally:
        await provider.aclose()


def test_factory_selects_deterministic_provider_exactly() -> None:
    provider = build_provider(
        "det", DeterministicProviderConfig(type="deterministic", response_text="fixed")
    )

    assert type(provider) is DeterministicProvider
    assert provider.provider_id == "det"
    assert provider.provider_type == "deterministic"


def test_factory_selects_ollama_provider_exactly() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    provider = build_provider("local-ollama", _ollama_config(), client=client)

    assert type(provider) is OllamaProvider
    assert provider.provider_id == "local-ollama"
    assert provider.provider_type == "ollama"
    asyncio.run(provider.aclose())


def test_factory_has_no_unknown_provider_fallback() -> None:
    unknown_config = cast(ProviderConfig, object())

    with pytest.raises(AssertionError):
        build_provider("unknown", unknown_config)


def test_deterministic_provider_is_repeatable_and_performs_no_http_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden_post(*_: object, **__: object) -> httpx.Response:
        raise AssertionError("deterministic provider attempted HTTP I/O")

    monkeypatch.setattr(httpx.AsyncClient, "post", forbidden_post)
    provider = DeterministicProvider(
        "det",
        DeterministicProviderConfig(
            type="deterministic",
            response_text="the configured deterministic response",
        ),
    )
    first_request = _chat_request(provider_model="one", temperature=0.1, max_tokens=3)
    second_request = _chat_request(provider_model="two", temperature=1.9, max_tokens=99)

    async def exercise() -> tuple[ProviderChatResult, ProviderChatResult]:
        first = await provider.chat(first_request)
        second = await provider.chat(second_request)
        await provider.aclose()
        return first, second

    first, second = asyncio.run(exercise())

    expected = ProviderChatResult(
        content="the configured deterministic response",
        finish_reason="stop",
        usage=None,
    )
    assert first == expected
    assert second == expected


def test_ollama_posts_exact_native_chat_request_and_parses_success() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "runtime-model",
                "message": {"role": "assistant", "content": "Hello."},
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 12,
                "eval_count": 4,
                "provider_extension": "ignored",
            },
        )

    result = asyncio.run(
        _invoke_ollama(
            handler,
            request=_chat_request(temperature=0.25, max_tokens=37),
        )
    )

    assert len(captured) == 1
    sent = captured[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://127.0.0.1:11434/api/chat"
    assert json.loads(sent.content) == {
        "model": "runtime-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello."},
        ],
        "stream": False,
        "options": {"temperature": 0.25, "num_predict": 37},
    }
    assert result == ProviderChatResult(
        content="Hello.",
        finish_reason="length",
        usage=ProviderTokenUsage(prompt_tokens=12, completion_tokens=4),
    )


def test_ollama_chat_payload_carries_keep_alive_only_when_set() -> None:
    # The exact-payload test above pins the unset case byte-for-byte; this one
    # pins presence when configured and absence (not null) when not.
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "runtime-model",
                "message": {"role": "assistant", "content": "Hello."},
                "done": True,
            },
        )

    asyncio.run(_invoke_ollama(handler, request=_chat_request(keep_alive="2h")))
    asyncio.run(_invoke_ollama(handler, request=_chat_request()))

    assert json.loads(captured[0].content)["keep_alive"] == "2h"
    assert "keep_alive" not in json.loads(captured[1].content)


def test_ollama_chat_payload_carries_think_only_when_the_caller_states_it() -> None:
    # A thinking model otherwise spends an invisible share of the caller's token
    # budget on reasoning. `think` must reach the wire when stated -- including
    # False, which is the whole point -- and be absent (not null) when unstated,
    # so an unstated request is byte-identical to before the field existed.
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "runtime-model",
                "message": {"role": "assistant", "content": "Hello."},
                "done": True,
            },
        )

    asyncio.run(_invoke_ollama(handler, request=_chat_request(think=False)))
    asyncio.run(_invoke_ollama(handler, request=_chat_request(think=True)))
    asyncio.run(_invoke_ollama(handler, request=_chat_request()))

    assert json.loads(captured[0].content)["think"] is False
    assert json.loads(captured[1].content)["think"] is True
    assert "think" not in json.loads(captured[2].content)


def test_ollama_chat_surfaces_thinking_the_caller_was_billed_for() -> None:
    # Ollama counts thinking in eval_count. Before this field existed the strict
    # message model dropped it, so callers paid for output they could not see.
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "runtime-model",
                "message": {
                    "role": "assistant",
                    "content": "OK",
                    "thinking": "The user asked for one word.",
                },
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 39,
            },
        )

    result = asyncio.run(_invoke_ollama(handler))

    assert result == ProviderChatResult(
        content="OK",
        finish_reason="stop",
        usage=ProviderTokenUsage(prompt_tokens=11, completion_tokens=39),
        thinking="The user asked for one word.",
    )


def test_ollama_chat_result_thinking_is_none_when_the_model_returns_none() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "runtime-model",
                "message": {"role": "assistant", "content": "Hello."},
                "done": True,
            },
        )

    assert asyncio.run(_invoke_ollama(handler)).thinking is None


@pytest.mark.parametrize(
    "usage_fields",
    [
        {},
        {"prompt_eval_count": 9},
        {"eval_count": 2},
    ],
)
def test_ollama_does_not_invent_incomplete_usage(usage_fields: dict[str, int]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "complete"},
                "done": True,
                "done_reason": "stop",
                **usage_fields,
            },
        )

    result = asyncio.run(_invoke_ollama(handler))

    assert result.usage is None


def test_ollama_maps_runtime_404_to_model_unavailable() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'runtime-model' not found"})

    with pytest.raises(ModelUnavailableError):
        asyncio.run(_invoke_ollama(handler))


def test_ollama_maps_generic_404_to_non_retryable_provider_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "page not found"})

    with pytest.raises(ProviderError) as raised:
        asyncio.run(_invoke_ollama(handler))

    assert type(raised.value) is ProviderError
    assert raised.value.retryable is False


def test_ollama_maps_connection_failure_to_provider_unavailable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        asyncio.run(_invoke_ollama(handler))


def test_ollama_maps_timeout_to_provider_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider stalled", request=request)

    with pytest.raises(ProviderTimeoutError):
        asyncio.run(_invoke_ollama(handler))


@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(400, False), (408, True), (500, True), (503, True)],
)
def test_ollama_maps_other_non_success_statuses_to_provider_error(
    status_code: int,
    retryable: bool,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "must not escape"})

    with pytest.raises(ProviderError) as raised:
        asyncio.run(_invoke_ollama(handler))

    assert raised.value.retryable is retryable


def test_ollama_maps_429_to_rate_limit_error_like_hosted_providers() -> None:
    """A 429 must not depend on provider type: hosted and local both surface
    provider_rate_limited (HTTP 429, retryable) — never a generic 502."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "must not escape"})

    with pytest.raises(ProviderRateLimitError):
        asyncio.run(_invoke_ollama(handler))


def test_ollama_maps_malformed_json_to_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"{this is not JSON",
            headers={"content-type": "application/json"},
        )

    with pytest.raises(ProviderProtocolError):
        asyncio.run(_invoke_ollama(handler))


@pytest.mark.parametrize(
    "body",
    [
        {"message": {"role": "user", "content": "wrong role"}, "done": True},
        {"message": {"content": "missing role"}, "done": True},
        {"message": {"role": "assistant", "content": 123}, "done": True},
        {"message": {"role": "assistant", "content": "bad done"}, "done": "true"},
        {
            "message": {"role": "assistant", "content": "bad prompt count"},
            "done": True,
            "prompt_eval_count": "1",
        },
        {
            "message": {"role": "assistant", "content": "bad eval count"},
            "done": True,
            "eval_count": True,
        },
    ],
)
def test_ollama_rejects_coercive_or_non_assistant_success_fields(
    body: dict[str, object],
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        asyncio.run(_invoke_ollama(handler))


@pytest.mark.parametrize(
    "body",
    [
        {"done": True},
        {"message": {"content": "missing done"}},
        {"message": {"content": "not complete"}, "done": False},
    ],
)
def test_ollama_maps_incomplete_response_to_protocol_error(body: dict[str, object]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        asyncio.run(_invoke_ollama(handler))


@pytest.mark.parametrize(
    "usage_fields",
    [
        {"prompt_eval_count": -1, "eval_count": 0},
        {"prompt_eval_count": 0, "eval_count": -1},
        {"prompt_eval_count": -1},
        {"eval_count": -1},
    ],
)
def test_ollama_maps_negative_usage_to_protocol_error(usage_fields: dict[str, int]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "invalid usage"},
                "done": True,
                **usage_fields,
            },
        )

    with pytest.raises(ProviderProtocolError):
        asyncio.run(_invoke_ollama(handler))
