"""Vendor reasoning fields stay ignored, buffered and streamed.

DeepSeek's reasoner models return the model's chain of thought in a
``reasoning_content`` field alongside the answer, and count its tokens inside a
``completion_tokens_details`` sub-object. Vulcan's contract is the stable
OpenAI subset, so that field is dropped: never forwarded to the client, never
substituted for a missing answer, and never counted twice. These tests pin that
against the shapes DeepSeek actually emits so a future parser change cannot
start leaking reasoning by accident.

Every exchange goes through ``httpx.MockTransport``; no test contacts a real
API.
"""

from __future__ import annotations

import asyncio
import json
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
    OpenAICompatibleProviderConfig,
)
from vulcan.errors import ProviderProtocolError
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderStreamEvent,
    ProviderTokenUsage,
    StreamDelta,
    StreamEnd,
)
from vulcan.providers.openai_compatible import OpenAICompatibleProvider

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]

KEY_ENV = "VULCAN_REASONING_TEST_API_KEY"
KEY_SENTINEL = "sk-reasoning-secret-7d02"
PROMPT_SENTINEL = "reasoning-prompt-must-not-escape-19bc"
# The chain of thought: parsed, then discarded. It must never reach a client.
REASONING_SENTINEL = "reasoning-content-must-not-escape-4af6"
ANSWER = "Four."


@pytest.fixture(autouse=True)
def _credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, KEY_SENTINEL)


def _sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode("utf-8")


def _provider(handler: MockHandler) -> OpenAICompatibleProvider:
    config = OpenAICompatibleProviderConfig(
        type="openai_compatible",
        base_url="https://api.deepseek-mock.example/v1",
        api_key_env=KEY_ENV,
        timeout_seconds=1.0,
    )
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return OpenAICompatibleProvider("reasoner", config, client=client)


def _request() -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model="native-reasoner",
        messages=(ProviderMessage(role="user", content=PROMPT_SENTINEL),),
        temperature=None,
        max_tokens=None,
    )


def _chat(handler: MockHandler) -> ProviderChatResult:
    provider = _provider(handler)

    async def run() -> ProviderChatResult:
        try:
            return await provider.chat(_request())
        finally:
            await provider.aclose()

    return asyncio.run(run())


def _stream(handler: MockHandler) -> list[ProviderStreamEvent]:
    provider = _provider(handler)

    async def run() -> list[ProviderStreamEvent]:
        try:
            return [event async for event in provider.chat_stream(_request())]
        finally:
            await provider.aclose()

    return asyncio.run(run())


def _reasoning_body() -> dict[str, Any]:
    """The buffered shape a DeepSeek reasoner returns."""

    return {
        "id": "chatcmpl-upstream",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "reasoning_content": REASONING_SENTINEL,
                    "content": ANSWER,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 96,
            "total_tokens": 107,
            # Reasoning tokens are a breakdown of completion_tokens, not an
            # addition to them.
            "completion_tokens_details": {"reasoning_tokens": 90},
        },
    }


# ── Adapter: buffered ────────────────────────────────────────────────────────


def test_buffered_reasoning_response_parses_and_drops_the_reasoning() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reasoning_body())

    result = _chat(handler)

    assert result == ProviderChatResult(
        content=ANSWER,
        finish_reason="stop",
        usage=ProviderTokenUsage(prompt_tokens=11, completion_tokens=96),
    )
    assert REASONING_SENTINEL not in repr(result)


def test_buffered_reasoning_token_details_do_not_inflate_usage() -> None:
    """`completion_tokens` is authoritative; the details sub-object is ignored."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reasoning_body())

    usage = _chat(handler).usage

    assert usage is not None
    assert usage.completion_tokens == 96  # not 96 + 90


@pytest.mark.parametrize("content", [None, 123, {"text": ANSWER}])
def test_buffered_reasoning_is_never_substituted_for_a_missing_answer(content: Any) -> None:
    """Reasoning is not a fallback: without a usable `content`, this is an error."""

    async def handler(_: httpx.Request) -> httpx.Response:
        body = _reasoning_body()
        body["choices"][0]["message"]["content"] = content
        return httpx.Response(200, json=body)

    with pytest.raises(ProviderProtocolError):
        _chat(handler)


# ── Adapter: streaming ───────────────────────────────────────────────────────


def test_streaming_reasoning_phase_yields_no_deltas() -> None:
    """DeepSeek streams reasoning first with `content` null, then the answer."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
                json.dumps(
                    {
                        "choices": [
                            {"delta": {"content": None, "reasoning_content": REASONING_SENTINEL}}
                        ]
                    }
                ),
                json.dumps({"choices": [{"delta": {"reasoning_content": " and so"}}]}),
                json.dumps({"choices": [{"delta": {"content": "Fo", "reasoning_content": None}}]}),
                json.dumps({"choices": [{"delta": {"content": "ur."}}]}),
                json.dumps(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 96},
                    }
                ),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    events = _stream(handler)

    assert events == [
        StreamDelta(text="Fo"),
        StreamDelta(text="ur."),
        StreamEnd(
            finish_reason="stop",
            usage=ProviderTokenUsage(prompt_tokens=11, completion_tokens=96),
        ),
    ]


def test_streaming_that_never_leaves_the_reasoning_phase_yields_no_text() -> None:
    """Hitting the token limit mid-reasoning ends the stream with no content."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"reasoning_content": REASONING_SENTINEL}}]}),
                json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    assert _stream(handler) == [StreamEnd(finish_reason="length", usage=None)]


def test_streaming_rejects_a_non_string_reasoning_bearing_content_field() -> None:
    """Strict parsing of the fields Vulcan uses is unchanged by vendor extras."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps(
                    {
                        "choices": [
                            {"delta": {"content": 5, "reasoning_content": REASONING_SENTINEL}}
                        ]
                    }
                ),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(ProviderProtocolError):
        _stream(handler)


# ── HTTP contract ────────────────────────────────────────────────────────────


def _app_config() -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        providers={
            "reasoner": OpenAICompatibleProviderConfig(
                type="openai_compatible",
                base_url="https://api.deepseek-mock.example/v1",
                api_key_env=KEY_ENV,
                timeout_seconds=1.0,
            )
        },
        models=(
            ModelConfig(
                id="public-reasoner",
                provider="reasoner",
                provider_model="native-reasoner",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )


def _client(handler: MockHandler) -> TestClient:
    app = create_app(_app_config(), providers={"reasoner": _provider(handler)})
    return TestClient(app, base_url="http://127.0.0.1")


def _post(client: TestClient, *, stream: bool) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "public-reasoner",
            "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
            "stream": stream,
        },
    )


def test_http_buffered_response_carries_the_answer_and_no_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reasoning_body())

    with caplog.at_level("DEBUG"), _client(handler) as client:
        response = _post(client, stream=False)

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == ANSWER
    assert body["usage"] == {"prompt_tokens": 11, "completion_tokens": 96, "total_tokens": 107}
    assert "reasoning_content" not in response.text
    for sentinel in (REASONING_SENTINEL, PROMPT_SENTINEL, KEY_SENTINEL, "native-reasoner"):
        assert sentinel not in response.text
        assert sentinel not in caplog.text


def test_http_streamed_frames_carry_the_answer_and_no_reasoning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps(
                    {
                        "choices": [
                            {"delta": {"content": None, "reasoning_content": REASONING_SENTINEL}}
                        ]
                    }
                ),
                json.dumps({"choices": [{"delta": {"content": ANSWER}}]}),
                json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    with caplog.at_level("DEBUG"), _client(handler) as client:
        response = _post(client, stream=True)

    assert response.status_code == 200
    frames = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    assert "".join(frame["choices"][0]["delta"].get("content", "") for frame in frames) == ANSWER
    assert "reasoning_content" not in response.text
    for sentinel in (REASONING_SENTINEL, PROMPT_SENTINEL, KEY_SENTINEL, "native-reasoner"):
        assert sentinel not in response.text
        assert sentinel not in caplog.text


def test_usage_counters_use_the_top_level_completion_tokens() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_reasoning_body())

    with _client(handler) as client:
        assert _post(client, stream=False).status_code == 200
        totals = client.get("/v1/usage").json()["totals"]

    assert totals["requests_with_usage"] == 1
    assert totals["completion_tokens"] == 96
    assert totals["total_tokens"] == 107
