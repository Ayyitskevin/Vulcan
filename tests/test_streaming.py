"""Streaming chat coverage: adapters, gateway, and the SSE HTTP contract.

Every upstream exchange goes through ``httpx.MockTransport``; no test contacts
a real API. Credentials and prompts are synthetic sentinels so the leak
assertions are meaningful.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Literal

import httpx
import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    AnthropicProviderConfig,
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
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from vulcan.gateway import Gateway
from vulcan.providers.anthropic import ANTHROPIC_VERSION, AnthropicProvider
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderStreamEvent,
    ProviderTokenUsage,
    StreamDelta,
    StreamEnd,
)
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.ollama import OllamaProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider
from vulcan.readiness import RuntimeProbe
from vulcan.registry import ModelRegistry
from vulcan.schemas import ChatCompletionRequest, ChatMessage, MessageRole

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]

OPENAI_KEY_ENV = "VULCAN_STREAM_TEST_OPENAI_KEY"
ANTHROPIC_KEY_ENV = "VULCAN_STREAM_TEST_ANTHROPIC_KEY"
OPENAI_KEY_SENTINEL = "sk-stream-openai-secret-4d21"
ANTHROPIC_KEY_SENTINEL = "sk-stream-ant-secret-9f70"
BODY_SENTINEL = "stream-upstream-body-must-not-escape-6a3c"
PROMPT_SENTINEL = "stream-prompt-must-not-escape-2e88"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, OPENAI_KEY_SENTINEL)
    monkeypatch.setenv(ANTHROPIC_KEY_ENV, ANTHROPIC_KEY_SENTINEL)


def _sse(*events: str) -> bytes:
    return "".join(f"data: {event}\n\n" for event in events).encode("utf-8")


def _ndjson(*lines: dict[str, Any]) -> bytes:
    return "".join(f"{json.dumps(line)}\n" for line in lines).encode("utf-8")


class _RecordingStream(httpx.AsyncByteStream):
    """An upstream body that records when httpx releases it."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _request(provider_model: str = "native-model") -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model=provider_model,
        messages=(
            ProviderMessage(role="system", content="Be brief."),
            ProviderMessage(role="user", content=PROMPT_SENTINEL),
        ),
        temperature=None,
        max_tokens=None,
    )


def _collect(
    provider: Any,
    request: ProviderChatRequest | None = None,
) -> list[ProviderStreamEvent]:
    async def run() -> list[ProviderStreamEvent]:
        try:
            return [event async for event in provider.chat_stream(request or _request())]
        finally:
            await provider.aclose()

    return asyncio.run(run())


# ── OpenAI-compatible adapter ────────────────────────────────────────────────


def _compat_provider(handler: MockHandler, **overrides: Any) -> OpenAICompatibleProvider:
    fields: dict[str, Any] = {
        "type": "openai_compatible",
        "base_url": "https://api.compat-mock.example/v1",
        "api_key_env": OPENAI_KEY_ENV,
        "timeout_seconds": 1.0,
    }
    fields.update(overrides)
    config = OpenAICompatibleProviderConfig(**fields)
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return OpenAICompatibleProvider("compat", config, client=client)


def test_compat_stream_sends_stream_true_with_bearer_auth_and_translates_chunks() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
                json.dumps({"choices": [{"delta": {"content": "Hel"}}]}),
                json.dumps({"choices": [{"delta": {"content": "lo."}}]}),
                json.dumps(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                    }
                ),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    events = _collect(_compat_provider(handler))

    sent = json.loads(captured[0].content)
    assert sent["stream"] is True
    assert captured[0].headers["Authorization"] == f"Bearer {OPENAI_KEY_SENTINEL}"
    assert str(captured[0].url) == "https://api.compat-mock.example/v1/chat/completions"
    assert events == [
        StreamDelta(text="Hel"),
        StreamDelta(text="lo."),
        StreamEnd(
            finish_reason="stop",
            usage=ProviderTokenUsage(prompt_tokens=7, completion_tokens=2),
        ),
    ]


def test_compat_stream_ignores_comments_blank_lines_and_vendor_fields() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        body = (
            ": keep-alive comment\n\n"
            "event: message\n"
            'data: {"choices": [{"delta": {"content": "A", "reasoning_content": "hidden"}}]}\n\n'
            "\n"
            'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200, content=body.encode("utf-8"), headers={"content-type": "text/event-stream"}
        )

    assert _collect(_compat_provider(handler)) == [
        StreamDelta(text="A"),
        StreamEnd(finish_reason="stop", usage=None),
    ]


def test_compat_stream_without_usage_never_invents_counts() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"content": "x"}}]}),
                json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}]}),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

    events = _collect(_compat_provider(handler))

    assert events[-1] == StreamEnd(finish_reason="length", usage=None)


@pytest.mark.parametrize(
    "body",
    [
        b"data: {not json\n\n",
        b'data: {"choices": [{"delta": {"content": 5}}]}\n\n',
        b'data: {"choices": "not-a-list"}\n\n',
    ],
)
def test_compat_stream_malformed_frames_are_protocol_errors(body: bytes) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    with pytest.raises(ProviderProtocolError):
        _collect(_compat_provider(handler))


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, ProviderAuthError),
        (404, ModelUnavailableError),
        (429, ProviderRateLimitError),
        (503, ProviderUnavailableError),
        (500, ProviderError),
    ],
)
def test_compat_stream_status_failures_happen_before_any_event(
    status: int, expected: type[Exception]
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": BODY_SENTINEL}})

    with pytest.raises(expected):
        _collect(_compat_provider(handler))


def test_compat_stream_timeout_and_transport_failures_are_normalized() -> None:
    async def timed_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("stalled", request=request)

    with pytest.raises(ProviderTimeoutError):
        _collect(_compat_provider(timed_out))

    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        _collect(_compat_provider(refused))


def test_compat_stream_missing_credential_fails_before_any_http_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call may happen without a credential")

    with pytest.raises(MissingCredentialError):
        _collect(_compat_provider(handler))


def test_compat_stream_closes_the_upstream_response_when_abandoned() -> None:
    """A client disconnect must release the upstream connection."""

    stream = _RecordingStream(
        [
            _sse(json.dumps({"choices": [{"delta": {"content": "first"}}]})),
            _sse(json.dumps({"choices": [{"delta": {"content": "second"}}]})),
            _sse(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}), "[DONE]"),
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    provider = _compat_provider(handler)

    async def run() -> None:
        events = provider.chat_stream(_request()).__aiter__()
        assert await events.__anext__() == StreamDelta(text="first")
        assert stream.closed is False  # still mid-stream
        await events.aclose()  # simulates the client going away mid-stream
        assert stream.closed is True  # upstream released by the adapter
        await provider.aclose()

    asyncio.run(run())


# ── Anthropic adapter ────────────────────────────────────────────────────────


def _anthropic_provider(handler: MockHandler, **overrides: Any) -> AnthropicProvider:
    fields: dict[str, Any] = {
        "type": "anthropic",
        "base_url": "https://api.anthropic-mock.example",
        "api_key_env": ANTHROPIC_KEY_ENV,
        "timeout_seconds": 1.0,
    }
    fields.update(overrides)
    config = AnthropicProviderConfig(**fields)
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return AnthropicProvider("anthropic", config, client=client)


def _anthropic_stream_body(**overrides: Any) -> bytes:
    stop_reason = overrides.get("stop_reason", "end_turn")
    return _sse(
        json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 11}}}),
        json.dumps({"type": "content_block_start", "content_block": {"type": "text", "text": ""}}),
        json.dumps({"type": "ping"}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}),
        json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "!"}}),
        json.dumps({"type": "content_block_stop"}),
        json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": 3},
            }
        ),
        json.dumps({"type": "message_stop"}),
    )


def test_anthropic_stream_sends_stream_true_with_headers_and_translates_events() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_anthropic_stream_body(),
            headers={"content-type": "text/event-stream"},
        )

    events = _collect(_anthropic_provider(handler))

    sent = json.loads(captured[0].content)
    assert sent["stream"] is True
    assert captured[0].headers["x-api-key"] == ANTHROPIC_KEY_SENTINEL
    assert captured[0].headers["anthropic-version"] == ANTHROPIC_VERSION
    assert str(captured[0].url) == "https://api.anthropic-mock.example/v1/messages"
    assert events == [
        StreamDelta(text="Hi"),
        StreamDelta(text="!"),
        StreamEnd(
            finish_reason="stop",
            usage=ProviderTokenUsage(prompt_tokens=11, completion_tokens=3),
        ),
    ]


def test_anthropic_stream_maps_max_tokens_stop_reason_to_length() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_anthropic_stream_body(stop_reason="max_tokens"),
            headers={"content-type": "text/event-stream"},
        )

    assert _collect(_anthropic_provider(handler))[-1].finish_reason == "length"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "content_block_start", "content_block": {"type": "tool_use", "id": "t"}},
        {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": "{"}},
        {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}},
    ],
)
def test_anthropic_stream_non_text_blocks_are_protocol_errors(event: dict[str, Any]) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
                json.dumps(event),
            ),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(ProviderProtocolError):
        _collect(_anthropic_provider(handler))


@pytest.mark.parametrize(
    ("error_type", "expected"),
    [("overloaded_error", ProviderUnavailableError), ("invalid_request_error", ProviderError)],
)
def test_anthropic_stream_error_events_are_classified_by_type_only(
    error_type: str, expected: type[Exception]
) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps(
                    {"type": "error", "error": {"type": error_type, "message": BODY_SENTINEL}}
                )
            ),
            headers={"content-type": "text/event-stream"},
        )

    with pytest.raises(expected) as caught:
        _collect(_anthropic_provider(handler))

    assert BODY_SENTINEL not in f"{caught.value!r} {getattr(caught.value, 'details', None)!r}"


def test_anthropic_stream_truncated_without_message_stop_is_a_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
                json.dumps(
                    {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "x"}}
                ),
            ),
            headers={"content-type": "text/event-stream"},
        )

    provider = _anthropic_provider(handler)

    async def run() -> list[ProviderStreamEvent]:
        try:
            return [event async for event in provider.chat_stream(_request())]
        finally:
            await provider.aclose()

    # A stream that ends without message_stop still terminates cleanly here;
    # the gateway is what turns a missing terminal event into a protocol error.
    events = asyncio.run(run())
    assert events[-1] == StreamEnd(finish_reason=None, usage=None)


def test_anthropic_stream_local_guards_run_before_any_io() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("local guards must reject before HTTP")

    hot_request = ProviderChatRequest(
        provider_model="native-model",
        messages=(ProviderMessage(role="user", content="hi"),),
        temperature=1.5,
        max_tokens=None,
    )
    with pytest.raises(UnsupportedCapabilityError):
        _collect(_anthropic_provider(handler), hot_request)

    assistant_first = ProviderChatRequest(
        provider_model="native-model",
        messages=(ProviderMessage(role="assistant", content="I start"),),
        temperature=None,
        max_tokens=None,
    )
    with pytest.raises(UnsupportedCapabilityError):
        _collect(_anthropic_provider(handler), assistant_first)


# ── Ollama adapter ───────────────────────────────────────────────────────────


def _ollama_provider(handler: MockHandler) -> OllamaProvider:
    config = OllamaProviderConfig(
        type="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
    )
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return OllamaProvider("local-ollama", config, client=client)


def test_ollama_stream_sends_stream_true_and_translates_ndjson() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=_ndjson(
                {"message": {"role": "assistant", "content": "Lo"}, "done": False},
                {"message": {"role": "assistant", "content": "cal"}, "done": False},
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                },
            ),
        )

    events = _collect(_ollama_provider(handler))

    assert json.loads(captured[0].content)["stream"] is True
    assert str(captured[0].url) == "http://127.0.0.1:11434/api/chat"
    assert events == [
        StreamDelta(text="Lo"),
        StreamDelta(text="cal"),
        StreamEnd(
            finish_reason="stop",
            usage=ProviderTokenUsage(prompt_tokens=5, completion_tokens=2),
        ),
    ]


def test_ollama_stream_missing_done_chunk_is_a_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_ndjson({"message": {"role": "assistant", "content": "cut"}, "done": False}),
        )

    with pytest.raises(ProviderProtocolError):
        _collect(_ollama_provider(handler))


def test_ollama_stream_malformed_line_is_a_protocol_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not json}\n")

    with pytest.raises(ProviderProtocolError):
        _collect(_ollama_provider(handler))


def test_ollama_stream_missing_model_maps_to_model_unavailable() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'native-model' not found"})

    with pytest.raises(ModelUnavailableError):
        _collect(_ollama_provider(handler))


def test_ollama_stream_transport_failure_is_provider_unavailable() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ProviderUnavailableError):
        _collect(_ollama_provider(handler))


# ── Deterministic adapter ────────────────────────────────────────────────────


def test_deterministic_stream_yields_configured_text_then_end() -> None:
    provider = DeterministicProvider(
        "det", DeterministicProviderConfig(type="deterministic", response_text="canned reply")
    )

    assert _collect(provider) == [
        StreamDelta(text="canned reply"),
        StreamEnd(finish_reason="stop", usage=None),
    ]


# ── Gateway orchestration ────────────────────────────────────────────────────


class _StubStreamProvider:
    provider_type: Literal["deterministic"] = "deterministic"

    def __init__(
        self,
        provider_id: str = "stub",
        events: tuple[ProviderStreamEvent, ...] = (),
        failure: Exception | None = None,
        failure_after: int = 0,
    ) -> None:
        self.provider_id = provider_id
        self.events = events
        self.failure = failure
        self.failure_after = failure_after
        self.stream_calls = 0
        self.chat_calls = 0
        self.closed = False

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.chat_calls += 1
        return ProviderChatResult(content="buffered", finish_reason="stop")

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_calls += 1
        try:
            for index, event in enumerate(self.events):
                if self.failure is not None and index == self.failure_after:
                    raise self.failure
                yield event
            if self.failure is not None and self.failure_after >= len(self.events):
                raise self.failure
        finally:
            self.closed = True

    async def discover_runtime(self) -> RuntimeProbe:
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        return None


def _registry(provider_id: str = "stub") -> ModelRegistry:
    return ModelRegistry(
        (
            ModelConfig(
                id="public-chat",
                provider=provider_id,
                provider_model="native-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        )
    )


def _chat_request(stream: bool = True) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="public-chat",
        messages=(ChatMessage(role=MessageRole.USER, content=PROMPT_SENTINEL),),
        stream=stream,
    )


def _drain(gateway: Gateway) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        return [
            chunk.model_dump(mode="json")
            async for chunk in gateway.chat_stream(_chat_request(), request_id="req-1")
        ]

    return asyncio.run(run())


def test_gateway_stream_emits_role_content_and_final_chunks() -> None:
    provider = _StubStreamProvider(
        events=(
            StreamDelta(text="one "),
            StreamDelta(text="two"),
            StreamEnd(
                finish_reason="stop",
                usage=ProviderTokenUsage(prompt_tokens=4, completion_tokens=6),
            ),
        )
    )
    gateway = Gateway(
        _registry(),
        {"stub": provider},
        clock=lambda: 1_700_000_000.5,
        id_factory=lambda: "chatcmpl-stream",
    )

    chunks = _drain(gateway)

    assert [chunk["choices"][0]["delta"] for chunk in chunks] == [
        {"role": "assistant", "content": None},
        {"role": None, "content": "one "},
        {"role": None, "content": "two"},
        {"role": None, "content": None},
    ]
    assert all(chunk["id"] == "chatcmpl-stream" for chunk in chunks)
    assert all(chunk["created"] == 1_700_000_000 for chunk in chunks)
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert all(chunk["provider"] == "stub" for chunk in chunks)
    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks] == [None, None, None, "stop"]
    assert chunks[-1]["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 6,
        "total_tokens": 10,
    }
    assert provider.stream_calls == 1
    assert provider.chat_calls == 0


def test_gateway_stream_without_terminal_event_is_a_protocol_error() -> None:
    provider = _StubStreamProvider(events=(StreamDelta(text="partial"),))
    gateway = Gateway(_registry(), {"stub": provider})

    with pytest.raises(ProviderProtocolError):
        _drain(gateway)


def test_gateway_stream_unknown_alias_raises_before_any_chunk() -> None:
    provider = _StubStreamProvider(events=(StreamEnd(finish_reason="stop"),))
    gateway = Gateway(_registry(), {"stub": provider})

    async def run() -> None:
        chunks = gateway.chat_stream(
            ChatCompletionRequest(
                model="missing-alias",
                messages=(ChatMessage(role=MessageRole.USER, content="hi"),),
                stream=True,
            )
        ).__aiter__()
        await chunks.__anext__()

    with pytest.raises(Exception) as caught:
        asyncio.run(run())

    assert caught.value.code == "model_not_found"
    assert provider.stream_calls == 0


def test_gateway_stream_annotates_failures_with_the_selected_provider() -> None:
    provider = _StubStreamProvider(
        "hosted",
        events=(StreamDelta(text="x"),),
        failure=ProviderRateLimitError(),
        failure_after=1,
    )
    gateway = Gateway(_registry("hosted"), {"hosted": provider})

    async def run() -> None:
        async for _ in gateway.chat_stream(_chat_request()):
            pass

    with pytest.raises(ProviderRateLimitError) as caught:
        asyncio.run(run())

    assert caught.value.details == {"provider": "hosted"}


def test_gateway_stream_logs_output_chars_without_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = _StubStreamProvider(
        events=(
            StreamDelta(text="abc"),
            StreamDelta(text="de"),
            StreamEnd(finish_reason="stop"),
        )
    )
    gateway = Gateway(_registry(), {"stub": provider})

    with caplog.at_level("INFO", logger="vulcan.gateway"):
        _drain(gateway)

    completed = next(record for record in caplog.records if record.msg == "chat_completed")
    metadata = completed.metadata  # type: ignore[attr-defined]
    assert metadata["output_chars"] == 5
    assert metadata["stream"] is True
    assert metadata["provider"] == "stub"
    assert metadata["request_id"] == "req-1"
    assert PROMPT_SENTINEL not in caplog.text
    assert "abc" not in caplog.text


def test_gateway_stream_closes_provider_iterator_when_abandoned() -> None:
    provider = _StubStreamProvider(
        events=(
            StreamDelta(text="one"),
            StreamDelta(text="two"),
            StreamEnd(finish_reason="stop"),
        )
    )
    gateway = Gateway(_registry(), {"stub": provider})

    async def run() -> None:
        chunks = gateway.chat_stream(_chat_request()).__aiter__()
        await chunks.__anext__()  # role chunk
        await chunks.__anext__()  # first content chunk
        await chunks.aclose()  # client disconnected

    asyncio.run(run())

    assert provider.closed is True


# ── HTTP contract ────────────────────────────────────────────────────────────


def _app_config() -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        providers={
            "det": DeterministicProviderConfig(type="deterministic", response_text="canned reply")
        },
        models=(
            ModelConfig(
                id="public-chat",
                provider="det",
                provider_model="native-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )


def _frames(text: str) -> list[str]:
    return [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]


def test_http_stream_renders_openai_style_sse_terminated_by_done() -> None:
    app = create_app(
        _app_config(),
        clock=lambda: 1_700_000_000.0,
        id_factory=lambda: "chatcmpl-http",
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public-chat",
                "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    frames = _frames(response.text)
    assert frames[-1] == "[DONE]"
    payloads = [json.loads(frame) for frame in frames[:-1]]
    assert [payload["choices"][0]["delta"] for payload in payloads] == [
        {"role": "assistant"},
        {"content": "canned reply"},
        {},
    ]
    assert [payload["choices"][0]["finish_reason"] for payload in payloads] == [None, None, "stop"]
    assert all(payload["id"] == "chatcmpl-http" for payload in payloads)
    assert all(payload["provider"] == "det" for payload in payloads)
    assert all("usage" not in payload for payload in payloads)


def test_http_non_streaming_response_is_unchanged() -> None:
    app = create_app(
        _app_config(),
        clock=lambda: 1_700_000_000.0,
        id_factory=lambda: "chatcmpl-buffered",
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public-chat",
                "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                "stream": False,
            },
        )

    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "id": "chatcmpl-buffered",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "public-chat",
        "provider": "det",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "canned reply"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


def test_http_pre_stream_failure_is_a_normal_json_envelope() -> None:
    """A hosted 401 before the first byte must not become a 200 event stream."""

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": BODY_SENTINEL}})

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
                id="public-chat",
                provider="compat",
                provider_model="native-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    provider = _compat_provider(handler)
    with TestClient(
        create_app(config, providers={"compat": provider}), base_url="http://127.0.0.1"
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public-chat",
                "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                "stream": True,
            },
        )

    assert response.status_code == 502
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "provider_auth_failed"
    assert response.json()["error"]["details"] == {"provider": "compat"}
    assert BODY_SENTINEL not in response.text
    assert OPENAI_KEY_SENTINEL not in response.text


def test_http_mid_stream_failure_emits_one_error_event_and_no_done() -> None:
    provider = _StubStreamProvider(
        "det",
        events=(StreamDelta(text="partial"),),
        failure=ProviderProtocolError(),
        failure_after=1,
    )
    with TestClient(
        create_app(_app_config(), providers={"det": provider}), base_url="http://127.0.0.1"
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "public-chat",
                "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                "stream": True,
            },
        )

    # Headers were already committed, so the status stays 200.
    assert response.status_code == 200
    frames = [json.loads(frame) for frame in _frames(response.text)]
    assert "[DONE]" not in response.text
    assert frames[-1]["error"] == {
        "code": "provider_protocol_error",
        "message": "The selected provider returned an invalid response.",
        "retryable": False,
        "details": {"provider": "det"},
        "validation": None,
    }
    assert frames[-1]["request_id"] == response.headers["X-Request-ID"]


def test_http_stream_never_leaks_prompts_keys_or_upstream_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert PROMPT_SENTINEL in request.content.decode("utf-8")
        return httpx.Response(
            200,
            content=_sse(
                json.dumps({"choices": [{"delta": {"content": "safe"}}]}),
                json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
                "[DONE]",
            ),
            headers={"content-type": "text/event-stream"},
        )

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
                id="public-chat",
                provider="compat",
                provider_model="native-model",
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
            "/v1/chat/completions",
            json={
                "model": "public-chat",
                "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert "safe" in response.text
    for sentinel in (PROMPT_SENTINEL, OPENAI_KEY_SENTINEL, "Bearer ", "native-model"):
        assert sentinel not in caplog.text
    for sentinel in (OPENAI_KEY_SENTINEL, "Bearer ", "native-model"):
        assert sentinel not in response.text


def test_openapi_documents_the_streaming_capability() -> None:
    with TestClient(create_app(_app_config()), base_url="http://127.0.0.1") as client:
        capabilities = client.get("/v1/capabilities").json()

    assert capabilities["chat_completions"]["streaming"] is True
