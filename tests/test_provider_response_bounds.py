"""Provider-response trust-boundary tests over synthetic transports only."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Literal

import httpx
import pytest

import vulcan.providers.http as provider_http
from vulcan.config import (
    AnthropicProviderConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from vulcan.errors import ProviderProtocolError
from vulcan.providers.anthropic import AnthropicProvider
from vulcan.providers.base import ProviderChatRequest, ProviderMessage, ProviderStreamEvent
from vulcan.providers.http import (
    iter_bounded_bytes,
    iter_bounded_lines,
    read_bounded_json,
    verify_hosted_credential,
)
from vulcan.providers.ollama import OllamaProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider

MockHandler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


class _RecordingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.iterated = False
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterated = True
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _response(stream: _RecordingStream) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("GET", "https://provider.example/response"),
        stream=stream,
    )


def test_bounded_bytes_accepts_the_exact_limit_across_chunks() -> None:
    stream = _RecordingStream([b"ab", b"cd"])

    async def run() -> bytes:
        response = _response(stream)
        try:
            return b"".join([chunk async for chunk in iter_bounded_bytes(response, max_bytes=4)])
        finally:
            await response.aclose()

    assert asyncio.run(run()) == b"abcd"
    assert stream.closed is True


def test_bounded_bytes_refuses_the_first_byte_past_the_limit() -> None:
    stream = _RecordingStream([b"ab", b"cd", b"e"])

    async def run() -> None:
        response = _response(stream)
        try:
            _ = [chunk async for chunk in iter_bounded_bytes(response, max_bytes=4)]
        finally:
            await response.aclose()

    with pytest.raises(ProviderProtocolError):
        asyncio.run(run())
    assert stream.closed is True


def test_bounded_bytes_rejects_a_non_positive_limit() -> None:
    async def run() -> None:
        response = _response(_RecordingStream([b"x"]))
        try:
            _ = [chunk async for chunk in iter_bounded_bytes(response, max_bytes=0)]
        finally:
            await response.aclose()

    with pytest.raises(ValueError, match="max_bytes must be positive"):
        asyncio.run(run())


def test_bounded_json_parses_fragmented_utf8() -> None:
    stream = _RecordingStream([b'{"snow": "', "雪".encode(), b'"}'])

    async def run() -> Any:
        response = _response(stream)
        try:
            return await read_bounded_json(response, max_bytes=64)
        finally:
            await response.aclose()

    assert asyncio.run(run()) == {"snow": "雪"}


def test_bounded_json_leaves_invalid_json_for_the_adapter_to_classify() -> None:
    async def run() -> Any:
        response = _response(_RecordingStream([b"{not-json}"]))
        try:
            return await read_bounded_json(response, max_bytes=64)
        finally:
            await response.aclose()

    with pytest.raises(ValueError):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("chunks", "expected"),
    [
        ([b"a\nb\n"], ["a", "b"]),
        ([b"a\r\nb\r\n"], ["a", "b"]),
        ([b"a\rb\r"], ["a", "b"]),
        ([b"a\r", b"\nb\r", b"c"], ["a", "b", "c"]),
        ([b"\xef", b"\xbb\xbfdata: ", "雪\n".encode()], ["data: 雪"]),
        ([b"\n"], [""]),
        ([b"tail-without-newline"], ["tail-without-newline"]),
    ],
)
def test_bounded_lines_preserves_supported_framing(
    chunks: list[bytes], expected: list[str]
) -> None:
    async def run() -> list[str]:
        response = _response(_RecordingStream(chunks))
        try:
            return [line async for line in iter_bounded_lines(response, max_bytes=64)]
        finally:
            await response.aclose()

    assert asyncio.run(run()) == expected


def test_bounded_lines_counts_utf8_bytes_not_characters() -> None:
    async def run() -> list[str]:
        response = _response(_RecordingStream(["雪\n".encode()]))
        try:
            return [line async for line in iter_bounded_lines(response, max_bytes=3)]
        finally:
            await response.aclose()

    with pytest.raises(ProviderProtocolError):
        asyncio.run(run())


def test_bounded_lines_rejects_invalid_utf8() -> None:
    async def run() -> list[str]:
        response = _response(_RecordingStream([b"data: \xff\n"]))
        try:
            return [line async for line in iter_bounded_lines(response, max_bytes=64)]
        finally:
            await response.aclose()

    with pytest.raises(ProviderProtocolError):
        asyncio.run(run())


def test_credential_verification_never_reads_the_body_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VULCAN_BOUND_TEST_KEY", "synthetic-bound-credential")
    stream = _RecordingStream([b"body must not be read"])

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    config = OpenAICompatibleProviderConfig(
        type="openai_compatible",
        base_url="https://provider.example/v1",
        api_key_env="VULCAN_BOUND_TEST_KEY",
        timeout_seconds=1.0,
    )

    async def run() -> tuple[str, bool]:
        client = httpx.AsyncClient(
            base_url=config.base_url,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        try:
            verdict = await verify_hosted_credential(config, client=client)
            return verdict, client.is_closed
        finally:
            await client.aclose()

    assert asyncio.run(run()) == ("verified", True)
    assert stream.iterated is False
    assert stream.closed is True


def _request() -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model="native-model",
        messages=(ProviderMessage(role="user", content="bounded"),),
        temperature=None,
        max_tokens=8,
    )


def _provider(
    kind: Literal["anthropic", "compat", "ollama"], handler: MockHandler
) -> AnthropicProvider | OpenAICompatibleProvider | OllamaProvider:
    if kind == "anthropic":
        config = AnthropicProviderConfig(
            type="anthropic",
            base_url="https://provider.example",
            api_key_env="VULCAN_BOUND_TEST_KEY",
            timeout_seconds=1.0,
        )
        client = httpx.AsyncClient(
            base_url=config.base_url,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        return AnthropicProvider("anthropic", config, client=client)
    if kind == "compat":
        config = OpenAICompatibleProviderConfig(
            type="openai_compatible",
            base_url="https://provider.example/v1",
            api_key_env="VULCAN_BOUND_TEST_KEY",
            timeout_seconds=1.0,
        )
        client = httpx.AsyncClient(
            base_url=config.base_url,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        )
        return OpenAICompatibleProvider("compat", config, client=client)
    config = OllamaProviderConfig(
        type="ollama",
        base_url="http://127.0.0.1:11434",
        timeout_seconds=1.0,
    )
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return OllamaProvider("local-ollama", config, client=client)


@pytest.mark.parametrize("kind", ["anthropic", "compat", "ollama"])
def test_buffered_adapters_refuse_oversize_success_bodies_and_close_upstream(
    kind: Literal["anthropic", "compat", "ollama"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VULCAN_BOUND_TEST_KEY", "synthetic-bound-credential")
    monkeypatch.setattr(provider_http, "MAX_PROVIDER_RESPONSE_BYTES", 64)
    bodies = {
        "anthropic": {"role": "assistant", "content": [], "padding": "x" * 80},
        "compat": {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "padding": "x" * 80,
        },
        "ollama": {
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
            "padding": "x" * 80,
        },
    }
    stream = _RecordingStream([json.dumps(bodies[kind]).encode()])

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    provider = _provider(kind, handler)

    async def run() -> None:
        try:
            await provider.chat(_request())
        finally:
            await provider.aclose()

    with pytest.raises(ProviderProtocolError):
        asyncio.run(run())
    assert stream.closed is True


@pytest.mark.parametrize("kind", ["anthropic", "compat", "ollama"])
def test_streaming_adapters_refuse_oversize_feeds_and_close_upstream(
    kind: Literal["anthropic", "compat", "ollama"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VULCAN_BOUND_TEST_KEY", "synthetic-bound-credential")
    monkeypatch.setattr(provider_http, "MAX_PROVIDER_RESPONSE_BYTES", 64)
    if kind == "ollama":
        body = (
            json.dumps(
                {"message": {"role": "assistant", "content": "x" * 80}, "done": False}
            ).encode()
            + b"\n"
        )
    else:
        body = b"data: " + json.dumps({"padding": "x" * 80}).encode() + b"\n\n"
    stream = _RecordingStream([body])

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    provider = _provider(kind, handler)

    async def run() -> list[ProviderStreamEvent]:
        try:
            return [event async for event in provider.chat_stream(_request())]
        finally:
            await provider.aclose()

    with pytest.raises(ProviderProtocolError):
        asyncio.run(run())
    assert stream.closed is True
