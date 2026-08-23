"""The in-flight concurrency bound over the inference endpoints.

With ``[server] max_concurrent_requests`` set, the gateway admits at most N
in-flight inference requests and hard-rejects the rest with a typed 503
(``gateway_overloaded``) — backpressure, never a queue, retry, or reroute.
Liveness and discovery endpoints stay ungated. Most tests drive the Gateway
directly with a controllable provider; one test pins the HTTP envelope.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
)
from vulcan.errors import GatewayOverloadedError, ProviderUnavailableError
from vulcan.gateway import Gateway
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderEmbeddingRequest,
    ProviderEmbeddingResult,
    ProviderStreamEvent,
    StreamDelta,
    StreamEnd,
)
from vulcan.readiness import RuntimeProbe
from vulcan.registry import ModelRegistry
from vulcan.schemas import ChatCompletionRequest, ChatMessage, EmbeddingsRequest, MessageRole


class ControllableProvider:
    """A provider whose stream holds its admission slot until released."""

    provider_type: Literal["deterministic"] = "deterministic"
    provider_id = "test-provider"

    def __init__(self) -> None:
        self.release_stream = False
        self.stream_started = False
        self.fail_chat = False
        self.chat_calls = 0
        self.closed = False

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.chat_calls += 1
        if self.fail_chat:
            raise ProviderUnavailableError
        return ProviderChatResult(content="buffered-ok", finish_reason="stop")

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        self.stream_started = True
        yield StreamDelta(text="first")
        while not self.release_stream:
            await asyncio.sleep(0.005)
        yield StreamEnd(finish_reason="stop")

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        return ProviderEmbeddingResult(vectors=((0.5,),))

    async def discover_runtime(self) -> RuntimeProbe:
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        self.closed = True


def _gateway(provider: ControllableProvider, bound: int | None) -> Gateway:
    registry = ModelRegistry(
        (
            ModelConfig(
                id="public-model",
                provider="test-provider",
                provider_model="provider-runtime-model",
                capabilities=frozenset({Capability.CHAT, Capability.EMBEDDINGS}),
            ),
        )
    )
    return Gateway(
        registry,
        {"test-provider": provider},
        max_concurrent_requests=bound,
    )


def _chat_request(*, stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="public-model",
        messages=(ChatMessage(role=MessageRole.USER, content="private prompt"),),
        stream=stream,
    )


def _embed_request() -> EmbeddingsRequest:
    return EmbeddingsRequest(model="public-model", input=("private input",))


def test_saturation_rejects_chat_and_embed_immediately() -> None:
    provider = ControllableProvider()
    gateway = _gateway(provider, 1)

    async def run() -> None:
        events = await _hold_stream(gateway)
        with pytest.raises(GatewayOverloadedError):
            await gateway.chat(_chat_request())
        with pytest.raises(GatewayOverloadedError):
            await gateway.embed(_embed_request())
        # The rejections never reached the provider.
        assert provider.chat_calls == 0
        provider.release_stream = True
        await _drain(events)

    asyncio.run(run())


async def _hold_stream(gateway: Gateway):
    """Start a stream and hold it mid-flight; returns the live iterator."""

    events = gateway.chat_stream(_chat_request(stream=True)).__aiter__()
    first = await events.__anext__()
    # The opening role chunk proves the stream is admitted and mid-flight.
    assert first.choices[0].delta.role == "assistant"
    return events


async def _drain(events) -> None:
    async for _ in events:
        pass


def test_slot_released_on_clean_stream_completion() -> None:
    provider = ControllableProvider()
    gateway = _gateway(provider, 1)

    async def run() -> None:
        events = await _hold_stream(gateway)
        provider.release_stream = True
        await _drain(events)
        result = await gateway.chat(_chat_request())
        assert result.choices[0].message.content == "buffered-ok"

    asyncio.run(run())


def test_slot_released_on_mid_stream_abandon() -> None:
    provider = ControllableProvider()
    gateway = _gateway(provider, 1)

    async def run() -> None:
        events = await _hold_stream(gateway)
        await events.aclose()  # the client went away mid-stream
        result = await gateway.chat(_chat_request())
        assert result.choices[0].message.content == "buffered-ok"

    asyncio.run(run())


def test_slot_released_on_provider_error() -> None:
    provider = ControllableProvider()
    provider.fail_chat = True
    gateway = _gateway(provider, 1)

    async def run() -> None:
        with pytest.raises(ProviderUnavailableError):
            await gateway.chat(_chat_request())
        provider.fail_chat = False
        result = await gateway.chat(_chat_request())
        assert result.choices[0].message.content == "buffered-ok"

    asyncio.run(run())


def test_no_bound_means_unbounded_admission() -> None:
    provider = ControllableProvider()
    gateway = _gateway(provider, None)

    async def run() -> None:
        first = await _hold_stream(gateway)
        second = await _hold_stream(gateway)  # a second in-flight stream is fine
        provider.release_stream = True
        await _drain(first)
        await _drain(second)

    asyncio.run(run())


def test_constructor_rejects_a_non_positive_bound() -> None:
    provider = ControllableProvider()
    with pytest.raises(ValueError, match="max_concurrent_requests"):
        _gateway(provider, 0)


def test_overloaded_envelope_and_ungated_liveness_endpoints() -> None:
    """HTTP level: typed 503 envelope under saturation; liveness stays open.

    The holding request runs on a second thread with its own TestClient: this
    TestClient/httpx2 stack completes a streaming response only when the body
    finishes, so a held stream in the main thread would deadlock the test.
    """

    provider = ControllableProvider()
    config = GatewayConfig(
        schema_version=2,
        server={
            "host": "127.0.0.1",
            "port": 8140,
            "max_concurrent_requests": 1,
        },
        providers={
            "test-provider": DeterministicProviderConfig(
                type="deterministic",
                response_text="unused",
            )
        },
        models=(
            ModelConfig(
                id="public-model",
                provider="test-provider",
                provider_model="provider-runtime-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    app = create_app(config, providers={"test-provider": provider})
    holder_errors: list[Exception] = []

    def hold_stream() -> None:
        try:
            with TestClient(app, base_url="http://127.0.0.1") as holder:
                response = holder.post(
                    "/v1/chat/completions",
                    json={
                        "model": "public-model",
                        "messages": [{"role": "user", "content": "private prompt"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200
        except Exception as exc:  # surfaced in the main thread
            holder_errors.append(exc)

    thread = threading.Thread(target=hold_stream)
    thread.start()
    for _ in range(400):
        if provider.stream_started:
            break
        time.sleep(0.005)
    assert provider.stream_started, "holding stream never reached the provider"

    try:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            rejected = client.post(
                "/v1/chat/completions",
                json={
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "private prompt"}],
                },
            )
            assert rejected.status_code == 503
            body = rejected.json()
            assert body["error"]["code"] == "gateway_overloaded"
            assert body["error"]["retryable"] is True
            # Liveness and discovery are never gated by the inference bound.
            assert client.get("/healthz").status_code == 200
            assert client.get("/v1/models").status_code == 200
            assert client.get("/v1/usage").status_code == 200
    finally:
        provider.release_stream = True
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert not holder_errors
