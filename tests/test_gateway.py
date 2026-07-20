"""Provider-independent gateway orchestration tests."""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from vulcan.config import Capability, DeterministicProviderConfig, ModelConfig
from vulcan.errors import ModelNotFoundError, UnsupportedCapabilityError
from vulcan.gateway import Gateway
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderMessage,
    ProviderTokenUsage,
)
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.registry import ModelRegistry
from vulcan.schemas import ChatCompletionRequest, ChatMessage, MessageRole


class RecordingProvider:
    kind: Literal["deterministic"] = "deterministic"

    def __init__(self, result: ProviderChatResult | None = None) -> None:
        self.calls: list[ProviderChatRequest] = []
        self.result = result or ProviderChatResult(content="unused", finish_reason="stop")

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.calls.append(request)
        return self.result

    async def aclose(self) -> None:
        return None


def _registry(*, capabilities: frozenset[Capability]) -> ModelRegistry:
    return ModelRegistry(
        (
            ModelConfig(
                id="public-model",
                runtime_name="provider-runtime-model",
                capabilities=capabilities,
            ),
        )
    )


def _request(*, model: str = "public-model", stream: bool = False) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=model,
        messages=(ChatMessage(role=MessageRole.USER, content="private prompt"),),
        temperature=0.4,
        max_tokens=17,
        stream=stream,
    )


def test_gateway_rejects_unknown_model_without_calling_provider() -> None:
    provider = RecordingProvider()
    gateway = Gateway(_registry(capabilities=frozenset({Capability.CHAT})), provider)

    with pytest.raises(ModelNotFoundError) as caught:
        asyncio.run(gateway.chat(_request(model="missing-model")))

    assert caught.value.details == {"model": "missing-model"}
    assert provider.calls == []


def test_gateway_rejects_model_without_chat_capability_before_provider_call() -> None:
    provider = RecordingProvider()
    gateway = Gateway(_registry(capabilities=frozenset({Capability.EMBEDDINGS})), provider)

    with pytest.raises(UnsupportedCapabilityError) as caught:
        asyncio.run(gateway.chat(_request()))

    assert caught.value.details == {"capability": "chat", "model": "public-model"}
    assert provider.calls == []


def test_gateway_streaming_guard_runs_before_model_lookup_and_provider_call() -> None:
    provider = RecordingProvider()
    gateway = Gateway(_registry(capabilities=frozenset({Capability.CHAT})), provider)

    with pytest.raises(UnsupportedCapabilityError) as caught:
        asyncio.run(gateway.chat(_request(model="missing-model", stream=True)))

    assert caught.value.details == {"capability": "streaming", "model": "missing-model"}
    assert provider.calls == []


def test_gateway_assembles_deterministic_completion_response() -> None:
    provider = DeterministicProvider(
        DeterministicProviderConfig(
            kind="deterministic",
            response_text="deterministic answer",
        )
    )
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.CHAT})),
        provider,
        clock=lambda: 1_725_000_000.9,
        id_factory=lambda: "chatcmpl-fixed",
    )

    response = asyncio.run(gateway.chat(_request()))

    assert response.model_dump(mode="json") == {
        "id": "chatcmpl-fixed",
        "object": "chat.completion",
        "created": 1_725_000_000,
        "model": "public-model",
        "provider": "deterministic",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "deterministic answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": None,
    }


def test_gateway_maps_public_request_to_provider_boundary_and_totals_usage() -> None:
    provider = RecordingProvider(
        ProviderChatResult(
            content="provider answer",
            finish_reason="length",
            usage=ProviderTokenUsage(prompt_tokens=7, completion_tokens=3),
        )
    )
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.CHAT})),
        provider,
        clock=lambda: 20.1,
        id_factory=lambda: "chatcmpl-recorded",
    )

    response = asyncio.run(gateway.chat(_request()))

    assert provider.calls == [
        ProviderChatRequest(
            runtime_model="provider-runtime-model",
            messages=(ProviderMessage(role="user", content="private prompt"),),
            temperature=0.4,
            max_tokens=17,
        )
    ]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 10
