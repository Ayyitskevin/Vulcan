"""Provider-independent gateway orchestration tests."""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest

from vulcan.config import Capability, DeterministicProviderConfig, ModelConfig
from vulcan.errors import (
    ConfigurationError,
    ModelNotFoundError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
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
    provider_type: Literal["deterministic"] = "deterministic"

    def __init__(
        self,
        provider_id: str = "test-provider",
        result: ProviderChatResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.calls: list[ProviderChatRequest] = []
        self.discover_calls = 0
        self.result = result or ProviderChatResult(content="unused", finish_reason="stop")
        self.failure = failure

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.result

    async def discover_runtime(self):
        from vulcan.readiness import RuntimeProbe

        self.discover_calls += 1
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        return None


def _registry(
    *,
    capabilities: frozenset[Capability],
    provider_id: str = "test-provider",
) -> ModelRegistry:
    return ModelRegistry(
        (
            ModelConfig(
                id="public-model",
                provider=provider_id,
                provider_model="provider-runtime-model",
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
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.CHAT})), {"test-provider": provider}
    )

    with pytest.raises(ModelNotFoundError) as caught:
        asyncio.run(gateway.chat(_request(model="missing-model")))

    assert caught.value.details == {"model": "missing-model"}
    assert provider.calls == []


def test_gateway_rejects_model_without_chat_capability_before_provider_call() -> None:
    provider = RecordingProvider()
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.EMBEDDINGS})), {"test-provider": provider}
    )

    with pytest.raises(UnsupportedCapabilityError) as caught:
        asyncio.run(gateway.chat(_request()))

    assert caught.value.details == {"capability": "chat", "model": "public-model"}
    assert provider.calls == []


def test_gateway_streaming_guard_runs_before_model_lookup_and_provider_call() -> None:
    provider = RecordingProvider()
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.CHAT})), {"test-provider": provider}
    )

    with pytest.raises(UnsupportedCapabilityError) as caught:
        asyncio.run(gateway.chat(_request(model="missing-model", stream=True)))

    assert caught.value.details == {"capability": "streaming", "model": "missing-model"}
    assert provider.calls == []


def test_gateway_assembles_deterministic_completion_response() -> None:
    provider = DeterministicProvider(
        "det",
        DeterministicProviderConfig(
            type="deterministic",
            response_text="deterministic answer",
        ),
    )
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.CHAT}), provider_id="det"),
        {"det": provider},
        clock=lambda: 1_725_000_000.9,
        id_factory=lambda: "chatcmpl-fixed",
    )

    response = asyncio.run(gateway.chat(_request()))

    assert response.model_dump(mode="json") == {
        "id": "chatcmpl-fixed",
        "object": "chat.completion",
        "created": 1_725_000_000,
        "model": "public-model",
        "provider": "det",
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
        result=ProviderChatResult(
            content="provider answer",
            finish_reason="length",
            usage=ProviderTokenUsage(prompt_tokens=7, completion_tokens=3),
        )
    )
    gateway = Gateway(
        _registry(capabilities=frozenset({Capability.CHAT})),
        {"test-provider": provider},
        clock=lambda: 20.1,
        id_factory=lambda: "chatcmpl-recorded",
    )

    response = asyncio.run(gateway.chat(_request()))

    assert provider.calls == [
        ProviderChatRequest(
            provider_model="provider-runtime-model",
            messages=(ProviderMessage(role="user", content="private prompt"),),
            temperature=0.4,
            max_tokens=17,
        )
    ]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 10


def test_gateway_routes_each_alias_to_exactly_its_configured_provider() -> None:
    alpha = RecordingProvider(
        "alpha", result=ProviderChatResult(content="from alpha", finish_reason="stop")
    )
    beta = RecordingProvider(
        "beta", result=ProviderChatResult(content="from beta", finish_reason="stop")
    )
    registry = ModelRegistry(
        (
            ModelConfig(
                id="alias-alpha",
                provider="alpha",
                provider_model="native-alpha",
                capabilities=frozenset({Capability.CHAT}),
            ),
            ModelConfig(
                id="alias-beta",
                provider="beta",
                provider_model="native-beta",
                capabilities=frozenset({Capability.CHAT}),
            ),
        )
    )
    gateway = Gateway(registry, {"alpha": alpha, "beta": beta})

    response = asyncio.run(gateway.chat(_request(model="alias-beta")))

    assert response.provider == "beta"
    assert response.choices[0].message.content == "from beta"
    assert [call.provider_model for call in beta.calls] == ["native-beta"]
    assert alpha.calls == []

    response = asyncio.run(gateway.chat(_request(model="alias-alpha")))
    assert response.provider == "alpha"
    assert [call.provider_model for call in alpha.calls] == ["native-alpha"]
    assert len(beta.calls) == 1


def test_gateway_never_falls_back_when_selected_provider_fails() -> None:
    failing = RecordingProvider("failing", failure=ProviderUnavailableError())
    healthy = RecordingProvider(
        "healthy", result=ProviderChatResult(content="never used", finish_reason="stop")
    )
    registry = ModelRegistry(
        (
            ModelConfig(
                id="alias-failing",
                provider="failing",
                provider_model="native-failing",
                capabilities=frozenset({Capability.CHAT}),
            ),
            ModelConfig(
                id="alias-healthy",
                provider="healthy",
                provider_model="native-healthy",
                capabilities=frozenset({Capability.CHAT}),
            ),
        )
    )
    gateway = Gateway(registry, {"failing": failing, "healthy": healthy})

    with pytest.raises(ProviderUnavailableError) as caught:
        asyncio.run(gateway.chat(_request(model="alias-failing")))

    assert caught.value.details == {"provider": "failing"}
    assert len(failing.calls) == 1
    assert healthy.calls == []  # the healthy provider must never see the request


def test_gateway_missing_provider_mapping_is_a_loud_configuration_error() -> None:
    provider = RecordingProvider("present")
    registry = _registry(capabilities=frozenset({Capability.CHAT}), provider_id="absent")
    gateway = Gateway(registry, {"present": provider})

    with pytest.raises(ConfigurationError) as caught:
        asyncio.run(gateway.chat(_request()))

    assert caught.value.details == {"provider": "absent"}
    assert provider.calls == []


def test_gateway_requires_at_least_one_provider() -> None:
    with pytest.raises(ValueError, match="at least one provider"):
        Gateway(_registry(capabilities=frozenset({Capability.CHAT})), {})
