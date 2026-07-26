"""Explicit no-I/O provider for tests and local contract checks."""

from collections.abc import AsyncIterator
from typing import Literal

from vulcan.config import DeterministicProviderConfig
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

# A fixed, obviously synthetic unit vector: this adapter performs no inference.
DETERMINISTIC_EMBEDDING = (0.125,) * 8


class DeterministicProvider:
    provider_type: Literal["deterministic"] = "deterministic"

    def __init__(self, provider_id: str, config: DeterministicProviderConfig) -> None:
        self.provider_id = provider_id
        self._response_text = config.response_text

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        del request
        return ProviderChatResult(content=self._response_text, finish_reason="stop")

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        del request
        yield StreamDelta(text=self._response_text)
        yield StreamEnd(finish_reason="stop", usage=None)

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        return ProviderEmbeddingResult(
            vectors=tuple(DETERMINISTIC_EMBEDDING for _ in request.inputs),
            usage=None,
        )

    async def discover_runtime(self) -> RuntimeProbe:
        # In-process and non-network: known ready, but there is no external
        # runtime inventory to reconcile against (live stays false).
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        return None
