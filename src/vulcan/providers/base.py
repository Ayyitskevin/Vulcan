"""Provider boundary independent of the HTTP contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from vulcan.readiness import RuntimeProbe

ProviderType = Literal["ollama", "anthropic", "openai_compatible", "deterministic"]


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ProviderChatRequest:
    provider_model: str
    messages: tuple[ProviderMessage, ...]
    temperature: float | None
    max_tokens: int | None


@dataclass(frozen=True, slots=True)
class ProviderTokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderChatResult:
    content: str
    finish_reason: Literal["stop", "length"] | None
    usage: ProviderTokenUsage | None = None


@dataclass(frozen=True, slots=True)
class StreamDelta:
    """One incremental piece of assistant text."""

    text: str


@dataclass(frozen=True, slots=True)
class StreamEnd:
    """Terminal event of a provider stream.

    Adapters must emit exactly one of these on clean completion; a stream that
    ends without it is treated as a truncated (protocol-error) response.
    """

    finish_reason: Literal["stop", "length"] | None
    usage: ProviderTokenUsage | None = None


ProviderStreamEvent = StreamDelta | StreamEnd


@dataclass(frozen=True, slots=True)
class ProviderEmbeddingRequest:
    provider_model: str
    inputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProviderEmbeddingUsage:
    """Embeddings have no completion tokens, so total mirrors prompt unless the
    upstream reports its own total."""

    prompt_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderEmbeddingResult:
    """One vector per input, in input order."""

    vectors: tuple[tuple[float, ...], ...]
    usage: ProviderEmbeddingUsage | None = None


class Provider(Protocol):
    @property
    def provider_id(self) -> str:
        """Configured provider instance ID exposed as safe metadata."""
        ...

    @property
    def provider_type(self) -> ProviderType:
        """Stable adapter type exposed as safe metadata."""
        ...

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        """Submit one non-streaming chat request."""

    def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        """Submit one streaming chat request.

        Implementations open the upstream connection and map non-success
        statuses to Vulcan errors *before* yielding the first event, so the
        gateway can still answer with a normal JSON error envelope. Closing
        the returned iterator must release the upstream response.
        """

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        """Embed one batch of inputs, returning one vector per input in order."""

    async def discover_runtime(self) -> RuntimeProbe:
        """Probe provider readiness without inventing model inventory."""

    async def aclose(self) -> None:
        """Release provider resources."""
