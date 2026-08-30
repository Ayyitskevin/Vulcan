"""Provider boundary independent of the HTTP contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from vulcan.readiness import RuntimeProbe

ProviderType = Literal["ollama", "anthropic", "openai_compatible", "deterministic"]


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    """One model-requested call, carried verbatim across the boundary.

    ``arguments`` stays the provider's own JSON string. Vulcan brokers the call
    and does not parse the caller's contract with its tools.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ProviderResponseFormat:
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ProviderTool:
    name: str
    description: str | None
    parameters: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderChatRequest:
    provider_model: str
    messages: tuple[ProviderMessage, ...]
    temperature: float | None
    max_tokens: int | None
    keep_alive: str | None = None
    tools: tuple[ProviderTool, ...] | None = None
    tool_choice: Literal["auto", "none", "required"] | None = None
    # None means unconstrained; a schema of None with json_object means "valid
    # JSON, any shape". Each adapter translates into its provider's vocabulary.
    response_format: ProviderResponseFormat | None = None


@dataclass(frozen=True, slots=True)
class ProviderTokenUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderChatResult:
    content: str
    finish_reason: Literal["stop", "length", "tool_calls"] | None
    usage: ProviderTokenUsage | None = None
    tool_calls: tuple[ProviderToolCall, ...] | None = None


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

    finish_reason: Literal["stop", "length", "tool_calls"] | None
    usage: ProviderTokenUsage | None = None
    # Tool calls arrive whole on the terminal event rather than as partial
    # deltas: a half-streamed arguments string is not something a caller can
    # execute, and reassembling one is the adapter's job, not the caller's.
    tool_calls: tuple[ProviderToolCall, ...] | None = None


ProviderStreamEvent = StreamDelta | StreamEnd


@dataclass(frozen=True, slots=True)
class ProviderEmbeddingRequest:
    provider_model: str
    inputs: tuple[str, ...]
    keep_alive: str | None = None


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
