"""Provider boundary independent of the HTTP contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from vulcan.readiness import RuntimeProbe


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class ProviderChatRequest:
    runtime_model: str
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


class Provider(Protocol):
    @property
    def kind(self) -> Literal["ollama", "deterministic"]:
        """Stable provider identifier exposed as safe metadata."""
        ...

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        """Submit one non-streaming chat request."""

    async def discover_runtime(self) -> RuntimeProbe:
        """Probe provider readiness without inventing model inventory."""

    async def aclose(self) -> None:
        """Release provider resources."""
