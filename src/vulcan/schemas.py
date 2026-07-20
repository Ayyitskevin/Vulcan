"""Typed HTTP contract for Vulcan's v1 API."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vulcan.config import PUBLIC_MODEL_PATTERN, Capability


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(StrictSchema):
    role: MessageRole
    content: str = Field(strict=True, min_length=1, max_length=32768)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatCompletionRequest(StrictSchema):
    model: str = Field(strict=True, pattern=PUBLIC_MODEL_PATTERN)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=64)
    temperature: float | None = Field(default=None, strict=True, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, strict=True, ge=1, le=32768)
    stream: bool = Field(default=False, strict=True)

    @model_validator(mode="after")
    def require_user_message_and_bounded_input(self) -> Self:
        if not any(message.role is MessageRole.USER for message in self.messages):
            raise ValueError("at least one user message is required")
        if sum(len(message.content) for message in self.messages) > 65536:
            raise ValueError("combined message content exceeds 65536 characters")
        return self


Availability = Literal["available", "unavailable", "unchecked"]


class ProviderHealth(StrictSchema):
    kind: Literal["ollama", "deterministic"]
    availability: Availability


class HealthResponse(StrictSchema):
    status: Literal["ok"] = "ok"
    service: Literal["vulcan"] = "vulcan"
    api_version: Literal["v1"] = "v1"
    provider: ProviderHealth
    models_configured: int = Field(ge=0)


class DiscoveryMetadata(StrictSchema):
    source: Literal["configuration"] = "configuration"
    live: bool = False
    availability: Availability = "unchecked"


class ModelRecord(StrictSchema):
    id: str
    object: Literal["model"] = "model"
    provider: Literal["ollama", "deterministic"]
    capabilities: tuple[Capability, ...]
    availability: Availability = "unchecked"
    description: str | None = None


class ModelListResponse(StrictSchema):
    object: Literal["list"] = "list"
    discovery: DiscoveryMetadata
    data: tuple[ModelRecord, ...]


class ChatCapability(StrictSchema):
    supported: Literal[True] = True
    streaming: Literal[False] = False
    message_roles: tuple[MessageRole, ...] = (
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )


class CapabilitiesResponse(StrictSchema):
    api_version: Literal["v1"] = "v1"
    model_discovery: Literal["configuration"] = "configuration"
    callable_capabilities: tuple[Capability, ...] = (Capability.CHAT,)
    chat_completions: ChatCapability = Field(default_factory=ChatCapability)


class AssistantMessage(StrictSchema):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(StrictSchema):
    index: Literal[0] = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length"] | None


class TokenUsage(StrictSchema):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionResponse(StrictSchema):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(ge=0)
    model: str
    provider: Literal["ollama", "deterministic"]
    choices: tuple[ChatChoice, ...]
    usage: TokenUsage | None = None


class ValidationIssue(StrictSchema):
    path: str
    reason: str


class ErrorBody(StrictSchema):
    code: str
    message: str
    retryable: bool
    details: dict[str, str | int | bool] | None = None
    validation: tuple[ValidationIssue, ...] | None = None


class ErrorEnvelope(StrictSchema):
    error: ErrorBody
    request_id: str
