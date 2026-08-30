"""Typed HTTP contract for Vulcan's v1 API."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from vulcan.config import PROVIDER_ID_PATTERN, PUBLIC_MODEL_PATTERN, SEAT_PATTERN, Capability

ProviderType = Literal["ollama", "anthropic", "openai_compatible", "deterministic"]

MAX_EMBEDDING_INPUTS = 64
MAX_EMBEDDING_INPUT_CHARS = 8192
MAX_EMBEDDING_COMBINED_CHARS = 65536

StrictText = Annotated[str, Field(strict=True)]
# Python's JSON parser accepts NaN/Infinity; a vector must never carry one.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(StrictSchema):
    role: MessageRole
    content: str = Field(strict=True, min_length=1, max_length=32768)
    # Set only on a tool result, naming the assistant tool call it answers.
    tool_call_id: str | None = Field(default=None, strict=True, max_length=128)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


TOOL_NAME_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class ToolFunction(StrictSchema):
    """A callable the caller is willing to execute on the model's behalf.

    ``parameters`` is a JSON Schema object passed to the provider unchanged. It
    is bounded but not interpreted here: Vulcan brokers the call, it does not
    validate the caller's own contract with its tools.
    """

    name: str = Field(strict=True, pattern=TOOL_NAME_PATTERN)
    description: str | None = Field(default=None, strict=True, max_length=1024)
    parameters: dict[str, Any] | None = None


class ToolDefinition(StrictSchema):
    type: Literal["function"] = "function"
    function: ToolFunction


class ToolCallFunction(StrictSchema):
    name: str = Field(strict=True, pattern=TOOL_NAME_PATTERN)
    arguments: str = Field(strict=True, max_length=32768)


class ToolCall(StrictSchema):
    id: str = Field(strict=True, min_length=1, max_length=128)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatCompletionRequest(StrictSchema):
    model: str = Field(strict=True, pattern=PUBLIC_MODEL_PATTERN)
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=64)
    temperature: float | None = Field(default=None, strict=True, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, strict=True, ge=1, le=32768)
    stream: bool = Field(default=False, strict=True)
    tools: tuple[ToolDefinition, ...] | None = Field(default=None, max_length=64)
    tool_choice: Literal["auto", "none", "required"] | None = None
    # Optional caller attribution for /v1/usage. Operator-chosen, non-secret,
    # never forwarded upstream (pinned by tests/test_seat.py sentinels).
    seat: str | None = Field(default=None, strict=True, pattern=SEAT_PATTERN)

    @model_validator(mode="after")
    def require_user_message_and_bounded_input(self) -> Self:
        if not any(message.role is MessageRole.USER for message in self.messages):
            raise ValueError("at least one user message is required")
        if sum(len(message.content) for message in self.messages) > 65536:
            raise ValueError("combined message content exceeds 65536 characters")
        for message in self.messages:
            if message.role is MessageRole.TOOL and message.tool_call_id is None:
                raise ValueError("a tool message requires tool_call_id")
            if message.role is not MessageRole.TOOL and message.tool_call_id is not None:
                raise ValueError("tool_call_id is only valid on a tool message")
        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice requires tools")
        if self.tools is not None:
            names = [definition.function.name for definition in self.tools]
            if len(set(names)) != len(names):
                raise ValueError("tool names must be unique")
        return self


class EmbeddingsRequest(StrictSchema):
    model: str = Field(strict=True, pattern=PUBLIC_MODEL_PATTERN)
    input: StrictText | tuple[StrictText, ...]
    # Optional caller attribution for /v1/usage. Operator-chosen, non-secret,
    # never forwarded upstream (pinned by tests/test_seat.py sentinels).
    seat: str | None = Field(default=None, strict=True, pattern=SEAT_PATTERN)

    @property
    def inputs(self) -> tuple[str, ...]:
        """The request's inputs as a tuple, whether one string or many."""

        return (self.input,) if isinstance(self.input, str) else self.input

    @model_validator(mode="after")
    def inputs_must_be_bounded_and_nonblank(self) -> Self:
        items = self.inputs
        if not items:
            raise ValueError("at least one embedding input is required")
        if len(items) > MAX_EMBEDDING_INPUTS:
            raise ValueError(f"at most {MAX_EMBEDDING_INPUTS} embedding inputs are allowed")
        for item in items:
            if not item.strip():
                raise ValueError("embedding input must not be blank")
            if len(item) > MAX_EMBEDDING_INPUT_CHARS:
                raise ValueError(f"embedding input exceeds {MAX_EMBEDDING_INPUT_CHARS} characters")
        if sum(len(item) for item in items) > MAX_EMBEDDING_COMBINED_CHARS:
            raise ValueError(
                f"combined embedding input exceeds {MAX_EMBEDDING_COMBINED_CHARS} characters"
            )
        return self


class EmbeddingRecord(StrictSchema):
    object: Literal["embedding"] = "embedding"
    index: int = Field(ge=0)
    embedding: tuple[FiniteFloat, ...]


class EmbeddingUsage(StrictSchema):
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class EmbeddingsResponse(StrictSchema):
    object: Literal["list"] = "list"
    model: str
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    data: tuple[EmbeddingRecord, ...]
    usage: EmbeddingUsage | None = None


Availability = Literal["available", "unavailable", "unchecked"]


class ProviderStatus(StrictSchema):
    """One configured provider instance's honest readiness annotation."""

    id: str = Field(pattern=PROVIDER_ID_PATTERN)
    type: ProviderType
    availability: Availability


class HealthResponse(StrictSchema):
    status: Literal["ok"] = "ok"
    service: Literal["vulcan"] = "vulcan"
    api_version: Literal["v1"] = "v1"
    providers: tuple[ProviderStatus, ...]
    models_configured: int = Field(ge=0)


class DiscoveryMetadata(StrictSchema):
    source: Literal["configuration"] = "configuration"


class ModelRecord(StrictSchema):
    # populate_by_name lets constructors use the keyword-safe attribute name
    # while the API contract keeps the public key `class`.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    object: Literal["model"] = "model"
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    provider_type: ProviderType
    capabilities: tuple[Capability, ...]
    availability: Availability = "unchecked"
    description: str | None = None
    class_: str | None = Field(default=None, alias="class")


class ModelListResponse(StrictSchema):
    object: Literal["list"] = "list"
    discovery: DiscoveryMetadata
    data: tuple[ModelRecord, ...]


class ChatCapability(StrictSchema):
    supported: Literal[True] = True
    streaming: Literal[True] = True
    message_roles: tuple[MessageRole, ...] = (
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    )


class EmbeddingsCapability(StrictSchema):
    supported: Literal[True] = True
    max_inputs: int = MAX_EMBEDDING_INPUTS
    max_input_characters: int = MAX_EMBEDDING_INPUT_CHARS


class CapabilitiesResponse(StrictSchema):
    api_version: Literal["v1"] = "v1"
    model_discovery: Literal["configuration"] = "configuration"
    # Derived from the configured models by the route, never a static claim:
    # the embeddings block is present only when an alias actually declares it.
    callable_capabilities: tuple[Capability, ...]
    chat_completions: ChatCapability = Field(default_factory=ChatCapability)
    embeddings: EmbeddingsCapability | None = None


class AssistantMessage(StrictSchema):
    role: Literal["assistant"] = "assistant"
    content: str
    tool_calls: tuple[ToolCall, ...] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_tool_calls(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Keep a tool-free reply byte-identical to what callers received before.

        ``usage: null`` is part of this contract elsewhere, so a blanket
        exclude-none would break it. Only this field is dropped, and only when
        the model requested no calls.
        """

        data = handler(self)
        if data.get("tool_calls") is None:
            data.pop("tool_calls", None)
        return data


class ChatChoice(StrictSchema):
    index: Literal[0] = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None


class TokenUsage(StrictSchema):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatCompletionResponse(StrictSchema):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(ge=0)
    model: str
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    choices: tuple[ChatChoice, ...]
    usage: TokenUsage | None = None


class ChunkDelta(StrictSchema):
    """Incremental assistant output; absent fields are omitted when serialized."""

    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_tool_calls(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Drop `tool_calls` when the model requested none.

        The HTTP frame builder already strips every absent delta field; doing it
        here too keeps a delta dumped directly — as tests and any non-HTTP
        consumer do — the same shape it was before tool calling existed.
        """

        data = handler(self)
        if data.get("tool_calls") is None:
            data.pop("tool_calls", None)
        return data


class ChatCompletionChunkChoice(StrictSchema):
    index: Literal[0] = 0
    delta: ChunkDelta
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionChunk(StrictSchema):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(ge=0)
    model: str
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    choices: tuple[ChatCompletionChunkChoice, ...]
    usage: TokenUsage | None = None


class UsageTotalsRecord(StrictSchema):
    """Token totals are only interpretable alongside ``requests_with_usage``:
    upstreams that omit counts contribute a request but no tokens."""

    requests: int = Field(ge=0)
    requests_with_usage: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelUsageRecord(StrictSchema):
    model: str
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    totals: UsageTotalsRecord


class ProviderUsageRecord(StrictSchema):
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    totals: UsageTotalsRecord


class SeatUsageRecord(StrictSchema):
    seat: str = Field(pattern=SEAT_PATTERN)
    totals: UsageTotalsRecord


class LedgerRecord(StrictSchema):
    """Honesty counters for the durable ledger, present only when enabled."""

    replayed_requests: int = Field(ge=0)
    skipped_lines: int = Field(ge=0)
    write_failures: int = Field(ge=0)


class SeatBudgetRecord(StrictSchema):
    """Operator-visible budget state for one seat, present when budgets are on."""

    seat: str = Field(pattern=SEAT_PATTERN)
    hosted_tokens_per_day: int | None = Field(default=None, ge=0)
    hosted_requests_per_day: int | None = Field(default=None, ge=0)
    tokens_today: int = Field(ge=0)
    requests_today: int = Field(ge=0)
    window_resets_at: int = Field(ge=0)


class UsageResponse(StrictSchema):
    object: Literal["usage"] = "usage"
    # "process": in-memory counters since this process started (the default).
    # "ledger": counters replayed from the durable ledger plus this process.
    scope: Literal["process", "ledger"] = "process"
    started_at: int = Field(ge=0)
    totals: UsageTotalsRecord
    by_model: tuple[ModelUsageRecord, ...]
    by_provider: tuple[ProviderUsageRecord, ...]
    by_seat: tuple[SeatUsageRecord, ...]
    # Requests whose seat label arrived after the recorder's cardinality cap
    # filled: counted here, deliberately absent from by_seat.
    untracked_seat_requests: int = Field(default=0, ge=0)
    ledger: LedgerRecord | None = None
    budgets: tuple[SeatBudgetRecord, ...] | None = None


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
