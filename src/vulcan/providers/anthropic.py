"""Native adapter for the Anthropic Messages API (text-only contract).

Translation rules (documented in docs/ARCHITECTURE.md):

- ``system`` messages from any position are concatenated, in order, into the
  top-level ``system`` parameter.
- Consecutive same-role user/assistant messages are merged so the transmitted
  conversation alternates strictly; the first turn must be ``user`` and is
  rejected locally otherwise instead of guessing at upstream behavior.
- ``max_tokens`` is mandatory upstream; absent client values use the
  provider's explicit ``default_max_tokens``.
- Anthropic accepts temperature 0..1 while Vulcan's contract allows 0..2;
  higher values are rejected locally rather than silently clamped.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vulcan.config import AnthropicProviderConfig
from vulcan.errors import (
    ProviderError,
    ProviderProtocolError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderEmbeddingRequest,
    ProviderEmbeddingResult,
    ProviderStreamEvent,
    ProviderTokenUsage,
    ProviderToolCall,
    StreamDelta,
    StreamEnd,
)
from vulcan.providers.http import (
    ANTHROPIC_VERSION,
    build_client,
    idle_bounded,
    iter_sse_payloads,
    open_response,
    raise_for_hosted_status,
    read_bounded_json,
    resolve_api_key,
    send_response,
)
from vulcan.readiness import RuntimeProbe

__all__ = ["ANTHROPIC_VERSION", "AnthropicProvider"]


class _AnthropicContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str
    text: str | None = None
    # Present only on a `tool_use` block. Anthropic sends the arguments as a
    # decoded object; the boundary serializes it so every provider yields one shape.
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None


class _AnthropicUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    input_tokens: int | None = None
    output_tokens: int | None = None


class _AnthropicMessageResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    role: Literal["assistant"]
    content: list[_AnthropicContentBlock] = Field(default_factory=list)
    stop_reason: str | None = None
    usage: _AnthropicUsage | None = None


class _AnthropicStreamDelta(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str | None = None
    text: str | None = None
    stop_reason: str | None = None
    # Arguments arrive as JSON fragments; only the concatenation parses.
    partial_json: str | None = None


class _AnthropicStreamMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    usage: _AnthropicUsage | None = None


class _AnthropicStreamError(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str | None = None


class _AnthropicStreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str
    index: int | None = None
    message: _AnthropicStreamMessage | None = None
    content_block: _AnthropicContentBlock | None = None
    delta: _AnthropicStreamDelta | None = None
    usage: _AnthropicUsage | None = None
    error: _AnthropicStreamError | None = None


def _tool_calls(
    blocks: list[_AnthropicContentBlock],
) -> tuple[ProviderToolCall, ...] | None:
    """Lift `tool_use` blocks out of the content list into the shared shape."""

    calls = [
        ProviderToolCall(
            id=block.id,
            name=block.name,
            arguments=json.dumps(block.input or {}, separators=(",", ":"), sort_keys=True),
        )
        for block in blocks
        if block.type == "tool_use" and block.id is not None and block.name is not None
    ]
    return tuple(calls) or None


def _finish_reason(stop_reason: str | None) -> Literal["stop", "length", "tool_calls"] | None:
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason in {"end_turn", "stop_sequence"}:
        return "stop"
    if stop_reason == "max_tokens":
        return "length"
    return None


def _usage(parsed: _AnthropicUsage | None) -> ProviderTokenUsage | None:
    """Token usage only when the upstream reported both counts, never invented."""

    if parsed is None:
        return None
    input_tokens = parsed.input_tokens
    output_tokens = parsed.output_tokens
    if (input_tokens is not None and input_tokens < 0) or (
        output_tokens is not None and output_tokens < 0
    ):
        raise ProviderProtocolError
    if input_tokens is None or output_tokens is None:
        return None
    return ProviderTokenUsage(prompt_tokens=input_tokens, completion_tokens=output_tokens)


# Extended-thinking blocks arrive intermittently beside a tool_use answer.
# They are known and skipped; they are not content and must never be joined
# into one.
_DISCARDED_BLOCK_TYPES: frozenset[str] = frozenset({"thinking", "redacted_thinking"})

_TOOL_CHOICE: dict[str, str] = {"auto": "auto", "required": "any", "none": "none"}


def _translate_messages(
    request: ProviderChatRequest,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split system text out and merge turns into a strict user/assistant alternation.

    A tool result is not a role in Anthropic's protocol: it is a `tool_result`
    content block inside a *user* turn, naming the `tool_use` it answers. It is
    therefore never merged into a neighbouring text turn.
    """

    system_parts = [message.content for message in request.messages if message.role == "system"]
    turns: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "system":
            continue
        if message.role == "tool":
            turns.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
            continue
        if not message.content.strip():
            # Anthropic 400s on empty text blocks; refuse locally with the
            # same treatment as the other translation guards, before any I/O.
            raise UnsupportedCapabilityError("empty_message_content")
        if turns and turns[-1]["role"] == message.role and isinstance(turns[-1]["content"], str):
            turns[-1]["content"] = f"{turns[-1]['content']}\n\n{message.content}"
        else:
            turns.append({"role": message.role, "content": message.content})
    if not turns or turns[0]["role"] != "user":
        raise UnsupportedCapabilityError("assistant_first_conversation")
    system = "\n\n".join(system_parts) if system_parts else None
    return system, turns


class AnthropicProvider:
    provider_type: Literal["anthropic"] = "anthropic"

    def __init__(
        self,
        provider_id: str,
        config: AnthropicProviderConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._api_key_env = config.api_key_env
        self._default_max_tokens = config.default_max_tokens
        self._stream_idle_seconds = config.stream_idle_timeout_seconds
        self._client = client or build_client(
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    def _payload(self, request: ProviderChatRequest, *, stream: bool) -> dict[str, Any]:
        """Translate one request; local-only guards run before any I/O."""

        if request.temperature is not None and request.temperature > 1.0:
            raise UnsupportedCapabilityError("temperature_above_one")
        system, turns = _translate_messages(request)
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "messages": turns,
            "max_tokens": (
                request.max_tokens if request.max_tokens is not None else self._default_max_tokens
            ),
        }
        if system is not None:
            payload["system"] = system
        if request.tools:
            # Anthropic names the JSON Schema `input_schema`, not `parameters`.
            payload["tools"] = [
                {
                    "name": tool.name,
                    **({"description": tool.description} if tool.description else {}),
                    "input_schema": tool.parameters or {"type": "object"},
                }
                for tool in request.tools
            ]
            if request.tool_choice is not None:
                payload["tool_choice"] = {"type": _TOOL_CHOICE[request.tool_choice]}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream"] = True
        return payload

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        payload = self._payload(request, stream=False)
        api_key = resolve_api_key(self._api_key_env)

        try:
            async with open_response(
                self._client,
                "POST",
                "/v1/messages",
                json_body=payload,
                headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            ) as response:
                if not response.is_success:
                    raise_for_hosted_status(response.status_code)
                try:
                    parsed = _AnthropicMessageResponse.model_validate(
                        await read_bounded_json(response)
                    )
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        # Text and tool_use are the answer; thinking is deliberately discarded,
        # never concatenated, because reasoning is not an answer here (the same
        # rule the OpenAI-compatible adapter applies to reasoning_content). A
        # genuinely unknown block is still a protocol error: silently dropping
        # one would misreport partial content as a complete answer.
        texts: list[str] = []
        for block in parsed.content:
            if block.type == "text" and block.text is not None:
                texts.append(block.text)
                continue
            if block.type == "tool_use":
                if block.id is None or block.name is None:
                    raise ProviderProtocolError
                continue
            if block.type in _DISCARDED_BLOCK_TYPES:
                continue
            raise ProviderProtocolError

        calls = _tool_calls(parsed.content)
        return ProviderChatResult(
            content="".join(texts),
            finish_reason="tool_calls" if calls else _finish_reason(parsed.stop_reason),
            usage=_usage(parsed.usage),
            tool_calls=calls,
        )

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        """Stream one Messages response, translating Anthropic's SSE events.

        The upstream connection is opened and classified before any event is
        yielded; closing this iterator closes the upstream response.
        """

        payload = self._payload(request, stream=True)
        api_key = resolve_api_key(self._api_key_env)
        finish_reason: Literal["stop", "length"] | None = None
        streaming_calls: dict[int | None, dict[str, str]] = {}
        discarded_blocks: set[int | None] = set()
        input_tokens: int | None = None
        output_tokens: int | None = None

        try:
            response = await send_response(
                self._client,
                "POST",
                "/v1/messages",
                json_body=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "Accept": "text/event-stream",
                },
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        try:
            if not response.is_success:
                # Body is never read: classification uses the status only.
                raise_for_hosted_status(response.status_code)
            async for data in idle_bounded(iter_sse_payloads(response), self._stream_idle_seconds):
                if not data:
                    continue
                try:
                    event = _AnthropicStreamEvent.model_validate(json.loads(data))
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc

                if event.type == "error":
                    # Classified from the event type only; text never escapes.
                    error_type = (event.error or _AnthropicStreamError()).type or ""
                    if "overloaded" in error_type:
                        raise ProviderUnavailableError
                    raise ProviderError(retryable=False)
                if event.type == "message_start":
                    if event.message is not None and event.message.usage is not None:
                        input_tokens = event.message.usage.input_tokens
                elif event.type == "content_block_start":
                    block = event.content_block
                    if block is None:
                        raise ProviderProtocolError
                    if block.type == "tool_use":
                        if block.id is None or block.name is None:
                            raise ProviderProtocolError
                        streaming_calls[event.index] = {
                            "id": block.id,
                            "name": block.name,
                            "arguments": "",
                        }
                        discarded_blocks.discard(event.index)
                    elif block.type in _DISCARDED_BLOCK_TYPES:
                        # Thinking is not the answer and is never yielded.
                        discarded_blocks.add(event.index)
                    elif block.type == "text":
                        if block.text:
                            yield StreamDelta(text=block.text)
                    else:
                        # An unknown block would silently truncate the reply.
                        raise ProviderProtocolError
                elif event.type == "content_block_delta":
                    if event.delta is None:
                        raise ProviderProtocolError
                    if event.delta.type == "input_json_delta":
                        call = streaming_calls.get(event.index)
                        if call is None:
                            raise ProviderProtocolError
                        call["arguments"] += event.delta.partial_json or ""
                    elif event.index in discarded_blocks:
                        continue
                    elif event.delta.type == "text_delta":
                        if event.delta.text:
                            yield StreamDelta(text=event.delta.text)
                    else:
                        raise ProviderProtocolError
                elif event.type == "message_delta":
                    if event.delta is not None and event.delta.stop_reason is not None:
                        finish_reason = _finish_reason(event.delta.stop_reason)
                    if event.usage is not None:
                        output_tokens = event.usage.output_tokens
                elif event.type == "message_stop":
                    break
            else:
                # The API closed without message_stop: truncated reply.
                raise ProviderProtocolError
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc
        finally:
            await response.aclose()

        streamed_calls = (
            tuple(
                ProviderToolCall(
                    id=call["id"], name=call["name"], arguments=call["arguments"] or "{}"
                )
                for _, call in sorted(streaming_calls.items(), key=lambda item: item[0] or 0)
            )
            or None
        )
        yield StreamEnd(
            finish_reason="tool_calls" if streamed_calls else finish_reason,
            usage=_usage(_AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens)),
            tool_calls=streamed_calls,
        )

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        """Anthropic publishes no embeddings API.

        Configuration rejects an embeddings-capable model on an anthropic
        provider at startup, so this is defence in depth rather than a
        reachable request path.
        """

        del request
        raise UnsupportedCapabilityError("embeddings")

    async def discover_runtime(self) -> RuntimeProbe:
        """Hosted models stay honestly unchecked until a real request uses them."""

        return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)

    async def aclose(self) -> None:
        await self._client.aclose()
