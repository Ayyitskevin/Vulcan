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
    ProviderStreamEvent,
    ProviderTokenUsage,
    StreamDelta,
    StreamEnd,
)
from vulcan.providers.http import (
    build_client,
    iter_sse_payloads,
    raise_for_hosted_status,
    resolve_api_key,
)
from vulcan.readiness import RuntimeProbe

# Wire-format version of the Messages API, not a model choice.
ANTHROPIC_VERSION = "2023-06-01"


class _AnthropicContentBlock(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str
    text: str | None = None


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


class _AnthropicStreamMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    usage: _AnthropicUsage | None = None


class _AnthropicStreamError(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str | None = None


class _AnthropicStreamEvent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    type: str
    message: _AnthropicStreamMessage | None = None
    content_block: _AnthropicContentBlock | None = None
    delta: _AnthropicStreamDelta | None = None
    usage: _AnthropicUsage | None = None
    error: _AnthropicStreamError | None = None


def _finish_reason(stop_reason: str | None) -> Literal["stop", "length"] | None:
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


def _translate_messages(
    request: ProviderChatRequest,
) -> tuple[str | None, list[dict[str, str]]]:
    """Split system text out and merge turns into a strict user/assistant alternation."""

    system_parts = [message.content for message in request.messages if message.role == "system"]
    turns: list[dict[str, str]] = []
    for message in request.messages:
        if message.role == "system":
            continue
        if turns and turns[-1]["role"] == message.role:
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
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream"] = True
        return payload

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        payload = self._payload(request, stream=False)
        api_key = resolve_api_key(self._api_key_env)

        try:
            response = await self._client.post(
                "/v1/messages",
                json=payload,
                headers={"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        if not response.is_success:
            raise_for_hosted_status(response.status_code)

        try:
            parsed = _AnthropicMessageResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ProviderProtocolError from exc

        # Vulcan requests no tools, so only text blocks are a valid reply;
        # silently dropping an unknown block would misreport partial content.
        texts: list[str] = []
        for block in parsed.content:
            if block.type != "text" or block.text is None:
                raise ProviderProtocolError
            texts.append(block.text)

        return ProviderChatResult(
            content="".join(texts),
            finish_reason=_finish_reason(parsed.stop_reason),
            usage=_usage(parsed.usage),
        )

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        """Stream one Messages response, translating Anthropic's SSE events.

        The upstream connection is opened and classified before any event is
        yielded; closing this iterator closes the upstream response.
        """

        payload = self._payload(request, stream=True)
        api_key = resolve_api_key(self._api_key_env)
        finish_reason: Literal["stop", "length"] | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None

        # An explicit send/close pair (rather than the stream() context manager)
        # keeps the upstream response closable when the consumer abandons this
        # generator mid-stream, e.g. on client disconnect.
        upstream = self._client.build_request(
            "POST",
            "/v1/messages",
            json=payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Accept": "text/event-stream",
            },
        )
        try:
            response = await self._client.send(upstream, stream=True)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        try:
            if not response.is_success:
                # Body is never read: classification uses the status only.
                raise_for_hosted_status(response.status_code)
            async for data in iter_sse_payloads(response):
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
                    # Vulcan requests no tools or thinking: any other block
                    # type would silently truncate the visible reply.
                    if event.content_block is None or event.content_block.type != "text":
                        raise ProviderProtocolError
                    if event.content_block.text:
                        yield StreamDelta(text=event.content_block.text)
                elif event.type == "content_block_delta":
                    if event.delta is None or event.delta.type != "text_delta":
                        raise ProviderProtocolError
                    if event.delta.text:
                        yield StreamDelta(text=event.delta.text)
                elif event.type == "message_delta":
                    if event.delta is not None and event.delta.stop_reason is not None:
                        finish_reason = _finish_reason(event.delta.stop_reason)
                    if event.usage is not None:
                        output_tokens = event.usage.output_tokens
                elif event.type == "message_stop":
                    break
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc
        finally:
            await response.aclose()

        yield StreamEnd(
            finish_reason=finish_reason,
            usage=_usage(_AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens)),
        )

    async def discover_runtime(self) -> RuntimeProbe:
        """Hosted models stay honestly unchecked until a real request uses them."""

        return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)

    async def aclose(self) -> None:
        await self._client.aclose()
