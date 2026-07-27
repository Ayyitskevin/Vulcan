"""One adapter for OpenAI and every OpenAI-compatible chat-completions vendor.

Vendor differences (base URL, credential variable, token-limit field name) are
configuration; the wire contract here is the stable common subset: one
non-streaming ``POST {base_url}/chat/completions`` with Bearer auth. Unknown
response fields are ignored because vendors extend the shape freely, but the
fields Vulcan relies on are validated strictly and never coerced.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vulcan.config import OpenAICompatibleProviderConfig
from vulcan.errors import (
    ProviderProtocolError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderEmbeddingRequest,
    ProviderEmbeddingResult,
    ProviderEmbeddingUsage,
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

SSE_DONE = "[DONE]"


class _CompatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    role: Literal["assistant"]
    content: str


class _CompatChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: _CompatMessage
    finish_reason: str | None = None


class _CompatUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _CompatChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    choices: list[_CompatChoice] = Field(min_length=1)
    usage: _CompatUsage | None = None


class _CompatStreamDelta(BaseModel):
    # Vendor extensions (e.g. DeepSeek reasoning_content) are ignored, not errors.
    model_config = ConfigDict(extra="ignore", strict=True)

    content: str | None = None


class _CompatStreamChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    delta: _CompatStreamDelta | None = None
    finish_reason: str | None = None


class _CompatStreamChunk(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    # A usage-only trailer chunk legitimately carries no choices.
    choices: list[_CompatStreamChoice] = Field(default_factory=list)
    usage: _CompatUsage | None = None


class _CompatEmbeddingRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    # allow_inf_nan=False: Python's JSON parser accepts NaN/Infinity, which must
    # never reach a client as a "vector".
    embedding: list[Annotated[float, Field(allow_inf_nan=False)]]
    index: int | None = None


class _CompatEmbeddingUsage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: int | None = None
    total_tokens: int | None = None


class _CompatEmbeddingsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    data: list[_CompatEmbeddingRecord] = Field(min_length=1)
    usage: _CompatEmbeddingUsage | None = None


def _finish_reason(value: str | None) -> Literal["stop", "length"] | None:
    if value == "stop":
        return "stop"
    if value == "length":
        return "length"
    return None


def _usage(parsed: _CompatUsage | None) -> ProviderTokenUsage | None:
    """Token usage only when the upstream reported both counts, never invented."""

    if parsed is None:
        return None
    prompt_tokens = parsed.prompt_tokens
    completion_tokens = parsed.completion_tokens
    if (prompt_tokens is not None and prompt_tokens < 0) or (
        completion_tokens is not None and completion_tokens < 0
    ):
        raise ProviderProtocolError
    if prompt_tokens is None or completion_tokens is None:
        return None
    return ProviderTokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


class OpenAICompatibleProvider:
    provider_type: Literal["openai_compatible"] = "openai_compatible"

    def __init__(
        self,
        provider_id: str,
        config: OpenAICompatibleProviderConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._api_key_env = config.api_key_env
        self._max_tokens_field = config.max_tokens_field
        self._client = client or build_client(
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    def _payload(self, request: ProviderChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload[self._max_tokens_field] = request.max_tokens
        # stream_options.include_usage is deliberately not sent: support varies
        # across compatible vendors. Usage is taken when a chunk provides it.
        return payload

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        api_key = resolve_api_key(self._api_key_env)
        payload = self._payload(request, stream=False)

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        if not response.is_success:
            raise_for_hosted_status(response.status_code)

        try:
            parsed = _CompatChatResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ProviderProtocolError from exc

        choice = parsed.choices[0]
        return ProviderChatResult(
            content=choice.message.content,
            finish_reason=_finish_reason(choice.finish_reason),
            usage=_usage(parsed.usage),
        )

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        """Stream one chat completion over SSE.

        The upstream connection is opened and its status classified before any
        event is yielded, so pre-stream failures still surface as ordinary
        Vulcan errors. Closing this iterator closes the upstream response.
        """

        api_key = resolve_api_key(self._api_key_env)
        payload = self._payload(request, stream=True)
        finish_reason: Literal["stop", "length"] | None = None
        usage: ProviderTokenUsage | None = None

        # An explicit send/close pair (rather than the stream() context manager)
        # keeps the upstream response closable when the consumer abandons this
        # generator mid-stream, e.g. on client disconnect.
        upstream = self._client.build_request(
            "POST",
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "text/event-stream"},
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
                if data == SSE_DONE:
                    break
                if not data:
                    continue
                try:
                    chunk = _CompatStreamChunk.model_validate(json.loads(data))
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc
                if chunk.usage is not None:
                    usage = _usage(chunk.usage) or usage
                for choice in chunk.choices:
                    if choice.finish_reason is not None:
                        finish_reason = _finish_reason(choice.finish_reason)
                    if choice.delta is not None and choice.delta.content:
                        yield StreamDelta(text=choice.delta.content)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc
        finally:
            await response.aclose()

        yield StreamEnd(finish_reason=finish_reason, usage=usage)

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        api_key = resolve_api_key(self._api_key_env)
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "input": list(request.inputs),
        }

        try:
            response = await self._client.post(
                "/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        if not response.is_success:
            raise_for_hosted_status(response.status_code)

        try:
            parsed = _CompatEmbeddingsResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ProviderProtocolError from exc

        # Vendors may return records out of order; `index` is authoritative when
        # present, and must form exactly 0..n-1 so no input is silently dropped.
        records = parsed.data
        if any(record.index is not None for record in records):
            if not all(record.index is not None for record in records):
                raise ProviderProtocolError
            if sorted(record.index for record in records) != list(  # type: ignore[misc]
                range(len(records))
            ):
                raise ProviderProtocolError
            records = sorted(records, key=lambda record: record.index or 0)

        usage = None
        if parsed.usage is not None and parsed.usage.prompt_tokens is not None:
            prompt_tokens = parsed.usage.prompt_tokens
            total_tokens = (
                parsed.usage.total_tokens
                if parsed.usage.total_tokens is not None
                else prompt_tokens
            )
            if prompt_tokens < 0 or total_tokens < 0:
                raise ProviderProtocolError
            usage = ProviderEmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)

        return ProviderEmbeddingResult(
            vectors=tuple(tuple(record.embedding) for record in records),
            usage=usage,
        )

    async def discover_runtime(self) -> RuntimeProbe:
        """Hosted models stay honestly unchecked until a real request uses them.

        Probing would call an authenticated (often billable) endpoint just to
        render health metadata, so Vulcan deliberately reports configured
        hosted providers as unchecked without any network I/O.
        """

        return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)

    async def aclose(self) -> None:
        await self._client.aclose()
