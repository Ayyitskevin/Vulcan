"""Native adapter for an explicitly configured local Ollama runtime."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Annotated, Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vulcan.config import OllamaProviderConfig
from vulcan.errors import (
    ModelUnavailableError,
    ProviderError,
    ProviderProtocolError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderEmbeddingRequest,
    ProviderEmbeddingResult,
    ProviderEmbeddingUsage,
    ProviderMessage,
    ProviderStreamEvent,
    ProviderTokenUsage,
    ProviderToolCall,
    StreamDelta,
    StreamEnd,
)
from vulcan.providers.http import (
    build_client,
    idle_bounded,
    iter_bounded_lines,
    open_response,
    read_bounded_json,
    send_response,
)
from vulcan.readiness import RuntimeProbe


class _OllamaToolCallFunction(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str
    # Ollama returns already-decoded arguments; OpenAI returns a JSON string.
    # Normalized to a string at the boundary so callers see one shape.
    arguments: dict[str, Any] | str


class _OllamaToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    function: _OllamaToolCallFunction


class _OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    role: Literal["assistant"]
    content: str
    # Ollama returns this for thinking models and counts it in eval_count. Without
    # the field, extra="ignore" dropped it and the caller was billed for output it
    # could neither see nor suppress.
    thinking: str | None = None
    # A list, not a tuple: this model is strict and parses JSON, which has no tuples.
    tool_calls: list[_OllamaToolCall] | None = None


class _OllamaChatResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: _OllamaMessage
    done: bool
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


class _OllamaStreamChunk(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    message: _OllamaMessage | None = None
    done: bool
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None


def _wire_message(message: ProviderMessage) -> dict[str, Any]:
    wire: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_call_id is not None:
        # Ollama keys a tool result by name rather than by call id.
        wire["tool_call_id"] = message.tool_call_id
    return wire


def _tool_calls(message: _OllamaMessage) -> tuple[ProviderToolCall, ...] | None:
    """Normalize Ollama tool calls, which carry no id, into the shared shape.

    Ollama emits neither a call id nor a stable ordering key, so an index-derived
    id is synthesized. It is positional and local to this response — a caller
    matching a tool result back to a call must echo exactly what it received.
    """

    if not message.tool_calls:
        return None
    calls: list[ProviderToolCall] = []
    for index, call in enumerate(message.tool_calls):
        arguments = call.function.arguments
        calls.append(
            ProviderToolCall(
                id=f"call_{index}",
                name=call.function.name,
                arguments=arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, separators=(",", ":"), sort_keys=True),
            )
        )
    return tuple(calls)


def _finish_reason(done_reason: str | None) -> Literal["stop", "length"] | None:
    if done_reason == "length":
        return "length"
    if done_reason in {None, "stop"}:
        return "stop"
    return None


def _usage(prompt_eval_count: int | None, eval_count: int | None) -> ProviderTokenUsage | None:
    """Token usage only when the runtime reported both counts, never invented."""

    if (prompt_eval_count is not None and prompt_eval_count < 0) or (
        eval_count is not None and eval_count < 0
    ):
        raise ProviderProtocolError
    if prompt_eval_count is None or eval_count is None:
        return None
    return ProviderTokenUsage(prompt_tokens=prompt_eval_count, completion_tokens=eval_count)


class _OllamaEmbedResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    # allow_inf_nan=False: Python's JSON parser accepts NaN/Infinity, which must
    # never reach a client as a "vector".
    embeddings: list[list[Annotated[float, Field(allow_inf_nan=False)]]] = Field(min_length=1)
    prompt_eval_count: int | None = None


class _OllamaTagModel(BaseModel):
    # Ollama returns JSON arrays; allow list→model coercion (not strict tuples).
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)


class _OllamaTagsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: list[_OllamaTagModel] = Field(default_factory=list)


class OllamaProvider:
    provider_type: Literal["ollama"] = "ollama"

    def __init__(
        self,
        provider_id: str,
        config: OllamaProviderConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._stream_idle_seconds = config.stream_idle_timeout_seconds
        self._client = client or build_client(
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    @staticmethod
    def _payload(request: ProviderChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "messages": [_wire_message(message) for message in request.messages],
            "stream": stream,
        }
        if request.response_format is not None:
            # Ollama takes the schema itself, or the string "json" for any shape.
            payload["format"] = request.response_format.json_schema or "json"
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        **({"description": tool.description} if tool.description else {}),
                        **({"parameters": tool.parameters} if tool.parameters is not None else {}),
                    },
                }
                for tool in request.tools
            ]
        options: dict[str, float | int] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
        # Thinking directive, only when the caller stated one. Absent means
        # byte-identical wire behavior to before this field existed.
        if request.think is not None:
            payload["think"] = request.think
        # Residency knob, only when the operator configured one: absent means
        # byte-identical wire behavior to before this field existed.
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive
        return payload

    @staticmethod
    def _raise_for_status(status_code: int, body: object) -> None:
        """Ollama's 404 means either 'no such model' or 'no such route'."""

        if status_code == 404:
            error_message = body.get("error") if isinstance(body, Mapping) else None
            if (
                isinstance(error_message, str)
                and "model" in error_message.casefold()
                and "not found" in error_message.casefold()
            ):
                raise ModelUnavailableError
            raise ProviderError(retryable=False)
        if status_code == 429:
            # Same condition as a hosted 429, same client-visible error class:
            # retry/backoff semantics must not depend on the provider type.
            raise ProviderRateLimitError
        raise ProviderError(retryable=status_code >= 500 or status_code == 408)

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        payload = self._payload(request, stream=False)

        try:
            async with open_response(
                self._client, "POST", "/api/chat", json_body=payload
            ) as response:
                if not response.is_success:
                    try:
                        error_body = await read_bounded_json(response)
                    except ValueError:
                        error_body = None
                    self._raise_for_status(response.status_code, error_body)

                try:
                    parsed = _OllamaChatResponse.model_validate(await read_bounded_json(response))
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc
        if not parsed.done:
            raise ProviderProtocolError

        calls = _tool_calls(parsed.message)
        return ProviderChatResult(
            content=parsed.message.content,
            finish_reason="tool_calls" if calls else _finish_reason(parsed.done_reason),
            usage=_usage(parsed.prompt_eval_count, parsed.eval_count),
            tool_calls=calls,
            thinking=parsed.message.thinking,
        )

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        """Stream one chat completion over Ollama's newline-delimited JSON.

        The upstream connection is opened and classified before any event is
        yielded; closing this iterator closes the upstream response.
        """

        payload = self._payload(request, stream=True)
        finish_reason: Literal["stop", "length"] | None = None
        streamed_calls: tuple[ProviderToolCall, ...] | None = None
        usage: ProviderTokenUsage | None = None

        try:
            response = await send_response(self._client, "POST", "/api/chat", json_body=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        try:
            if not response.is_success:
                error_body: object = None
                try:
                    error_body = await read_bounded_json(response)
                except ValueError:
                    error_body = None
                self._raise_for_status(response.status_code, error_body)
            async for line in idle_bounded(iter_bounded_lines(response), self._stream_idle_seconds):
                if not line.strip():
                    continue
                try:
                    chunk = _OllamaStreamChunk.model_validate(json.loads(line))
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc
                if chunk.message is not None and chunk.message.content:
                    yield StreamDelta(text=chunk.message.content)
                if chunk.message is not None and chunk.message.tool_calls:
                    # Collected, not yielded: a caller cannot act on half a
                    # call, so whole calls ride the terminal event instead.
                    streamed_calls = _tool_calls(chunk.message)
                if chunk.done:
                    finish_reason = _finish_reason(chunk.done_reason)
                    usage = _usage(chunk.prompt_eval_count, chunk.eval_count)
                    break
            else:
                # The runtime closed without a done chunk: truncated reply.
                raise ProviderProtocolError
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc
        finally:
            await response.aclose()

        yield StreamEnd(
            finish_reason="tool_calls" if streamed_calls else finish_reason,
            usage=usage,
            tool_calls=streamed_calls,
        )

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "input": list(request.inputs),
        }
        # Same residency passthrough as chat; unset means byte-identical.
        if request.keep_alive is not None:
            payload["keep_alive"] = request.keep_alive

        try:
            async with open_response(
                self._client, "POST", "/api/embed", json_body=payload
            ) as response:
                if not response.is_success:
                    try:
                        error_body = await read_bounded_json(response)
                    except ValueError:
                        error_body = None
                    self._raise_for_status(response.status_code, error_body)

                try:
                    parsed = _OllamaEmbedResponse.model_validate(await read_bounded_json(response))
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        usage = None
        if parsed.prompt_eval_count is not None:
            if parsed.prompt_eval_count < 0:
                raise ProviderProtocolError
            # Embeddings have no completion tokens, so total equals prompt.
            usage = ProviderEmbeddingUsage(
                prompt_tokens=parsed.prompt_eval_count,
                total_tokens=parsed.prompt_eval_count,
            )

        return ProviderEmbeddingResult(
            vectors=tuple(tuple(vector) for vector in parsed.embeddings),
            usage=usage,
        )

    async def discover_runtime(self) -> RuntimeProbe:
        """List installed runtime names via Ollama ``/api/tags`` with finite timeout.

        Failures never invent availability: transport errors → unavailable,
        timeouts and protocol/malformed bodies → unchecked, successful lists →
        available with the exact name set returned by the runtime.
        """

        try:
            async with open_response(self._client, "GET", "/api/tags") as response:
                if not response.is_success:
                    return RuntimeProbe(
                        live=False,
                        provider_availability="unavailable",
                        runtime_names=None,
                    )
                try:
                    parsed = _OllamaTagsResponse.model_validate(await read_bounded_json(response))
                except (ValueError, ValidationError, ProviderProtocolError):
                    return RuntimeProbe(
                        live=False,
                        provider_availability="unchecked",
                        runtime_names=None,
                    )
        except httpx.TimeoutException:
            return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)
        except httpx.RequestError:
            return RuntimeProbe(live=False, provider_availability="unavailable", runtime_names=None)

        names = frozenset(model.name for model in parsed.models)
        return RuntimeProbe(live=True, provider_availability="available", runtime_names=names)

    async def aclose(self) -> None:
        await self._client.aclose()
