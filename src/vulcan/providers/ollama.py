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
from vulcan.providers.http import build_client
from vulcan.readiness import RuntimeProbe


class _OllamaMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    role: Literal["assistant"]
    content: str


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
        self._client = client or build_client(
            base_url=config.base_url,
            timeout_seconds=config.timeout_seconds,
        )

    @staticmethod
    def _payload(request: ProviderChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": stream,
        }
        options: dict[str, float | int] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options
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
        raise ProviderError(retryable=status_code >= 500 or status_code in {408, 429})

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        payload = self._payload(request, stream=False)

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        if not response.is_success:
            try:
                error_body = response.json()
            except ValueError:
                error_body = None
            self._raise_for_status(response.status_code, error_body)

        try:
            parsed = _OllamaChatResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ProviderProtocolError from exc
        if not parsed.done:
            raise ProviderProtocolError

        return ProviderChatResult(
            content=parsed.message.content,
            finish_reason=_finish_reason(parsed.done_reason),
            usage=_usage(parsed.prompt_eval_count, parsed.eval_count),
        )

    async def chat_stream(self, request: ProviderChatRequest) -> AsyncIterator[ProviderStreamEvent]:
        """Stream one chat completion over Ollama's newline-delimited JSON.

        The upstream connection is opened and classified before any event is
        yielded; closing this iterator closes the upstream response.
        """

        payload = self._payload(request, stream=True)
        finish_reason: Literal["stop", "length"] | None = None
        usage: ProviderTokenUsage | None = None

        # An explicit send/close pair (rather than the stream() context manager)
        # keeps the upstream response closable when the consumer abandons this
        # generator mid-stream, e.g. on client disconnect.
        upstream = self._client.build_request("POST", "/api/chat", json=payload)
        try:
            response = await self._client.send(upstream, stream=True)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        try:
            if not response.is_success:
                error_body: object = None
                try:
                    error_body = json.loads(await response.aread())
                except ValueError:
                    error_body = None
                self._raise_for_status(response.status_code, error_body)
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = _OllamaStreamChunk.model_validate(json.loads(line))
                except (ValueError, ValidationError) as exc:
                    raise ProviderProtocolError from exc
                if chunk.message is not None and chunk.message.content:
                    yield StreamDelta(text=chunk.message.content)
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

        yield StreamEnd(finish_reason=finish_reason, usage=usage)

    async def embed(self, request: ProviderEmbeddingRequest) -> ProviderEmbeddingResult:
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "input": list(request.inputs),
        }

        try:
            response = await self._client.post("/api/embed", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        if not response.is_success:
            try:
                error_body = response.json()
            except ValueError:
                error_body = None
            self._raise_for_status(response.status_code, error_body)

        try:
            parsed = _OllamaEmbedResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ProviderProtocolError from exc

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
            response = await self._client.get("/api/tags")
        except httpx.TimeoutException:
            return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)
        except httpx.RequestError:
            return RuntimeProbe(live=False, provider_availability="unavailable", runtime_names=None)

        if not response.is_success:
            return RuntimeProbe(live=False, provider_availability="unavailable", runtime_names=None)

        try:
            parsed = _OllamaTagsResponse.model_validate(response.json())
        except (ValueError, ValidationError):
            return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)

        names = frozenset(model.name for model in parsed.models)
        return RuntimeProbe(live=True, provider_availability="available", runtime_names=names)

    async def aclose(self) -> None:
        await self._client.aclose()
