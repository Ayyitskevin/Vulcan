"""Native adapter for an explicitly configured local Ollama runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

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
    ProviderTokenUsage,
)


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


class OllamaProvider:
    kind: Literal["ollama"] = "ollama"

    def __init__(
        self,
        config: OllamaProviderConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        payload: dict[str, Any] = {
            "model": request.runtime_model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": False,
        }
        options: dict[str, float | int] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        if options:
            payload["options"] = options

        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError from exc

        if response.status_code == 404:
            try:
                error_body = response.json()
            except ValueError:
                error_body = None
            error_message = error_body.get("error") if isinstance(error_body, Mapping) else None
            if (
                isinstance(error_message, str)
                and "model" in error_message.casefold()
                and "not found" in error_message.casefold()
            ):
                raise ModelUnavailableError
            raise ProviderError(retryable=False)
        if not response.is_success:
            raise ProviderError(
                retryable=response.status_code >= 500 or response.status_code in {408, 429}
            )

        try:
            parsed = _OllamaChatResponse.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            raise ProviderProtocolError from exc
        if not parsed.done:
            raise ProviderProtocolError

        finish_reason: Literal["stop", "length"] | None
        if parsed.done_reason == "length":
            finish_reason = "length"
        elif parsed.done_reason in {None, "stop"}:
            finish_reason = "stop"
        else:
            finish_reason = None

        usage = None
        if (parsed.prompt_eval_count is not None and parsed.prompt_eval_count < 0) or (
            parsed.eval_count is not None and parsed.eval_count < 0
        ):
            raise ProviderProtocolError
        if parsed.prompt_eval_count is not None and parsed.eval_count is not None:
            usage = ProviderTokenUsage(
                prompt_tokens=parsed.prompt_eval_count,
                completion_tokens=parsed.eval_count,
            )
        return ProviderChatResult(
            content=parsed.message.content,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
