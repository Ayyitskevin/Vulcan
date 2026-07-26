"""One adapter for OpenAI and every OpenAI-compatible chat-completions vendor.

Vendor differences (base URL, credential variable, token-limit field name) are
configuration; the wire contract here is the stable common subset: one
non-streaming ``POST {base_url}/chat/completions`` with Bearer auth. Unknown
response fields are ignored because vendors extend the shape freely, but the
fields Vulcan relies on are validated strictly and never coerced.
"""

from __future__ import annotations

from typing import Any, Literal

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
    ProviderTokenUsage,
)
from vulcan.providers.http import build_client, raise_for_hosted_status, resolve_api_key
from vulcan.readiness import RuntimeProbe


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

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        api_key = resolve_api_key(self._api_key_env)
        payload: dict[str, Any] = {
            "model": request.provider_model,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload[self._max_tokens_field] = request.max_tokens

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
        finish_reason: Literal["stop", "length"] | None
        if choice.finish_reason == "stop":
            finish_reason = "stop"
        elif choice.finish_reason == "length":
            finish_reason = "length"
        else:
            finish_reason = None

        usage = None
        if parsed.usage is not None:
            prompt_tokens = parsed.usage.prompt_tokens
            completion_tokens = parsed.usage.completion_tokens
            if (prompt_tokens is not None and prompt_tokens < 0) or (
                completion_tokens is not None and completion_tokens < 0
            ):
                raise ProviderProtocolError
            if prompt_tokens is not None and completion_tokens is not None:
                usage = ProviderTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        return ProviderChatResult(
            content=choice.message.content,
            finish_reason=finish_reason,
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
