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

from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from vulcan.config import AnthropicProviderConfig
from vulcan.errors import (
    ProviderProtocolError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from vulcan.providers.base import (
    ProviderChatRequest,
    ProviderChatResult,
    ProviderTokenUsage,
)
from vulcan.providers.http import build_client, raise_for_hosted_status, resolve_api_key
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

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        if request.temperature is not None and request.temperature > 1.0:
            raise UnsupportedCapabilityError("temperature_above_one")
        system, turns = _translate_messages(request)
        api_key = resolve_api_key(self._api_key_env)

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

        finish_reason: Literal["stop", "length"] | None
        if parsed.stop_reason in {"end_turn", "stop_sequence"}:
            finish_reason = "stop"
        elif parsed.stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = None

        usage = None
        if parsed.usage is not None:
            input_tokens = parsed.usage.input_tokens
            output_tokens = parsed.usage.output_tokens
            if (input_tokens is not None and input_tokens < 0) or (
                output_tokens is not None and output_tokens < 0
            ):
                raise ProviderProtocolError
            if input_tokens is not None and output_tokens is not None:
                usage = ProviderTokenUsage(
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )
        return ProviderChatResult(
            content="".join(texts),
            finish_reason=finish_reason,
            usage=usage,
        )

    async def discover_runtime(self) -> RuntimeProbe:
        """Hosted models stay honestly unchecked until a real request uses them."""

        return RuntimeProbe(live=False, provider_availability="unchecked", runtime_names=None)

    async def aclose(self) -> None:
        await self._client.aclose()
