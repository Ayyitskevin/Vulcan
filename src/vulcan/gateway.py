"""Provider-independent request orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from uuid import uuid4

from vulcan.config import Capability
from vulcan.errors import UnsupportedCapabilityError, VulcanError
from vulcan.providers.base import Provider, ProviderChatRequest, ProviderMessage
from vulcan.registry import ModelRegistry
from vulcan.schemas import (
    AssistantMessage,
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TokenUsage,
)

logger = logging.getLogger("vulcan.gateway")


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


class Gateway:
    def __init__(
        self,
        registry: ModelRegistry,
        provider: Provider,
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = _new_completion_id,
    ) -> None:
        self.registry = registry
        self.provider = provider
        self._clock = clock
        self._id_factory = id_factory

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str | None = None,
    ) -> ChatCompletionResponse:
        input_chars = sum(len(message.content) for message in request.messages)
        metadata: dict[str, object] = {
            "provider": self.provider.kind,
            "model": request.model,
            "turn_count": len(request.messages),
            "input_chars": input_chars,
        }
        if request_id is not None:
            metadata["request_id"] = request_id
        try:
            if request.stream:
                raise UnsupportedCapabilityError("streaming", request.model)
            model = self.registry.require_capability(request.model, Capability.CHAT)
            provider_request = ProviderChatRequest(
                runtime_model=model.runtime_name,
                messages=tuple(
                    ProviderMessage(role=message.role.value, content=message.content)
                    for message in request.messages
                ),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            result = await self.provider.chat(provider_request)
        except VulcanError as exc:
            logger.warning("chat_failed", extra={"metadata": {**metadata, "error_code": exc.code}})
            raise

        usage = None
        if result.usage is not None:
            usage = TokenUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
            )
        logger.info(
            "chat_completed",
            extra={"metadata": {**metadata, "output_chars": len(result.content)}},
        )
        return ChatCompletionResponse(
            id=self._id_factory(),
            created=int(self._clock()),
            model=request.model,
            provider=self.provider.kind,
            choices=(
                ChatChoice(
                    message=AssistantMessage(content=result.content),
                    finish_reason=result.finish_reason,
                ),
            ),
            usage=usage,
        )
