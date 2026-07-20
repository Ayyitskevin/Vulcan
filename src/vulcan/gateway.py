"""Provider-independent request orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from vulcan.config import Capability
from vulcan.errors import ModelUnavailableError, UnsupportedCapabilityError, VulcanError
from vulcan.providers.base import Provider, ProviderChatRequest, ProviderMessage
from vulcan.readiness import (
    READINESS_PROBE_TTL_SECONDS,
    DiscoveryReadiness,
    reconcile_configured_models,
)
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


@dataclass(frozen=True, slots=True)
class _CachedReadiness:
    report: DiscoveryReadiness
    expires_at: float


class Gateway:
    def __init__(
        self,
        registry: ModelRegistry,
        provider: Provider,
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = _new_completion_id,
        readiness_ttl_seconds: float = READINESS_PROBE_TTL_SECONDS,
    ) -> None:
        if readiness_ttl_seconds < 0:
            raise ValueError("readiness_ttl_seconds must be non-negative")
        self.registry = registry
        self.provider = provider
        self._clock = clock
        self._id_factory = id_factory
        self._readiness_ttl_seconds = readiness_ttl_seconds
        self._cached_readiness: _CachedReadiness | None = None

    def _readiness_log_metadata(
        self,
        report: DiscoveryReadiness,
        *,
        forced: bool,
        reused: bool,
    ) -> dict[str, object]:
        counts = {"available": 0, "unavailable": 0, "unchecked": 0}
        for item in report.models:
            counts[item.availability] += 1
        return {
            "provider": self.provider.kind,
            "live": report.live,
            "provider_availability": report.provider_availability,
            "models_configured": len(report.models),
            "models_available": counts["available"],
            "models_unavailable": counts["unavailable"],
            "models_unchecked": counts["unchecked"],
            "forced": forced,
            "reused": reused,
            "probe_ttl_seconds": self._readiness_ttl_seconds,
        }

    async def readiness(self, *, force: bool = False) -> DiscoveryReadiness:
        """Probe (or reuse) provider readiness and reconcile configured models only.

        Within ``readiness_ttl_seconds`` of a successful capture, returns the
        same report without re-probing. ``force=True`` always runs a new probe.
        Deterministic providers perform no network I/O either way.
        """

        now = self._clock()
        if (
            not force
            and self._cached_readiness is not None
            and now < self._cached_readiness.expires_at
        ):
            report = self._cached_readiness.report
            logger.info(
                "readiness_reused",
                extra={"metadata": self._readiness_log_metadata(report, forced=False, reused=True)},
            )
            return report

        probe = await self.provider.discover_runtime()
        report = reconcile_configured_models(self.registry.list(), probe)
        self._cached_readiness = _CachedReadiness(
            report=report,
            expires_at=now + self._readiness_ttl_seconds,
        )
        logger.info(
            "readiness_probed",
            extra={"metadata": self._readiness_log_metadata(report, forced=force, reused=False)},
        )
        return report

    def invalidate_readiness(self) -> None:
        """Drop cached readiness so the next call re-probes the provider."""

        self._cached_readiness = None

    def _known_unavailable(self, report: DiscoveryReadiness, model_id: str) -> bool:
        """True only after a successful live inventory proved the runtime name absent."""

        if not report.live:
            return False
        for item in report.models:
            if item.model_id == model_id and item.availability == "unavailable":
                return True
        return False

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
            # Preflight only short-circuits when a live list proved absence.
            # Unchecked/provider-down falls through so the adapter fails loud.
            readiness = await self.readiness()
            if self._known_unavailable(readiness, model.id):
                raise ModelUnavailableError(model.id)
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
        except ModelUnavailableError:
            # Provider (or preflight) proved the model unavailable — drop any
            # stale "available" inventory so the next health/models re-probes.
            self.invalidate_readiness()
            logger.warning(
                "chat_failed", extra={"metadata": {**metadata, "error_code": "model_unavailable"}}
            )
            raise
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
