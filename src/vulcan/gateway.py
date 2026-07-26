"""Provider-independent request orchestration with exact, no-fallback routing."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from uuid import uuid4

from vulcan.config import Capability
from vulcan.errors import (
    ConfigurationError,
    ModelUnavailableError,
    UnsupportedCapabilityError,
    VulcanError,
)
from vulcan.providers.base import Provider, ProviderChatRequest, ProviderMessage
from vulcan.readiness import (
    READINESS_PROBE_TTL_SECONDS,
    GatewayReadiness,
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
    report: GatewayReadiness
    expires_at: float


class Gateway:
    def __init__(
        self,
        registry: ModelRegistry,
        providers: Mapping[str, Provider],
        *,
        clock: Callable[[], float] = time.time,
        id_factory: Callable[[], str] = _new_completion_id,
        readiness_ttl_seconds: float = READINESS_PROBE_TTL_SECONDS,
    ) -> None:
        if readiness_ttl_seconds < 0:
            raise ValueError("readiness_ttl_seconds must be non-negative")
        if not providers:
            raise ValueError("at least one provider must be configured")
        self.registry = registry
        self.providers = dict(providers)
        self._clock = clock
        self._id_factory = id_factory
        self._readiness_ttl_seconds = readiness_ttl_seconds
        self._cached_readiness: _CachedReadiness | None = None

    def _provider_for(self, provider_id: str) -> Provider:
        """Exact routing: the configured provider or a loud configuration error."""

        try:
            return self.providers[provider_id]
        except KeyError:
            # Config validation makes this unreachable; fail loud, never fall back.
            raise ConfigurationError(details={"provider": provider_id}) from None

    def _readiness_log_metadata(
        self,
        report: GatewayReadiness,
        *,
        forced: bool,
        reused: bool,
    ) -> dict[str, object]:
        counts = {"available": 0, "unavailable": 0, "unchecked": 0}
        for item in report.models:
            counts[item.availability] += 1
        return {
            "providers_configured": len(self.providers),
            "models_configured": len(report.models),
            "models_available": counts["available"],
            "models_unavailable": counts["unavailable"],
            "models_unchecked": counts["unchecked"],
            "forced": forced,
            "reused": reused,
            "probe_ttl_seconds": self._readiness_ttl_seconds,
        }

    async def readiness(self, *, force: bool = False) -> GatewayReadiness:
        """Probe (or reuse) every provider's readiness and reconcile configured models.

        Within ``readiness_ttl_seconds`` of a successful capture, returns the
        same report without re-probing. ``force=True`` always runs new probes.
        Only Ollama providers perform network I/O; hosted providers report
        unchecked and deterministic providers report available, both in-process.
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

        probes = {
            provider_id: await provider.discover_runtime()
            for provider_id, provider in self.providers.items()
        }
        provider_types = {
            provider_id: provider.provider_type for provider_id, provider in self.providers.items()
        }
        report = reconcile_configured_models(self.registry.list(), probes, provider_types)
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
        """Drop cached readiness so the next call re-probes the providers."""

        self._cached_readiness = None

    def _known_unavailable(self, report: GatewayReadiness, model_id: str) -> bool:
        """True only after a successful live inventory proved the native name absent.

        Only a live probe (Ollama) ever marks a model "unavailable"; hosted
        providers stay unchecked, so this never short-circuits hosted traffic.
        """

        return report.model_availability(model_id) == "unavailable"

    async def chat(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str | None = None,
    ) -> ChatCompletionResponse:
        input_chars = sum(len(message.content) for message in request.messages)
        metadata: dict[str, object] = {
            "model": request.model,
            "turn_count": len(request.messages),
            "input_chars": input_chars,
        }
        if request_id is not None:
            metadata["request_id"] = request_id
        provider: Provider | None = None
        try:
            if request.stream:
                raise UnsupportedCapabilityError("streaming", request.model)
            model = self.registry.require_capability(request.model, Capability.CHAT)
            provider = self._provider_for(model.provider_id)
            metadata["provider"] = provider.provider_id
            metadata["provider_type"] = provider.provider_type
            # Preflight only short-circuits when a live list proved absence.
            # Unchecked/provider-down falls through so the adapter fails loud.
            readiness = await self.readiness()
            if self._known_unavailable(readiness, model.id):
                raise ModelUnavailableError(model.id)
            provider_request = ProviderChatRequest(
                provider_model=model.provider_model,
                messages=tuple(
                    ProviderMessage(role=message.role.value, content=message.content)
                    for message in request.messages
                ),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            result = await provider.chat(provider_request)
        except ModelUnavailableError as exc:
            # Provider (or preflight) proved the model unavailable — drop any
            # stale "available" inventory so the next health/models re-probes.
            self.invalidate_readiness()
            self._annotate_provider(exc, provider)
            logger.warning(
                "chat_failed", extra={"metadata": {**metadata, "error_code": "model_unavailable"}}
            )
            raise
        except VulcanError as exc:
            self._annotate_provider(exc, provider)
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
            provider=provider.provider_id,
            choices=(
                ChatChoice(
                    message=AssistantMessage(content=result.content),
                    finish_reason=result.finish_reason,
                ),
            ),
            usage=usage,
        )

    @staticmethod
    def _annotate_provider(exc: VulcanError, provider: Provider | None) -> None:
        """Attach the safe configured provider ID to a routed request's failure."""

        if provider is None or exc.code in {"model_not_found", "unsupported_capability"}:
            return
        details = dict(exc.details) if exc.details else {}
        details.setdefault("provider", provider.provider_id)
        exc.details = details
