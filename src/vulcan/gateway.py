"""Provider-independent request orchestration with exact, no-fallback routing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from vulcan.budgets import BudgetBook
from vulcan.config import Capability
from vulcan.errors import (
    ConfigurationError,
    ModelUnavailableError,
    ProviderProtocolError,
    UnsupportedCapabilityError,
    VulcanError,
)
from vulcan.providers.base import (
    Provider,
    ProviderChatRequest,
    ProviderEmbeddingRequest,
    ProviderMessage,
    ProviderStreamEvent,
    StreamDelta,
)
from vulcan.readiness import (
    READINESS_PROBE_TTL_SECONDS,
    GatewayReadiness,
    RuntimeProbe,
    reconcile_configured_models,
    runtime_name_matches,
)
from vulcan.registry import ConfiguredModel, ModelRegistry
from vulcan.schemas import (
    AssistantMessage,
    ChatChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChunkDelta,
    EmbeddingRecord,
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingUsage,
    TokenUsage,
)
from vulcan.usage import UsageRecorder, UsageSnapshot

logger = logging.getLogger("vulcan.gateway")


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class _CachedProbe:
    probe: RuntimeProbe
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
        usage: UsageRecorder | None = None,
        budgets: BudgetBook | None = None,
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
        self._cached_probes: dict[str, _CachedProbe] = {}
        self._probe_locks: dict[str, asyncio.Lock] = {}
        self._usage = usage if usage is not None else UsageRecorder()
        self._budgets = budgets

    def _provider_for(self, provider_id: str) -> Provider:
        """Exact routing: the configured provider or a loud configuration error."""

        try:
            return self.providers[provider_id]
        except KeyError:
            # Config validation makes this unreachable; fail loud, never fall back.
            raise ConfigurationError(details={"provider": provider_id}) from None

    def _fresh_cached_probe(self, provider_id: str) -> RuntimeProbe | None:
        cached = self._cached_probes.get(provider_id)
        if cached is not None and self._clock() < cached.expires_at:
            return cached.probe
        return None

    async def _probe_provider(
        self, provider_id: str, *, force: bool = False
    ) -> tuple[RuntimeProbe, bool]:
        """One provider's probe, reused within the TTL; returns (probe, reused).

        Single-flight per provider: concurrent callers wait on one probe rather
        than each opening their own, and re-check the cache after acquiring the
        lock. The expiry is computed *after* the probe completes so a probe
        slower than the TTL still yields one full reuse window instead of an
        already-expired entry (which would re-pay the probe on every request).
        """

        if not force:
            probe = self._fresh_cached_probe(provider_id)
            if probe is not None:
                return probe, True

        lock = self._probe_locks.setdefault(provider_id, asyncio.Lock())
        async with lock:
            # Another caller may have refreshed this provider while we waited.
            if not force:
                probe = self._fresh_cached_probe(provider_id)
                if probe is not None:
                    return probe, True
            probe = await self._provider_for(provider_id).discover_runtime()
            self._cached_probes[provider_id] = _CachedProbe(
                probe=probe,
                expires_at=self._clock() + self._readiness_ttl_seconds,
            )
            return probe, False

    def _readiness_log_metadata(
        self,
        report: GatewayReadiness,
        *,
        forced: bool,
        providers_reused: int,
    ) -> dict[str, object]:
        counts = {"available": 0, "unavailable": 0, "unchecked": 0}
        for item in report.models:
            counts[item.availability] += 1
        return {
            "providers_configured": len(self.providers),
            "providers_reused": providers_reused,
            "models_configured": len(report.models),
            "models_available": counts["available"],
            "models_unavailable": counts["unavailable"],
            "models_unchecked": counts["unchecked"],
            "forced": forced,
            "reused": providers_reused == len(self.providers),
            "probe_ttl_seconds": self._readiness_ttl_seconds,
        }

    async def readiness(self, *, force: bool = False) -> GatewayReadiness:
        """Probe (or reuse) every provider's readiness and reconcile configured models.

        Each provider's probe is reused within ``readiness_ttl_seconds`` of its
        capture; ``force=True`` always runs new probes. Only Ollama providers
        perform network I/O; hosted providers report unchecked and
        deterministic providers report available, both in-process.
        """

        probes: dict[str, RuntimeProbe] = {}
        reused_count = 0
        for provider_id in self.providers:
            probe, reused = await self._probe_provider(provider_id, force=force)
            probes[provider_id] = probe
            reused_count += 1 if reused else 0
        provider_types = {
            provider_id: provider.provider_type for provider_id, provider in self.providers.items()
        }
        report = reconcile_configured_models(self.registry.list(), probes, provider_types)
        event = "readiness_reused" if reused_count == len(self.providers) else "readiness_probed"
        logger.info(
            event,
            extra={
                "metadata": self._readiness_log_metadata(
                    report, forced=force, providers_reused=reused_count
                )
            },
        )
        return report

    def invalidate_readiness(self, provider_id: str | None = None) -> None:
        """Drop cached probes so the next call re-probes.

        With ``provider_id``, only that provider's probe is dropped — a failure
        on one provider must not discard another provider's valid state.
        """

        if provider_id is None:
            self._cached_probes.clear()
        else:
            self._cached_probes.pop(provider_id, None)

    @staticmethod
    def _known_unavailable(probe: RuntimeProbe, provider_model: str) -> bool:
        """True only after a successful live inventory proved the native name absent.

        Only a live probe (Ollama) ever satisfies this; hosted and
        deterministic probes are never live, so this never short-circuits
        their traffic.
        """

        if not probe.live or probe.runtime_names is None:
            return False
        return not runtime_name_matches(provider_model, probe.runtime_names)

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
                # Streaming belongs on chat_stream; the HTTP layer routes it
                # there. Reaching here would silently buffer a stream instead.
                raise UnsupportedCapabilityError("streaming", request.model)
            model, provider = self._select_provider(request, metadata)
            if self._budgets is not None:
                # Budgets gate BEFORE the upstream call: over-budget requests
                # are refused loudly, never rerouted (one alias, one provider).
                self._budgets.check(seat=request.seat, provider_id=provider.provider_id)
            provider_request = await self._preflight(model, request)
            result = await provider.chat(provider_request)
        except VulcanError as exc:
            self._handle_chat_failure(exc, provider, metadata)
            raise

        usage = None
        if result.usage is not None:
            usage = TokenUsage(
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                total_tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
            )
        self._usage.record(
            model=request.model,
            provider=provider.provider_id,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            seat=request.seat,
        )
        if self._budgets is not None:
            self._budgets.spend(
                seat=request.seat,
                provider_id=provider.provider_id,
                tokens=usage.total_tokens if usage is not None else None,
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

    def _select_provider(
        self,
        request: ChatCompletionRequest,
        metadata: dict[str, object],
    ) -> tuple[ConfiguredModel, Provider]:
        """Resolve the alias to exactly one configured provider; never a fallback."""

        model = self.registry.require_capability(request.model, Capability.CHAT)
        provider = self._provider_for(model.provider_id)
        metadata["provider"] = provider.provider_id
        metadata["provider_type"] = provider.provider_type
        return model, provider

    async def _assert_model_available(self, model: ConfiguredModel) -> None:
        """Probe ONLY the routed provider before spending an upstream call.

        Short-circuits solely when a live inventory proved the native model
        absent; unchecked or unreachable providers fall through so the adapter
        fails loud. Unrelated providers are never contacted.
        """

        probe, _ = await self._probe_provider(model.provider_id)
        if self._known_unavailable(probe, model.provider_model):
            raise ModelUnavailableError(model.id)

    async def _preflight(
        self,
        model: ConfiguredModel,
        request: ChatCompletionRequest,
    ) -> ProviderChatRequest:
        """Probe the routed provider and translate the chat request."""

        await self._assert_model_available(model)
        return ProviderChatRequest(
            provider_model=model.provider_model,
            messages=tuple(
                ProviderMessage(role=message.role.value, content=message.content)
                for message in request.messages
            ),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

    def _handle_failure(
        self,
        exc: VulcanError,
        provider: Provider | None,
        metadata: dict[str, object],
        *,
        event: str,
    ) -> None:
        """Annotate, invalidate stale inventory, and log one safe failure event."""

        if isinstance(exc, ModelUnavailableError) and provider is not None:
            # The model is proven absent — drop that provider's stale inventory
            # so its next probe is fresh, without touching other providers.
            self.invalidate_readiness(provider.provider_id)
        self._annotate_provider(exc, provider)
        logger.warning(event, extra={"metadata": {**metadata, "error_code": exc.code}})

    def _handle_chat_failure(
        self,
        exc: VulcanError,
        provider: Provider | None,
        metadata: dict[str, object],
    ) -> None:
        self._handle_failure(exc, provider, metadata, event="chat_failed")

    @staticmethod
    async def _next_event(
        events: AsyncIterator[ProviderStreamEvent],
    ) -> ProviderStreamEvent | None:
        try:
            return await events.__anext__()
        except StopAsyncIteration:
            return None

    async def chat_stream(
        self,
        request: ChatCompletionRequest,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream one chat completion as OpenAI-style chunks.

        Routing, preflight, and logging match the buffered path exactly. All
        pre-stream failures (unknown alias, missing credential, upstream auth,
        …) raise before the first chunk is yielded so the HTTP layer can still
        answer with a normal JSON error envelope; failures after that raise
        mid-iteration for the caller to render as a terminal error event.
        """

        metadata: dict[str, object] = {
            "model": request.model,
            "turn_count": len(request.messages),
            "input_chars": sum(len(message.content) for message in request.messages),
            "stream": True,
        }
        if request_id is not None:
            metadata["request_id"] = request_id

        provider: Provider | None = None
        try:
            model, provider = self._select_provider(request, metadata)
            if self._budgets is not None:
                # Budgets gate BEFORE the upstream call: over-budget requests
                # are refused loudly, never rerouted (one alias, one provider).
                self._budgets.check(seat=request.seat, provider_id=provider.provider_id)
            provider_request = await self._preflight(model, request)
            events = provider.chat_stream(provider_request).__aiter__()
            # Opening the upstream stream (and classifying its status) happens
            # on this first pull, while a JSON error envelope is still possible.
            event = await self._next_event(events)
        except VulcanError as exc:
            self._handle_chat_failure(exc, provider, metadata)
            raise

        completion_id = self._id_factory()
        created = int(self._clock())

        def chunk(
            delta: ChunkDelta,
            *,
            finish_reason: Literal["stop", "length"] | None = None,
            usage: TokenUsage | None = None,
        ) -> ChatCompletionChunk:
            return ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=request.model,
                provider=provider.provider_id,
                choices=(ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason),),
                usage=usage,
            )

        yield chunk(ChunkDelta(role="assistant"))

        output_chars = 0
        final_usage: TokenUsage | None = None
        try:
            completed = False
            while event is not None:
                if isinstance(event, StreamDelta):
                    if event.text:
                        output_chars += len(event.text)
                        yield chunk(ChunkDelta(content=event.text))
                    event = await self._next_event(events)
                    continue
                if event.usage is not None:
                    final_usage = TokenUsage(
                        prompt_tokens=event.usage.prompt_tokens,
                        completion_tokens=event.usage.completion_tokens,
                        total_tokens=event.usage.prompt_tokens + event.usage.completion_tokens,
                    )
                yield chunk(ChunkDelta(), finish_reason=event.finish_reason, usage=final_usage)
                completed = True
                break
            if not completed:
                # The upstream closed without a terminal event: truncated reply.
                raise ProviderProtocolError
        except VulcanError as exc:
            self._handle_chat_failure(exc, provider, metadata)
            raise
        finally:
            # Releases the upstream response on normal completion, on error,
            # and when the client disconnects mid-stream.
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()

        self._usage.record(
            model=request.model,
            provider=provider.provider_id,
            prompt_tokens=final_usage.prompt_tokens if final_usage is not None else None,
            completion_tokens=final_usage.completion_tokens if final_usage is not None else None,
            seat=request.seat,
        )
        if self._budgets is not None:
            self._budgets.spend(
                seat=request.seat,
                provider_id=provider.provider_id,
                tokens=final_usage.total_tokens if final_usage is not None else None,
            )
        logger.info(
            "chat_completed",
            extra={"metadata": {**metadata, "output_chars": output_chars}},
        )

    async def embed(
        self,
        request: EmbeddingsRequest,
        *,
        request_id: str | None = None,
    ) -> EmbeddingsResponse:
        """Embed one batch of inputs through exactly one configured provider."""

        inputs = request.inputs
        metadata: dict[str, object] = {
            "model": request.model,
            "input_count": len(inputs),
            "input_chars": sum(len(item) for item in inputs),
        }
        if request_id is not None:
            metadata["request_id"] = request_id
        provider: Provider | None = None
        try:
            model = self.registry.require_capability(request.model, Capability.EMBEDDINGS)
            provider = self._provider_for(model.provider_id)
            metadata["provider"] = provider.provider_id
            metadata["provider_type"] = provider.provider_type
            if self._budgets is not None:
                # Same gate as chat: refused loudly before the upstream call.
                self._budgets.check(seat=request.seat, provider_id=provider.provider_id)
            await self._assert_model_available(model)
            result = await provider.embed(
                ProviderEmbeddingRequest(
                    provider_model=model.provider_model,
                    inputs=inputs,
                )
            )
            if len(result.vectors) != len(inputs):
                # A vector per input, or the client cannot align them at all.
                raise ProviderProtocolError
        except VulcanError as exc:
            self._handle_failure(exc, provider, metadata, event="embeddings_failed")
            raise

        usage = None
        if result.usage is not None:
            usage = EmbeddingUsage(
                prompt_tokens=result.usage.prompt_tokens,
                total_tokens=result.usage.total_tokens,
            )
        self._usage.record(
            model=request.model,
            provider=provider.provider_id,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            seat=request.seat,
        )
        if self._budgets is not None:
            self._budgets.spend(
                seat=request.seat,
                provider_id=provider.provider_id,
                tokens=usage.total_tokens if usage is not None else None,
            )
        logger.info(
            "embeddings_completed",
            extra={
                "metadata": {
                    **metadata,
                    "vector_count": len(result.vectors),
                    "dimensions": len(result.vectors[0]) if result.vectors else 0,
                }
            },
        )
        return EmbeddingsResponse(
            model=request.model,
            provider=provider.provider_id,
            data=tuple(
                EmbeddingRecord(index=index, embedding=vector)
                for index, vector in enumerate(result.vectors)
            ),
            usage=usage,
        )

    def usage_snapshot(self) -> UsageSnapshot:
        """Process-lifetime counters for completed requests."""

        return self._usage.snapshot()

    @staticmethod
    def _annotate_provider(exc: VulcanError, provider: Provider | None) -> None:
        """Attach the safe configured provider ID to a routed request's failure."""

        if provider is None or exc.code in {"model_not_found", "unsupported_capability"}:
            return
        details = dict(exc.details) if exc.details else {}
        details.setdefault("provider", provider.provider_id)
        exc.details = details
