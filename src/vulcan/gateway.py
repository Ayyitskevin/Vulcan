"""Provider-independent request orchestration with exact, no-fallback routing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from vulcan.budgets import BudgetBook
from vulcan.config import Capability
from vulcan.errors import (
    ConfigurationError,
    GatewayOverloadedError,
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
    ProviderTool,
    ProviderToolCall,
    StreamDelta,
)
from vulcan.readiness import (
    READINESS_PROBE_TTL_SECONDS,
    Availability,
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
    ToolCall,
    ToolCallFunction,
)
from vulcan.usage import UsageRecorder, UsageSnapshot

logger = logging.getLogger("vulcan.gateway")


def _new_completion_id() -> str:
    return f"chatcmpl-{uuid4().hex}"


@dataclass(frozen=True, slots=True)
class _CachedProbe:
    probe: RuntimeProbe
    expires_at: float


@dataclass(slots=True)
class _RequestScope:
    """Routing state for one admitted request; settles budget + meter once.

    Created by ``Gateway._lifecycle``. The only mutation is ``commit()``,
    which the entry points call the moment the upstream has completed — for
    a stream, BEFORE the terminal chunk is yielded, so a consumer that
    disconnects at the final yield cannot evade the meter or the budget.
    """

    _gateway: Gateway
    model: ConfiguredModel
    provider: Provider
    reservation: int | None
    seat: str | None
    settled: bool = False

    def commit(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None = None,
        settle_tokens: int | None,
    ) -> None:
        """Record the meter and settle the budget reservation exactly once."""

        self._gateway._usage.record(
            model=self.model.id,
            provider=self.provider.provider_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            seat=self.seat,
        )
        if self._gateway._budgets is not None and self.reservation is not None:
            self._gateway._budgets.settle(
                seat=self.seat,
                provider_id=self.provider.provider_id,
                tokens=settle_tokens,
                reservation_day=self.reservation,
            )
        self.settled = True


def _response_tool_calls(
    calls: tuple[ProviderToolCall, ...] | None,
) -> tuple[ToolCall, ...] | None:
    """Translate provider tool calls into the public response shape."""

    if not calls:
        return None
    return tuple(
        ToolCall(id=call.id, function=ToolCallFunction(name=call.name, arguments=call.arguments))
        for call in calls
    )


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
        max_concurrent_requests: int | None = None,
    ) -> None:
        if readiness_ttl_seconds < 0:
            raise ValueError("readiness_ttl_seconds must be non-negative")
        if not providers:
            raise ValueError("at least one provider must be configured")
        if max_concurrent_requests is not None and max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be positive")
        self.registry = registry
        self.providers = dict(providers)
        self._clock = clock
        self._id_factory = id_factory
        self._readiness_ttl_seconds = readiness_ttl_seconds
        self._cached_probes: dict[str, _CachedProbe] = {}
        self._probe_locks: dict[str, asyncio.Lock] = {}
        self._last_live: dict[str, bool] = {}
        self._usage = usage if usage is not None else UsageRecorder()
        self._budgets = budgets
        self._concurrency = (
            asyncio.Semaphore(max_concurrent_requests)
            if max_concurrent_requests is not None
            else None
        )

    async def _admit(self) -> bool:
        """Take an in-flight slot without queueing; False when saturated.

        ``locked()`` then ``acquire()`` with no suspension point between them
        is atomic on the event loop, so exactly the configured number of
        callers hold slots and every extra caller is refused immediately —
        backpressure, never a retry, queue, or reroute.
        """

        semaphore = self._concurrency
        if semaphore is None:
            return True
        if semaphore.locked():
            return False
        await semaphore.acquire()
        return True

    def _release_admission(self) -> None:
        if self._concurrency is not None:
            self._concurrency.release()

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
            # Name availability transitions: healthz never fails on a dead
            # backend (by design), so without this a provider dying at 3am is
            # invisible unless someone is polling and diffing. The first probe
            # sets the baseline silently; only flips are logged.
            previous = self._last_live.get(provider_id)
            if previous is not None and previous != probe.live:
                transition_log = logger.info if probe.live else logger.warning
                transition_log(
                    "provider_availability_changed",
                    extra={"metadata": {"provider": provider_id, "available": probe.live}},
                )
            self._last_live[provider_id] = probe.live
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

    async def model_readiness(self, model_id: str, *, force: bool = False) -> Availability:
        """Availability annotation for one configured model.

        Probes ONLY the provider the model routes to — the metadata path
        follows the same principle as chat preflight: unrelated providers are
        never contacted.
        """

        model = self.registry.get(model_id)
        probe, _ = await self._probe_provider(model.provider_id, force=force)
        provider = self._provider_for(model.provider_id)
        report = reconcile_configured_models(
            (model,), {model.provider_id: probe}, {model.provider_id: provider.provider_type}
        )
        return report.model_availability(model_id)

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

    @asynccontextmanager
    async def _lifecycle(
        self,
        *,
        capability: Capability,
        model_id: str,
        seat: str | None,
        metadata: dict[str, object],
        failure_event: str,
    ) -> AsyncIterator[_RequestScope]:
        """Admit, route, and reserve for one request; release on any exit.

        The chat, stream, and embeddings entry points share this scaffold so
        the settle/release discipline lives in exactly one place: overload and
        over-budget requests are refused loudly before any upstream call, an
        unsettled reservation is always returned (VulcanError, CancelledError,
        and client disconnect alike), and the in-flight slot always goes back.
        """

        provider: Provider | None = None
        reservation: int | None = None
        scope: _RequestScope | None = None
        admitted = await self._admit()
        try:
            try:
                if not admitted:
                    # Hard reject, not a queue: the caller owns any retry.
                    raise GatewayOverloadedError
                model = self.registry.require_capability(model_id, capability)
                provider = self._provider_for(model.provider_id)
                metadata["provider"] = provider.provider_id
                metadata["provider_type"] = provider.provider_type
                if self._budgets is not None:
                    # Budgets gate BEFORE the upstream call: over-budget
                    # requests are refused loudly, never rerouted. check()
                    # atomically reserves the request slot when it passes.
                    reservation = self._budgets.check(seat=seat, provider_id=provider.provider_id)
                scope = _RequestScope(self, model, provider, reservation, seat)
                yield scope
            except VulcanError as exc:
                self._handle_failure(exc, provider, metadata, event=failure_event)
                raise
        finally:
            if (
                reservation is not None
                and (scope is None or not scope.settled)
                and provider is not None
                and self._budgets is not None
            ):
                self._budgets.release(
                    seat=seat,
                    provider_id=provider.provider_id,
                    reservation_day=reservation,
                )
            if admitted:
                self._release_admission()

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
        if request.stream:
            # Streaming belongs on chat_stream; the HTTP layer routes it
            # there. Guard before any model lookup: reaching here would
            # otherwise silently buffer a stream.
            raise UnsupportedCapabilityError("streaming", request.model)
        async with self._lifecycle(
            capability=Capability.CHAT,
            model_id=request.model,
            seat=request.seat,
            metadata=metadata,
            failure_event="chat_failed",
        ) as scope:
            provider_request = await self._preflight(scope.model, request)
            result = await scope.provider.chat(provider_request)

            usage = None
            if result.usage is not None:
                usage = TokenUsage(
                    prompt_tokens=result.usage.prompt_tokens,
                    completion_tokens=result.usage.completion_tokens,
                    total_tokens=result.usage.prompt_tokens + result.usage.completion_tokens,
                )
            scope.commit(
                prompt_tokens=usage.prompt_tokens if usage is not None else None,
                completion_tokens=usage.completion_tokens if usage is not None else None,
                settle_tokens=usage.total_tokens if usage is not None else None,
            )
            logger.info(
                "chat_completed",
                extra={"metadata": {**metadata, "output_chars": len(result.content)}},
            )
            return ChatCompletionResponse(
                id=self._id_factory(),
                created=int(self._clock()),
                model=request.model,
                provider=scope.provider.provider_id,
                choices=(
                    ChatChoice(
                        message=AssistantMessage(
                            content=result.content,
                            tool_calls=_response_tool_calls(result.tool_calls),
                        ),
                        finish_reason=result.finish_reason,
                    ),
                ),
                usage=usage,
            )

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
                ProviderMessage(
                    role=message.role.value,
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                )
                for message in request.messages
            ),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            keep_alive=model.keep_alive,
            tools=tuple(
                ProviderTool(
                    name=definition.function.name,
                    description=definition.function.description,
                    parameters=definition.function.parameters,
                )
                for definition in request.tools
            )
            if request.tools
            else None,
            tool_choice=request.tool_choice,
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

        async with self._lifecycle(
            capability=Capability.CHAT,
            model_id=request.model,
            seat=request.seat,
            metadata=metadata,
            failure_event="chat_failed",
        ) as scope:
            provider_request = await self._preflight(scope.model, request)
            events = scope.provider.chat_stream(provider_request).__aiter__()
            # Opening the upstream stream (and classifying its status)
            # happens on this first pull, while a JSON error envelope is
            # still possible.
            event = await self._next_event(events)

            completion_id = self._id_factory()
            created = int(self._clock())

            def chunk(
                delta: ChunkDelta,
                *,
                finish_reason: Literal["stop", "length", "tool_calls"] | None = None,
                usage: TokenUsage | None = None,
            ) -> ChatCompletionChunk:
                return ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=request.model,
                    provider=scope.provider.provider_id,
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
                    # The upstream HAS completed: its tokens are generated and
                    # (when reported) known. Commit the meter and the budget
                    # BEFORE yielding the terminal chunk, so a consumer that
                    # disconnects at the final yield cannot evade either —
                    # repeated final-chunk abandonment must never be free.
                    scope.commit(
                        prompt_tokens=final_usage.prompt_tokens
                        if final_usage is not None
                        else None,
                        completion_tokens=final_usage.completion_tokens
                        if final_usage is not None
                        else None,
                        settle_tokens=final_usage.total_tokens if final_usage is not None else None,
                    )
                    logger.info(
                        "chat_completed",
                        extra={"metadata": {**metadata, "output_chars": output_chars}},
                    )
                    # Whole tool calls ride the terminal chunk: the adapters
                    # reassemble them so a caller never sees half an argument
                    # string it cannot execute.
                    yield chunk(
                        ChunkDelta(tool_calls=_response_tool_calls(event.tool_calls)),
                        finish_reason=event.finish_reason,
                        usage=final_usage,
                    )
                    completed = True
                    break
                if not completed:
                    # The upstream closed without a terminal event: truncated reply.
                    raise ProviderProtocolError
            finally:
                # Releases the upstream response on normal completion, on error,
                # and when the client disconnects mid-stream.
                aclose = getattr(events, "aclose", None)
                if aclose is not None:
                    await aclose()

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
        async with self._lifecycle(
            capability=Capability.EMBEDDINGS,
            model_id=request.model,
            seat=request.seat,
            metadata=metadata,
            failure_event="embeddings_failed",
        ) as scope:
            await self._assert_model_available(scope.model)
            result = await scope.provider.embed(
                ProviderEmbeddingRequest(
                    provider_model=scope.model.provider_model,
                    inputs=inputs,
                    keep_alive=scope.model.keep_alive,
                )
            )
            if len(result.vectors) != len(inputs):
                # A vector per input, or the client cannot align them.
                raise ProviderProtocolError

            usage = None
            if result.usage is not None:
                usage = EmbeddingUsage(
                    prompt_tokens=result.usage.prompt_tokens,
                    total_tokens=result.usage.total_tokens,
                )
            scope.commit(
                prompt_tokens=usage.prompt_tokens if usage is not None else None,
                settle_tokens=usage.total_tokens if usage is not None else None,
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
                provider=scope.provider.provider_id,
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
