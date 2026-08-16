"""FastAPI application factory and stable v1 routes."""

from __future__ import annotations

import ipaddress
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response, StreamingResponse

from vulcan import __version__
from vulcan.config import GatewayConfig
from vulcan.errors import VulcanError
from vulcan.gateway import Gateway
from vulcan.providers.base import Provider
from vulcan.providers.factory import build_providers
from vulcan.registry import ModelRegistry
from vulcan.schemas import (
    Availability,
    CapabilitiesResponse,
    ChatCapability,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DiscoveryMetadata,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ErrorBody,
    ErrorEnvelope,
    HealthResponse,
    LedgerRecord,
    ModelListResponse,
    ModelRecord,
    ModelUsageRecord,
    ProviderStatus,
    ProviderUsageRecord,
    SeatUsageRecord,
    UsageResponse,
    UsageTotalsRecord,
    ValidationIssue,
)
from vulcan.usage import UsageLedger, UsageRecorder, UsageTotals

logger = logging.getLogger("vulcan.api")

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorEnvelope},
    404: {"model": ErrorEnvelope},
    422: {"model": ErrorEnvelope},
    429: {"model": ErrorEnvelope},
    500: {"model": ErrorEnvelope},
    502: {"model": ErrorEnvelope},
    503: {"model": ErrorEnvelope},
    504: {"model": ErrorEnvelope},
}


def _is_loopback_host_header(value: str | None) -> bool:
    """Accept only literal loopback addresses or localhost in the HTTP Host header."""

    if value is None or not value or value != value.strip():
        return False

    port: str | None = None
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            return False
        host = value[1:closing]
        suffix = value[closing + 1 :]
        if suffix:
            if not suffix.startswith(":"):
                return False
            port = suffix[1:]
    elif value.count(":") == 1:
        host, port = value.rsplit(":", 1)
    elif ":" in value:
        return False
    else:
        host = value

    if not host or "%" in host:
        return False
    if port is not None and (
        not port.isascii() or not port.isdecimal() or not 1 <= int(port) <= 65535
    ):
        return False

    candidate = host.lower()
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    return request_id if isinstance(request_id, str) else str(uuid4())


def _route_name(request: Request) -> str:
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    return route_path if isinstance(route_path, str) else "<unmatched>"


def _json_error(
    *,
    request: Request,
    status_code: int,
    code: str,
    message: str,
    retryable: bool,
    details: dict[str, str | int | bool] | None = None,
    validation: tuple[ValidationIssue, ...] | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    envelope = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            retryable=retryable,
            details=details,
            validation=validation,
        ),
        request_id=request_id,
    )
    response = JSONResponse(status_code=status_code, content=envelope.model_dump(mode="json"))
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


SSE_DONE = "data: [DONE]\n\n"


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _chunk_frame(chunk: ChatCompletionChunk) -> str:
    """Render one chunk, omitting absent delta fields like the OpenAI wire shape.

    ``finish_reason`` stays present (null until the final chunk); ``usage`` is
    omitted entirely unless the upstream reported both counts.
    """

    payload = chunk.model_dump(mode="json")
    for choice in payload["choices"]:
        choice["delta"] = {
            key: value for key, value in choice["delta"].items() if value is not None
        }
    if payload.get("usage") is None:
        payload.pop("usage", None)
    return _sse_frame(payload)


def _error_frame(exc: VulcanError, request_id: str) -> str:
    """Terminal event for a failure after the response headers were committed."""

    body = ErrorBody(
        code=exc.code,
        message=exc.message,
        retryable=exc.retryable,
        details=exc.details,
    )
    return _sse_frame({"error": body.model_dump(mode="json"), "request_id": request_id})


def _usage_totals(totals: UsageTotals) -> UsageTotalsRecord:
    return UsageTotalsRecord(
        requests=totals.requests,
        requests_with_usage=totals.requests_with_usage,
        prompt_tokens=totals.prompt_tokens,
        completion_tokens=totals.completion_tokens,
        total_tokens=totals.total_tokens,
    )


def create_app(
    config: GatewayConfig,
    *,
    providers: Mapping[str, Provider] | None = None,
    clock: Callable[[], float] = time.time,
    id_factory: Callable[[], str] | None = None,
) -> FastAPI:
    registry = ModelRegistry(config.models)
    selected_providers: dict[str, Provider] = dict(providers or build_providers(config))
    gateway_kwargs: dict[str, Any] = {
        "clock": clock,
        "readiness_ttl_seconds": config.readiness.probe_ttl_seconds,
    }
    if id_factory is not None:
        gateway_kwargs["id_factory"] = id_factory
    usage_scope = "process"
    started_at = int(clock())
    if config.usage is not None:
        # Fail-loud by design: an unopenable ledger raises LedgerError here
        # rather than silently degrading to in-memory counters.
        ledger = UsageLedger(config.usage.ledger_path, clock=clock)
        gateway_kwargs["usage"] = UsageRecorder.with_ledger(ledger)
        usage_scope = "ledger"
        if ledger.stats.earliest_ts is not None:
            started_at = min(started_at, ledger.stats.earliest_ts)
    gateway = Gateway(registry, selected_providers, **gateway_kwargs)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        for provider in selected_providers.values():
            await provider.aclose()

    app = FastAPI(
        title="Vulcan Local Inference Gateway",
        version=__version__,
        description=(
            "A local-first, single-user gateway exposing a configuration-driven subset "
            "of the OpenAI chat contract over explicitly configured local and BYOK models."
        ),
        license_info={"name": "AGPL-3.0-only"},
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.gateway = gateway

    @app.middleware("http")
    async def request_metadata(
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request.state.request_id = str(uuid4())
        started = time.perf_counter()
        if _is_loopback_host_header(request.headers.get("host")):
            response = await call_next(request)
        else:
            response = _json_error(
                request=request,
                status_code=400,
                code="invalid_host",
                message="The Host header must identify a loopback address.",
                retryable=False,
            )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        logger.info(
            "request_complete",
            extra={
                "metadata": {
                    "request_id": request.state.request_id,
                    "method": request.method,
                    "route": _route_name(request),
                    "status": response.status_code,
                    "duration_ms": duration_ms,
                }
            },
        )
        return response

    @app.exception_handler(VulcanError)
    async def vulcan_error_handler(request: Request, exc: VulcanError) -> JSONResponse:
        return _json_error(
            request=request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            retryable=exc.retryable,
            details=exc.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        issues = tuple(
            ValidationIssue(
                path=".".join(str(part) for part in error["loc"]),
                reason=error["type"],
            )
            for error in exc.errors()
        )
        return _json_error(
            request=request,
            status_code=422,
            code="invalid_request",
            message="The request does not match the Vulcan v1 contract.",
            retryable=False,
            validation=issues,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else "http_error"
        message = (
            "The requested endpoint does not exist." if exc.status_code == 404 else "HTTP error."
        )
        return _json_error(
            request=request,
            status_code=exc.status_code,
            code=code,
            message=message,
            retryable=False,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # "exception_type" (not "exception_class") because the formatter owns
        # the reserved exception_class slot and would silently drop this key.
        logger.error(
            "internal_error",
            extra={
                "metadata": {
                    "request_id": _request_id(request),
                    "method": request.method,
                    "route": _route_name(request),
                    "exception_type": type(exc).__name__,
                }
            },
        )
        return _json_error(
            request=request,
            status_code=500,
            code="internal_error",
            message="Vulcan could not complete the request.",
            retryable=False,
        )

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        responses={400: {"model": ErrorEnvelope}},
    )
    async def healthz(refresh: bool = False) -> HealthResponse:
        # Gateway liveness is always ok if this handler runs. Provider readiness
        # is a separate probe and never overrides the process-up signal.
        # refresh=true forces a new probe (bypasses the short TTL reuse bound).
        readiness = await gateway.readiness(force=refresh)
        return HealthResponse(
            providers=tuple(
                ProviderStatus(
                    id=item.provider_id,
                    type=item.provider_type,
                    availability=item.availability,
                )
                for item in readiness.providers
            ),
            models_configured=len(registry.list()),
        )

    def _model_record(model_id: str, availability: Availability) -> ModelRecord:
        model = registry.get(model_id)
        provider = selected_providers[model.provider_id]
        return ModelRecord(
            id=model.id,
            provider=provider.provider_id,
            provider_type=provider.provider_type,
            capabilities=tuple(sorted(model.capabilities, key=str)),
            availability=availability,
            description=model.description,
        )

    @app.get("/v1/models", response_model=ModelListResponse, responses=ERROR_RESPONSES)
    async def list_models(refresh: bool = False) -> ModelListResponse:
        # refresh=true forces a new probe (bypasses the short TTL reuse bound).
        readiness = await gateway.readiness(force=refresh)
        return ModelListResponse(
            discovery=DiscoveryMetadata(),
            data=tuple(
                _model_record(model.id, readiness.model_availability(model.id))
                for model in registry.list()
            ),
        )

    @app.get(
        "/v1/models/{model_id}",
        response_model=ModelRecord,
        responses=ERROR_RESPONSES,
    )
    async def get_model(model_id: str, refresh: bool = False) -> ModelRecord:
        """Retrieve one configured public model with the same readiness annotation."""

        # Raise the stable model_not_found envelope for unknown public IDs.
        registry.get(model_id)
        readiness = await gateway.readiness(force=refresh)
        return _model_record(model_id, readiness.model_availability(model_id))

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        responses=ERROR_RESPONSES,
    )
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(chat_completions=ChatCapability())

    async def _stream_response(payload: ChatCompletionRequest, request: Request) -> Response:
        """Render one chat stream as Server-Sent Events.

        The first chunk is pulled *before* the response starts so pre-stream
        failures still produce a normal JSON error envelope with the right
        status. Once headers are committed, a failure can only be reported as
        one terminal error event, and the stream then closes without [DONE].
        """

        request_id = _request_id(request)
        chunks = gateway.chat_stream(payload, request_id=request_id).__aiter__()
        # A VulcanError here propagates to the normal exception handlers.
        first = await chunks.__anext__()

        async def body() -> AsyncIterator[str]:
            try:
                yield _chunk_frame(first)
                async for chunk in chunks:
                    yield _chunk_frame(chunk)
            except VulcanError as exc:
                yield _error_frame(exc, request_id)
                return
            finally:
                await chunks.aclose()
            yield SSE_DONE

        return StreamingResponse(
            body(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        responses=ERROR_RESPONSES,
    )
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
        if payload.stream:
            return await _stream_response(payload, request)
        return await gateway.chat(payload, request_id=_request_id(request))

    @app.post(
        "/v1/embeddings",
        response_model=EmbeddingsResponse,
        responses=ERROR_RESPONSES,
    )
    async def embeddings(payload: EmbeddingsRequest, request: Request) -> EmbeddingsResponse:
        return await gateway.embed(payload, request_id=_request_id(request))

    @app.get("/v1/usage", response_model=UsageResponse, responses=ERROR_RESPONSES)
    async def usage() -> UsageResponse:
        """Process-lifetime counters for completed requests.

        In-memory only: these reset when the gateway restarts, and they carry
        no costs or currencies. Token totals cover only the requests whose
        upstream reported counts (``requests_with_usage``).
        """

        snapshot = gateway.usage_snapshot()
        ledger_record = None
        if snapshot.ledger is not None:
            ledger_record = LedgerRecord(
                replayed_requests=snapshot.ledger.replayed_requests,
                skipped_lines=snapshot.ledger.skipped_lines,
                write_failures=snapshot.ledger.write_failures,
            )
        return UsageResponse(
            scope=usage_scope,  # type: ignore[arg-type]
            started_at=started_at,
            totals=_usage_totals(snapshot.totals),
            by_model=tuple(
                ModelUsageRecord(
                    model=item.model,
                    provider=item.provider,
                    totals=_usage_totals(item.totals),
                )
                for item in snapshot.by_model
            ),
            by_provider=tuple(
                ProviderUsageRecord(provider=item.provider, totals=_usage_totals(item.totals))
                for item in snapshot.by_provider
            ),
            by_seat=tuple(
                SeatUsageRecord(seat=item.seat, totals=_usage_totals(item.totals))
                for item in snapshot.by_seat
            ),
            ledger=ledger_record,
        )

    return app
