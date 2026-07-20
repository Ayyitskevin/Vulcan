"""FastAPI application factory and stable v1 routes."""

from __future__ import annotations

import ipaddress
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from vulcan import __version__
from vulcan.config import GatewayConfig
from vulcan.errors import VulcanError
from vulcan.gateway import Gateway
from vulcan.providers.base import Provider
from vulcan.providers.factory import build_provider
from vulcan.registry import ModelRegistry
from vulcan.schemas import (
    CapabilitiesResponse,
    ChatCapability,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DiscoveryMetadata,
    ErrorBody,
    ErrorEnvelope,
    HealthResponse,
    ModelListResponse,
    ModelRecord,
    ProviderHealth,
    ValidationIssue,
)

logger = logging.getLogger("vulcan.api")

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorEnvelope},
    404: {"model": ErrorEnvelope},
    422: {"model": ErrorEnvelope},
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


def create_app(
    config: GatewayConfig,
    *,
    provider: Provider | None = None,
    clock: Callable[[], float] = time.time,
    id_factory: Callable[[], str] | None = None,
) -> FastAPI:
    registry = ModelRegistry(config.models)
    selected_provider = provider or build_provider(config.provider)
    if id_factory is None:
        gateway = Gateway(registry, selected_provider, clock=clock)
    else:
        gateway = Gateway(registry, selected_provider, clock=clock, id_factory=id_factory)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await selected_provider.aclose()

    app = FastAPI(
        title="Vulcan Local Inference Gateway",
        version=__version__,
        description="A local-only, configuration-driven subset of the OpenAI chat contract.",
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
        logger.error(
            "internal_error",
            extra={
                "metadata": {
                    "request_id": _request_id(request),
                    "method": request.method,
                    "route": _route_name(request),
                    "exception_class": type(exc).__name__,
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
    async def healthz() -> HealthResponse:
        return HealthResponse(
            provider=ProviderHealth(kind=selected_provider.kind),
            models_configured=len(registry.list()),
        )

    @app.get("/v1/models", response_model=ModelListResponse, responses=ERROR_RESPONSES)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            discovery=DiscoveryMetadata(),
            data=tuple(
                ModelRecord(
                    id=model.id,
                    provider=selected_provider.kind,
                    capabilities=tuple(sorted(model.capabilities, key=str)),
                    description=model.description,
                )
                for model in registry.list()
            ),
        )

    @app.get(
        "/v1/capabilities",
        response_model=CapabilitiesResponse,
        responses=ERROR_RESPONSES,
    )
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(chat_completions=ChatCapability())

    @app.post(
        "/v1/chat/completions",
        response_model=ChatCompletionResponse,
        responses=ERROR_RESPONSES,
    )
    async def chat_completions(
        payload: ChatCompletionRequest, request: Request
    ) -> ChatCompletionResponse:
        return await gateway.chat(payload, request_id=_request_id(request))

    return app
