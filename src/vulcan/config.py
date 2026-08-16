"""Strict, versioned multi-provider startup configuration."""

from __future__ import annotations

import ipaddress
import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

PUBLIC_MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"
PROVIDER_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
# Seat labels: operator-chosen caller identities on API requests (same shape as
# provider IDs — safe, non-secret values that may appear in usage metadata).
SEAT_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
ENV_VAR_NAME_PATTERN = r"^[A-Z][A-Z0-9_]{0,127}$"

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


class StrictConfigModel(BaseModel):
    """Base for immutable configuration that rejects undocumented keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Capability(StrEnum):
    """Capabilities a configured model may declare in the first contract."""

    CHAT = "chat"
    EMBEDDINGS = "embeddings"


class ServerConfig(StrictConfigModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8140, ge=1, le=65535, strict=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("host")
    @classmethod
    def host_must_be_loopback(cls, value: str) -> str:
        candidate = value.strip().lower()
        if candidate == "localhost":
            return candidate
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError as exc:
            raise ValueError("server host must be localhost or a loopback IP address") from exc
        if not address.is_loopback:
            raise ValueError("server host must be a loopback IP address")
        return candidate


def _reject_boolean_timeout(value: object) -> object:
    if isinstance(value, bool):
        raise PydanticCustomError(
            "boolean_not_allowed", "provider timeout_seconds must be numeric, not boolean"
        )
    return value


def _validated_url_parts(value: str) -> tuple[str, str]:
    """Shared URL checks; returns (scheme, hostname) for policy decisions."""

    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("provider base_url must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("provider base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("provider base_url must not contain a query or fragment")
    hostname = parsed.hostname
    if hostname is None or not hostname:
        raise ValueError("provider base_url must include a host")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("provider base_url contains an invalid port") from exc
    if parsed.netloc.endswith(":") or port == 0:
        raise ValueError("provider base_url contains an invalid port")
    return parsed.scheme, hostname


def _is_loopback_hostname(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_local_base_url(value: str) -> str:
    """Loopback-only URL with no path: the v1 Ollama protection, unchanged."""

    parsed = urlsplit(value)
    if parsed.path not in {"", "/"}:
        raise ValueError("provider base_url must not contain a path, query, or fragment")
    _, hostname = _validated_url_parts(value)
    if not _is_loopback_hostname(hostname):
        raise ValueError("provider base_url must use localhost or a loopback IP")
    return value.rstrip("/")


def validate_hosted_base_url(value: str) -> str:
    """Explicitly configured hosted endpoint: HTTPS anywhere, or HTTP to loopback only.

    A path is allowed (several vendors mount their API under one), but query,
    fragment, and URL credentials are rejected. Cleartext HTTP is restricted to
    loopback so a bearer credential can never leave the machine unencrypted.
    """

    scheme, hostname = _validated_url_parts(value)
    if scheme == "http" and not _is_loopback_hostname(hostname):
        raise ValueError("hosted provider base_url must use https (http is loopback-only)")
    parsed = urlsplit(value)
    if parsed.path and "//" in parsed.path:
        raise ValueError("provider base_url path must not contain empty segments")
    return value.rstrip("/")


class OllamaProviderConfig(StrictConfigModel):
    type: Literal["ollama"]
    base_url: str
    timeout_seconds: float = Field(strict=True, ge=0.1, le=300.0)

    _timeout_not_boolean = field_validator("timeout_seconds", mode="before")(
        _reject_boolean_timeout
    )

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_loopback(cls, value: str) -> str:
        return validate_local_base_url(value)


class AnthropicProviderConfig(StrictConfigModel):
    type: Literal["anthropic"]
    base_url: str = ANTHROPIC_DEFAULT_BASE_URL
    api_key_env: str = Field(pattern=ENV_VAR_NAME_PATTERN)
    timeout_seconds: float = Field(strict=True, ge=0.1, le=300.0)
    default_max_tokens: int = Field(default=4096, strict=True, ge=1, le=32768)

    _timeout_not_boolean = field_validator("timeout_seconds", mode="before")(
        _reject_boolean_timeout
    )

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_hosted_safe(cls, value: str) -> str:
        return validate_hosted_base_url(value)


class OpenAICompatibleProviderConfig(StrictConfigModel):
    type: Literal["openai_compatible"]
    base_url: str
    api_key_env: str = Field(pattern=ENV_VAR_NAME_PATTERN)
    timeout_seconds: float = Field(strict=True, ge=0.1, le=300.0)
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    _timeout_not_boolean = field_validator("timeout_seconds", mode="before")(
        _reject_boolean_timeout
    )

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_hosted_safe(cls, value: str) -> str:
        return validate_hosted_base_url(value)


class DeterministicProviderConfig(StrictConfigModel):
    type: Literal["deterministic"]
    response_text: str = Field(min_length=1, max_length=8192)


ProviderConfig = Annotated[
    OllamaProviderConfig
    | AnthropicProviderConfig
    | OpenAICompatibleProviderConfig
    | DeterministicProviderConfig,
    Field(discriminator="type"),
]

HOSTED_PROVIDER_TYPES = frozenset({"anthropic", "openai_compatible"})


class ModelConfig(StrictConfigModel):
    id: str = Field(pattern=PUBLIC_MODEL_PATTERN)
    provider: str = Field(pattern=PROVIDER_ID_PATTERN)
    provider_model: str = Field(min_length=1, max_length=256)
    capabilities: frozenset[Capability] = Field(min_length=1)
    description: str | None = Field(default=None, max_length=240)

    @field_validator("provider_model")
    @classmethod
    def provider_model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("provider_model must not be blank")
        return value


class ReadinessConfig(StrictConfigModel):
    """Bounded reuse of one readiness probe across health/models/chat.

    ``probe_ttl_seconds`` of 0 means never reuse (probe every call). Upper bound
    keeps stale "available" memory short and explicit.
    """

    probe_ttl_seconds: float = Field(default=5.0, strict=True, ge=0.0, le=60.0)

    @field_validator("probe_ttl_seconds", mode="before")
    @classmethod
    def probe_ttl_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise PydanticCustomError(
                "boolean_not_allowed", "readiness probe_ttl_seconds must be numeric, not boolean"
            )
        return value


class GatewayConfig(StrictConfigModel):
    schema_version: Literal[2]
    server: ServerConfig = Field(default_factory=ServerConfig)
    readiness: ReadinessConfig = Field(default_factory=ReadinessConfig)
    providers: dict[str, ProviderConfig] = Field(min_length=1)
    models: tuple[ModelConfig, ...] = Field(min_length=1)

    @field_validator("schema_version", mode="before")
    @classmethod
    def schema_version_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise PydanticCustomError(
                "boolean_not_allowed", "schema_version must be an integer, not boolean"
            )
        if not isinstance(value, int):
            raise PydanticCustomError("integer_required", "schema_version must be an integer")
        return value

    @field_validator("providers")
    @classmethod
    def provider_ids_must_be_well_formed(
        cls, providers: dict[str, ProviderConfig]
    ) -> dict[str, ProviderConfig]:
        for provider_id in providers:
            if not re.fullmatch(PROVIDER_ID_PATTERN, provider_id):
                raise PydanticCustomError(
                    "invalid_provider_id",
                    "provider IDs must match the documented lowercase pattern",
                )
        return providers

    @field_validator("models")
    @classmethod
    def registry_must_be_valid(cls, models: tuple[ModelConfig, ...]) -> tuple[ModelConfig, ...]:
        ids = [model.id for model in models]
        if len(ids) != len(set(ids)):
            raise PydanticCustomError("duplicate_model_id", "configured model IDs must be unique")
        if not any(Capability.CHAT in model.capabilities for model in models):
            raise PydanticCustomError("chat_model_required", "at least one model must support chat")
        return models

    @model_validator(mode="after")
    def models_must_reference_configured_providers(self) -> GatewayConfig:
        for model in self.models:
            provider = self.providers.get(model.provider)
            if provider is None:
                raise PydanticCustomError(
                    "unknown_provider_reference",
                    "every model must reference a configured provider ID",
                )
            # Anthropic publishes no embeddings API: fail at startup rather than
            # letting an alias exist that can only ever fail at request time.
            if provider.type == "anthropic" and Capability.EMBEDDINGS in model.capabilities:
                raise PydanticCustomError(
                    "anthropic_embeddings_unsupported",
                    "anthropic providers do not support the embeddings capability",
                )
        return self


class ConfigIssue(StrictConfigModel):
    path: str
    reason: str


class ConfigLoadError(Exception):
    """Safe startup error; raw file content and Pydantic input are discarded."""

    def __init__(self, reason: str, issues: tuple[ConfigIssue, ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.issues = issues


_V1_MIGRATION_MESSAGE = (
    "configuration uses the retired schema_version 1; migrate to schema_version 2: "
    "declare named [providers.<id>] tables, set models[].provider to a provider ID, "
    "and rename models[].runtime_name to provider_model (see README, "
    "'Migrating from schema v1')"
)


def load_config(path: Path) -> GatewayConfig:
    """Load a TOML config without leaking its values through errors."""

    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigLoadError("configuration file was not found") from exc
    except OSError as exc:
        raise ConfigLoadError("configuration file could not be read") from exc

    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigLoadError("configuration file is not valid UTF-8 TOML") from exc

    # Recognize a v1 file before field validation so the operator gets one
    # clear migration instruction instead of a wall of unrelated field errors.
    declared_version = document.get("schema_version")
    if (type(declared_version) is int and declared_version == 1) or (
        "provider" in document and "providers" not in document
    ):
        raise ConfigLoadError(_V1_MIGRATION_MESSAGE)

    try:
        return GatewayConfig.model_validate(document)
    except ValidationError as exc:
        issues = tuple(
            ConfigIssue(
                path=".".join(str(part) for part in error["loc"]),
                reason=error["type"],
            )
            for error in exc.errors(include_url=False, include_input=False)
        )
    raise ConfigLoadError("configuration validation failed", issues) from None
