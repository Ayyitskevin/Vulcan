"""Strict, local-only startup configuration."""

from __future__ import annotations

import ipaddress
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_core import PydanticCustomError

PUBLIC_MODEL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$"


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


class OllamaProviderConfig(StrictConfigModel):
    kind: Literal["ollama"]
    base_url: str
    timeout_seconds: float = Field(strict=True, ge=0.1, le=300.0)

    @field_validator("timeout_seconds", mode="before")
    @classmethod
    def timeout_must_not_be_boolean(cls, value: object) -> object:
        if isinstance(value, bool):
            raise PydanticCustomError(
                "boolean_not_allowed", "provider timeout_seconds must be numeric, not boolean"
            )
        return value

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_loopback(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("provider base_url must use http or https")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("provider base_url must not contain credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("provider base_url must not contain a path, query, or fragment")
        hostname = parsed.hostname
        if hostname is None:
            raise ValueError("provider base_url must include a host")
        if hostname.lower() != "localhost":
            try:
                address = ipaddress.ip_address(hostname)
            except ValueError as exc:
                raise ValueError("provider base_url must use localhost or a loopback IP") from exc
            if not address.is_loopback:
                raise ValueError("provider base_url must use a loopback IP")
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("provider base_url contains an invalid port") from exc
        if parsed.netloc.endswith(":") or port == 0:
            raise ValueError("provider base_url contains an invalid port")
        return value.rstrip("/")


class DeterministicProviderConfig(StrictConfigModel):
    kind: Literal["deterministic"]
    response_text: str = Field(min_length=1, max_length=8192)


ProviderConfig = Annotated[
    OllamaProviderConfig | DeterministicProviderConfig,
    Field(discriminator="kind"),
]


class ModelConfig(StrictConfigModel):
    id: str = Field(pattern=PUBLIC_MODEL_PATTERN)
    runtime_name: str = Field(min_length=1, max_length=256)
    capabilities: frozenset[Capability] = Field(min_length=1)
    description: str | None = Field(default=None, max_length=240)

    @field_validator("runtime_name")
    @classmethod
    def runtime_name_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("runtime_name must not be blank")
        return value


class GatewayConfig(StrictConfigModel):
    schema_version: Literal[1]
    server: ServerConfig = Field(default_factory=ServerConfig)
    provider: ProviderConfig
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

    @field_validator("models")
    @classmethod
    def registry_must_be_valid(cls, models: tuple[ModelConfig, ...]) -> tuple[ModelConfig, ...]:
        ids = [model.id for model in models]
        if len(ids) != len(set(ids)):
            raise PydanticCustomError("duplicate_model_id", "configured model IDs must be unique")
        if not any(Capability.CHAT in model.capabilities for model in models):
            raise PydanticCustomError("chat_model_required", "at least one model must support chat")
        return models


class ConfigIssue(StrictConfigModel):
    path: str
    reason: str


class ConfigLoadError(Exception):
    """Safe startup error; raw file content and Pydantic input are discarded."""

    def __init__(self, reason: str, issues: tuple[ConfigIssue, ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.issues = issues


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
