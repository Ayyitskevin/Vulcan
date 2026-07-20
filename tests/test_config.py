from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vulcan.config import (
    ConfigLoadError,
    GatewayConfig,
    OllamaProviderConfig,
    ServerConfig,
    load_config,
)


def _error_types(error: ValidationError) -> set[str]:
    return {item["type"] for item in error.errors(include_url=False, include_input=False)}


def test_server_defaults_to_ipv4_loopback() -> None:
    server = ServerConfig()

    assert server.host == "127.0.0.1"
    assert server.port == 8140


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [
        ("127.0.0.1", "127.0.0.1"),
        ("127.42.0.9", "127.42.0.9"),
        ("::1", "::1"),
        ("localhost", "localhost"),
        (" LOCALHOST ", "localhost"),
    ],
)
def test_server_accepts_loopback_hosts(configured: str, normalized: str) -> None:
    assert ServerConfig(host=configured).host == normalized


@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "::",
        "192.168.1.10",
        "10.0.0.8",
        "fe80::1",
        "example.com",
        "localhost.example.com",
        "",
    ],
)
def test_server_rejects_non_loopback_hosts(host: str) -> None:
    with pytest.raises(ValidationError):
        ServerConfig(host=host)


@pytest.mark.parametrize("port", [0, 65536, True])
def test_server_rejects_invalid_ports(port: int) -> None:
    with pytest.raises(ValidationError):
        ServerConfig(port=port)


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434"),
        ("http://127.1.2.3:11434/", "http://127.1.2.3:11434"),
        ("http://localhost:11434", "http://localhost:11434"),
        ("https://[::1]:11434/", "https://[::1]:11434"),
    ],
)
def test_ollama_provider_accepts_only_loopback_urls(
    configured: str,
    normalized: str,
) -> None:
    provider = OllamaProviderConfig(
        kind="ollama",
        base_url=configured,
        timeout_seconds=60,
    )

    assert provider.base_url == normalized


@pytest.mark.parametrize(
    "base_url",
    [
        "http://0.0.0.0:11434",
        "http://192.168.1.10:11434",
        "http://[::]:11434",
        "https://api.openai.com/v1",
        "http://localhost.example.com:11434",
        "http://user@127.0.0.1:11434",
        "http://user:password@127.0.0.1:11434",
        "http://127.0.0.1:11434/api",
        "http://127.0.0.1:11434?token=secret",
        "http://127.0.0.1:11434#fragment",
        "ftp://127.0.0.1:11434",
        "file:///tmp/provider.sock",
        "http:///missing-host",
        "http://127.0.0.1:99999",
        "http://127.0.0.1:0",
        "http://127.0.0.1:",
    ],
)
def test_ollama_provider_rejects_unsafe_or_ambiguous_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        OllamaProviderConfig(
            kind="ollama",
            base_url=base_url,
            timeout_seconds=60,
        )


@pytest.mark.parametrize("timeout_seconds", [0.09, 300.01])
def test_ollama_provider_rejects_out_of_range_timeouts(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        OllamaProviderConfig(
            kind="ollama",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=timeout_seconds,
        )


def test_ollama_provider_accepts_integer_timeout_without_boolean_coercion() -> None:
    provider = OllamaProviderConfig(
        kind="ollama",
        base_url="http://127.0.0.1:11434",
        timeout_seconds=60,
    )

    assert provider.timeout_seconds == 60.0


def test_ollama_provider_rejects_boolean_timeout() -> None:
    with pytest.raises(ValidationError) as raised:
        OllamaProviderConfig(
            kind="ollama",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=True,
        )

    assert "boolean_not_allowed" in _error_types(raised.value)


def test_ollama_provider_rejects_string_timeout() -> None:
    with pytest.raises(ValidationError) as raised:
        OllamaProviderConfig(
            kind="ollama",
            base_url="http://127.0.0.1:11434",
            timeout_seconds="60",  # type: ignore[arg-type]
        )

    assert "float_type" in _error_types(raised.value)


def test_load_config_reports_missing_file_without_path_echo(tmp_path: Path) -> None:
    missing = tmp_path / "private-sentinel-config.toml"

    with pytest.raises(ConfigLoadError) as raised:
        load_config(missing)

    assert raised.value.reason == "configuration file was not found"
    assert raised.value.issues == ()
    assert "private-sentinel" not in str(raised.value)


@pytest.mark.parametrize(
    "content",
    [
        b"[provider\nkind = 'ollama'",
        b"\xff\xfe\x00",
    ],
)
def test_load_config_rejects_malformed_or_non_utf8_toml(
    tmp_path: Path,
    content: bytes,
) -> None:
    path = tmp_path / "vulcan.toml"
    path.write_bytes(content)

    with pytest.raises(ConfigLoadError) as raised:
        load_config(path)

    assert raised.value.reason == "configuration file is not valid UTF-8 TOML"
    assert raised.value.issues == ()


def test_load_config_rejects_unknown_fields_without_value_echo(tmp_path: Path) -> None:
    sentinel = "CONFIG_VALUE_MUST_NOT_BE_ECHOED"
    path = tmp_path / "vulcan.toml"
    path.write_text(
        f"""
schema_version = 1
undocumented = "{sentinel}"

[provider]
kind = "deterministic"
response_text = "safe-response"

[[models]]
id = "chat-model"
runtime_name = "runtime-model"
capabilities = ["chat"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError) as raised:
        load_config(path)

    assert raised.value.reason == "configuration validation failed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert any(
        issue.path == "undocumented" and issue.reason == "extra_forbidden"
        for issue in raised.value.issues
    )
    public_error = f"{raised.value.reason} {raised.value.issues!r} {raised.value!r}"
    assert sentinel not in public_error


@pytest.mark.parametrize("missing_field", ["schema_version", "provider", "models"])
def test_gateway_config_rejects_missing_required_sections(
    valid_config_document: dict[str, Any],
    missing_field: str,
) -> None:
    del valid_config_document[missing_field]

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert any(
        tuple(item["loc"]) == (missing_field,) and item["type"] == "missing"
        for item in raised.value.errors(include_url=False, include_input=False)
    )


def test_gateway_config_rejects_unknown_provider_kind(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["provider"]["kind"] = "automatic-cloud-fallback"

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "union_tag_invalid" in _error_types(raised.value)


def test_gateway_config_rejects_unknown_nested_fields(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["provider"]["api_key"] = "must-not-be-supported"

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "extra_forbidden" in _error_types(raised.value)


def test_gateway_config_rejects_unknown_schema_version(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["schema_version"] = 2

    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(valid_config_document)


def test_gateway_config_rejects_boolean_schema_version(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["schema_version"] = True

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "boolean_not_allowed" in _error_types(raised.value)


@pytest.mark.parametrize("schema_version", [1.0, "1"])
def test_gateway_config_rejects_non_integer_schema_version(
    valid_config_document: dict[str, Any],
    schema_version: object,
) -> None:
    valid_config_document["schema_version"] = schema_version

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "integer_required" in _error_types(raised.value)


def test_gateway_config_rejects_duplicate_public_model_ids(
    valid_config_document: dict[str, Any],
) -> None:
    duplicate = dict(valid_config_document["models"][0])
    duplicate["runtime_name"] = "another-runtime-model"
    valid_config_document["models"].append(duplicate)

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "duplicate_model_id" in _error_types(raised.value)


def test_gateway_config_rejects_empty_model_registry(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"] = []

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "too_short" in _error_types(raised.value)


def test_gateway_config_requires_at_least_one_chat_capable_model(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"][0]["capabilities"] = ["embeddings"]

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    errors = raised.value.errors(include_url=False, include_input=False)
    assert any(
        tuple(item["loc"]) == ("models",) and item["type"] == "chat_model_required"
        for item in errors
    )


def test_gateway_config_rejects_empty_capability_set(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"][0]["capabilities"] = []

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "too_short" in _error_types(raised.value)


def test_gateway_config_rejects_unknown_capability(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"][0]["capabilities"] = ["chat", "vision"]

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "enum" in _error_types(raised.value)


def test_gateway_config_is_deeply_immutable(gateway_config: GatewayConfig) -> None:
    with pytest.raises(ValidationError):
        gateway_config.server.port = 9000

    assert isinstance(gateway_config.models, tuple)
    assert isinstance(gateway_config.models[0].capabilities, frozenset)
