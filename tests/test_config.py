from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vulcan.config import (
    AnthropicProviderConfig,
    ConfigLoadError,
    GatewayConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    ReadinessConfig,
    ServerConfig,
    load_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _error_types(error: ValidationError) -> set[str]:
    return {item["type"] for item in error.errors(include_url=False, include_input=False)}


def test_server_defaults_to_ipv4_loopback() -> None:
    server = ServerConfig()

    assert server.host == "127.0.0.1"
    assert server.port == 8140


def test_readiness_defaults_and_bounds() -> None:
    assert ReadinessConfig().probe_ttl_seconds == 5.0
    assert ReadinessConfig(probe_ttl_seconds=0.0).probe_ttl_seconds == 0.0
    assert ReadinessConfig(probe_ttl_seconds=60.0).probe_ttl_seconds == 60.0
    with pytest.raises(ValidationError):
        ReadinessConfig(probe_ttl_seconds=-0.1)
    with pytest.raises(ValidationError):
        ReadinessConfig(probe_ttl_seconds=60.1)
    with pytest.raises(ValidationError):
        ReadinessConfig(probe_ttl_seconds=True)  # type: ignore[arg-type]


def test_gateway_config_defaults_readiness_when_section_omitted(
    valid_config_document: dict[str, Any],
) -> None:
    config = GatewayConfig.model_validate(valid_config_document)
    assert config.readiness.probe_ttl_seconds == 5.0


def test_gateway_config_accepts_readiness_probe_ttl(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["readiness"] = {"probe_ttl_seconds": 0.0}
    config = GatewayConfig.model_validate(valid_config_document)
    assert config.readiness.probe_ttl_seconds == 0.0


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
        type="ollama",
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
            type="ollama",
            base_url=base_url,
            timeout_seconds=60,
        )


@pytest.mark.parametrize("timeout_seconds", [0.09, 300.01])
def test_ollama_provider_rejects_out_of_range_timeouts(timeout_seconds: float) -> None:
    with pytest.raises(ValidationError):
        OllamaProviderConfig(
            type="ollama",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=timeout_seconds,
        )


def test_ollama_provider_accepts_integer_timeout_without_boolean_coercion() -> None:
    provider = OllamaProviderConfig(
        type="ollama",
        base_url="http://127.0.0.1:11434",
        timeout_seconds=60,
    )

    assert provider.timeout_seconds == 60.0


def test_ollama_provider_rejects_boolean_timeout() -> None:
    with pytest.raises(ValidationError) as raised:
        OllamaProviderConfig(
            type="ollama",
            base_url="http://127.0.0.1:11434",
            timeout_seconds=True,
        )

    assert "boolean_not_allowed" in _error_types(raised.value)


def test_ollama_provider_rejects_string_timeout() -> None:
    with pytest.raises(ValidationError) as raised:
        OllamaProviderConfig(
            type="ollama",
            base_url="http://127.0.0.1:11434",
            timeout_seconds="60",  # type: ignore[arg-type]
        )

    assert "float_type" in _error_types(raised.value)


# ── Hosted provider configuration (BYOK) ─────────────────────────────────────


def _openai_compatible(**overrides: Any) -> OpenAICompatibleProviderConfig:
    fields: dict[str, Any] = {
        "type": "openai_compatible",
        "base_url": "https://api.example-vendor.com/v1",
        "api_key_env": "VULCAN_TEST_API_KEY",
        "timeout_seconds": 30.0,
    }
    fields.update(overrides)
    return OpenAICompatibleProviderConfig(**fields)


@pytest.mark.parametrize(
    ("configured", "normalized"),
    [
        ("https://api.example-vendor.com/v1", "https://api.example-vendor.com/v1"),
        ("https://api.example-vendor.com/v1/", "https://api.example-vendor.com/v1"),
        (
            "https://api.example-vendor.com/api/paas/v4",
            "https://api.example-vendor.com/api/paas/v4",
        ),
        ("https://api.example-vendor.com", "https://api.example-vendor.com"),
        # Cleartext HTTP is loopback-only, for local mocks and proxies.
        ("http://127.0.0.1:9999/v1", "http://127.0.0.1:9999/v1"),
        ("http://localhost:9999", "http://localhost:9999"),
    ],
)
def test_hosted_provider_accepts_explicit_https_or_loopback_http(
    configured: str, normalized: str
) -> None:
    assert _openai_compatible(base_url=configured).base_url == normalized


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example-vendor.com/v1",  # cleartext off-loopback
        "https://user:password@api.example-vendor.com/v1",  # URL credentials
        "https://api.example-vendor.com/v1?token=x",  # query
        "https://api.example-vendor.com/v1#frag",  # fragment
        "https://api.example-vendor.com//v1",  # empty path segment
        "ftp://api.example-vendor.com/v1",
        "https:///missing-host",
        "https://api.example-vendor.com:0/v1",
        "https://api.example-vendor.com:99999/v1",
    ],
)
def test_hosted_provider_rejects_unsafe_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        _openai_compatible(base_url=base_url)


@pytest.mark.parametrize(
    "api_key_env",
    [
        "lowercase_key",
        "1LEADING_DIGIT",
        "_LEADING_UNDERSCORE",
        "WITH-HYPHEN",
        "WITH SPACE",
        "WITH.DOT",
        "",
        "$INJECTION",
    ],
)
def test_hosted_provider_rejects_malformed_env_var_names(api_key_env: str) -> None:
    with pytest.raises(ValidationError):
        _openai_compatible(api_key_env=api_key_env)


def test_hosted_provider_requires_api_key_env_and_never_accepts_inline_keys() -> None:
    with pytest.raises(ValidationError) as missing:
        OpenAICompatibleProviderConfig(
            type="openai_compatible",
            base_url="https://api.example-vendor.com/v1",
            timeout_seconds=30.0,
        )
    assert "missing" in _error_types(missing.value)

    with pytest.raises(ValidationError) as inline:
        _openai_compatible(api_key="sk-raw-secret-must-not-be-supported")
    assert "extra_forbidden" in _error_types(inline.value)


def test_openai_compatible_max_tokens_field_is_constrained() -> None:
    assert _openai_compatible().max_tokens_field == "max_tokens"
    assert (
        _openai_compatible(max_tokens_field="max_completion_tokens").max_tokens_field
        == "max_completion_tokens"
    )
    with pytest.raises(ValidationError):
        _openai_compatible(max_tokens_field="tokens_limit")


def test_anthropic_provider_defaults_and_bounds() -> None:
    provider = AnthropicProviderConfig(
        type="anthropic",
        api_key_env="VULCAN_ANTHROPIC_API_KEY",
        timeout_seconds=30.0,
    )
    assert provider.base_url == "https://api.anthropic.com"
    assert provider.default_max_tokens == 4096

    with pytest.raises(ValidationError):
        AnthropicProviderConfig(
            type="anthropic",
            api_key_env="VULCAN_ANTHROPIC_API_KEY",
            timeout_seconds=30.0,
            default_max_tokens=0,
        )
    with pytest.raises(ValidationError):
        AnthropicProviderConfig(
            type="anthropic",
            api_key_env="VULCAN_ANTHROPIC_API_KEY",
            timeout_seconds=30.0,
            default_max_tokens=32769,
        )


# ── Whole-document validation ────────────────────────────────────────────────


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
        b"[providers.p\ntype = 'ollama'",
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
schema_version = 2
undocumented = "{sentinel}"

[providers.det]
type = "deterministic"
response_text = "safe-response"

[[models]]
id = "chat-model"
provider = "det"
provider_model = "runtime-model"
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


def test_load_config_parses_the_shipped_multi_provider_example() -> None:
    config = load_config(REPO_ROOT / "config" / "vulcan.example.toml")

    assert config.schema_version == 2
    assert {provider.type for provider in config.providers.values()} == {
        "ollama",
        "anthropic",
        "openai_compatible",
    }
    # Every alias resolves to a configured provider, and no secrets are inline.
    for model in config.models:
        assert model.provider in config.providers
    for provider in config.providers.values():
        api_key_env = getattr(provider, "api_key_env", None)
        if api_key_env is not None:
            assert api_key_env.startswith("VULCAN_")
    # The example demonstrates the optional class → alias routing labels.
    classes = {model.id: model.class_ for model in config.models}
    assert classes["local-chat"] == "code"
    assert classes["kimi-chat"] == "hosted-chat"


def test_load_config_parses_the_shipped_deterministic_example() -> None:
    config = load_config(REPO_ROOT / "config" / "vulcan.deterministic.toml")

    assert config.schema_version == 2
    assert list(config.providers) == ["deterministic"]


# ── v1 migration rejection ───────────────────────────────────────────────────


def test_load_config_rejects_schema_v1_with_migration_guidance(tmp_path: Path) -> None:
    path = tmp_path / "vulcan.toml"
    path.write_text(
        """
schema_version = 1

[provider]
kind = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[[models]]
id = "local-chat"
runtime_name = "some-model"
capabilities = ["chat"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError) as raised:
        load_config(path)

    reason = raised.value.reason
    assert "schema_version 1" in reason
    assert "schema_version 2" in reason
    assert "provider_model" in reason
    assert raised.value.issues == ()


def test_load_config_recognizes_legacy_provider_table_without_version(tmp_path: Path) -> None:
    path = tmp_path / "vulcan.toml"
    path.write_text(
        """
schema_version = 2

[provider]
kind = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[[models]]
id = "local-chat"
runtime_name = "some-model"
capabilities = ["chat"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigLoadError) as raised:
        load_config(path)

    assert "schema_version 2" in raised.value.reason


@pytest.mark.parametrize("missing_field", ["schema_version", "providers", "models"])
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


def test_gateway_config_rejects_unknown_provider_type(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["providers"]["det"]["type"] = "automatic-cloud-fallback"

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "union_tag_invalid" in _error_types(raised.value)


def test_gateway_config_rejects_unknown_nested_fields(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["providers"]["det"]["api_key"] = "must-not-be-supported"

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "extra_forbidden" in _error_types(raised.value)


@pytest.mark.parametrize(
    "provider_id",
    ["UPPER", "has space", "has/slash", "-leading-hyphen", "_leading_underscore", "é"],
)
def test_gateway_config_rejects_malformed_provider_ids(
    valid_config_document: dict[str, Any],
    provider_id: str,
) -> None:
    valid_config_document["providers"][provider_id] = {
        "type": "deterministic",
        "response_text": "x",
    }

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "invalid_provider_id" in _error_types(raised.value)


def test_gateway_config_rejects_model_referencing_unknown_provider(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"][0]["provider"] = "not-configured"

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "unknown_provider_reference" in _error_types(raised.value)


def test_gateway_config_rejects_empty_provider_table(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["providers"] = {}

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "too_short" in _error_types(raised.value)


def test_gateway_config_accepts_multiple_named_providers(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["providers"]["openai"] = {
        "type": "openai_compatible",
        "base_url": "https://api.example-vendor.com/v1",
        "api_key_env": "VULCAN_TEST_API_KEY",
        "timeout_seconds": 30.0,
    }
    valid_config_document["providers"]["anthropic"] = {
        "type": "anthropic",
        "api_key_env": "VULCAN_ANTHROPIC_API_KEY",
        "timeout_seconds": 30.0,
    }
    valid_config_document["models"].append(
        {
            "id": "cloud-model",
            "provider": "openai",
            "provider_model": "vendor-native-name",
            "capabilities": ["chat"],
        }
    )

    config = GatewayConfig.model_validate(valid_config_document)

    assert set(config.providers) == {"det", "openai", "anthropic"}
    assert config.models[1].provider == "openai"


def test_gateway_config_rejects_unknown_schema_version(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["schema_version"] = 3

    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(valid_config_document)


def test_gateway_config_rejects_schema_version_one(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["schema_version"] = 1

    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(valid_config_document)


def test_gateway_config_rejects_boolean_schema_version(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["schema_version"] = True

    with pytest.raises(ValidationError) as raised:
        GatewayConfig.model_validate(valid_config_document)

    assert "boolean_not_allowed" in _error_types(raised.value)


@pytest.mark.parametrize("schema_version", [2.0, "2"])
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
    duplicate["provider_model"] = "another-runtime-model"
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


# ── Optional per-alias class label ────────────────────────────────────────────


def test_model_class_label_defaults_to_none_when_omitted(
    valid_config_document: dict[str, Any],
) -> None:
    config = GatewayConfig.model_validate(valid_config_document)

    assert config.models[0].class_ is None


def test_model_class_label_accepts_a_free_form_short_string(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"][0]["class"] = "code-fast"

    config = GatewayConfig.model_validate(valid_config_document)

    assert config.models[0].class_ == "code-fast"


def test_model_class_label_is_bounded(
    valid_config_document: dict[str, Any],
) -> None:
    valid_config_document["models"][0]["class"] = "x" * 65

    with pytest.raises(ValidationError):
        GatewayConfig.model_validate(valid_config_document)


def test_gateway_config_is_deeply_immutable(gateway_config: GatewayConfig) -> None:
    with pytest.raises(ValidationError):
        gateway_config.server.port = 9000

    assert isinstance(gateway_config.models, tuple)
    assert isinstance(gateway_config.models[0].capabilities, frozenset)
