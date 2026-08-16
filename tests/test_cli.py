"""CLI boundary tests that never start a listener or contact a provider."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import uvicorn

import vulcan.cli as cli
from vulcan.config import DeterministicProviderConfig, GatewayConfig

MULTI_PROVIDER_CONFIG = """\
schema_version = 2

[server]
host = "127.0.0.1"
port = 18140
log_level = "INFO"

[providers.local-ollama]
type = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[providers.openai]
type = "openai_compatible"
base_url = "https://api.example-vendor.com/v1"
api_key_env = "VULCAN_CLI_TEST_OPENAI_KEY"
timeout_seconds = 60.0

[providers.anthropic]
type = "anthropic"
api_key_env = "VULCAN_CLI_TEST_ANTHROPIC_KEY"
timeout_seconds = 60.0

[[models]]
id = "local-chat"
provider = "local-ollama"
provider_model = "some-installed-model"
capabilities = ["chat"]

[[models]]
id = "cloud-chat"
provider = "openai"
provider_model = "vendor-model"
capabilities = ["chat"]
"""


def _write_config(tmp_path: Path, content: str, *, name: str = "vulcan.toml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _forbid_server_start(*args: object, **kwargs: object) -> None:
    del args, kwargs
    pytest.fail("uvicorn.run must not be reached for invalid configuration")


def test_missing_config_is_one_sanitized_json_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_sentinel = "missing-config-secret-a321675f"
    missing_path = tmp_path / f"{path_sentinel}.toml"
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)

    exit_code = cli.main(["serve", "--config", str(missing_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    assert json.loads(captured.err) == {
        "error": {
            "code": "configuration_error",
            "message": "configuration file was not found",
            "retryable": False,
            "validation": None,
        }
    }
    assert path_sentinel not in captured.err
    assert str(missing_path) not in captured.err
    assert "Traceback" not in captured.err


def test_invalid_config_reports_paths_and_reason_codes_without_raw_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_sentinel = "raw-provider-password-b94e0ff1"
    runtime_sentinel = "raw-runtime-value-928e885f"
    config_path = _write_config(
        tmp_path,
        f"""\
schema_version = 2

[providers.local]
type = "ollama"
    timeout_seconds = 60
base_url = "http://local-user:{credential_sentinel}@127.0.0.1:11434"

[[models]]
id = "public-model"
provider = "local"
provider_model = "{runtime_sentinel}"
capabilities = ["chat"]
""",
    )
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)

    exit_code = cli.main(["serve", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload == {
        "error": {
            "code": "configuration_error",
            "message": "configuration validation failed",
            "retryable": False,
            "validation": [
                {
                    "path": "providers.local.ollama.base_url",
                    "reason": "value_error",
                }
            ],
        }
    }
    for raw_value in (
        credential_sentinel,
        runtime_sentinel,
        "local-user",
        "http://local-user",
    ):
        assert raw_value not in captured.err
    assert "Traceback" not in captured.err


def test_v1_config_fails_with_migration_guidance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(
        tmp_path,
        """\
schema_version = 1

[provider]
kind = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[[models]]
id = "local-chat"
runtime_name = "some-model"
capabilities = ["chat"]
""",
    )
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)

    exit_code = cli.main(["serve", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "configuration_error"
    assert "schema_version 2" in payload["error"]["message"]
    assert "provider_model" in payload["error"]["message"]
    assert "Traceback" not in captured.err


def test_malformed_toml_never_echoes_source_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sentinel = "malformed-source-secret-56bfe272"
    config_path = _write_config(
        tmp_path,
        f'[providers.det]\ntype = "deterministic"\nresponse_text = "{source_sentinel}\n',
    )
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)

    exit_code = cli.main(["serve", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "error": {
            "code": "configuration_error",
            "message": "configuration file is not valid UTF-8 TOML",
            "retryable": False,
            "validation": None,
        }
    }
    assert source_sentinel not in captured.err
    assert "Traceback" not in captured.err


def test_valid_cli_passes_loopback_and_hardening_options_without_starting_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_sentinel = "configured-no-io-response-6a1360c1"
    config_path = _write_config(
        tmp_path,
        f"""\
schema_version = 2

[server]
host = "127.0.0.1"
port = 18140
log_level = "INFO"

[providers.det]
type = "deterministic"
response_text = "{response_sentinel}"

[[models]]
id = "local-test"
provider = "det"
provider_model = "local-test-runtime"
capabilities = ["chat"]
""",
    )
    calls: dict[str, object] = {}
    fake_app = object()

    def fake_create_app(config: GatewayConfig) -> object:
        calls["config"] = config
        return fake_app

    def fake_configure_logging(level: str) -> None:
        calls["log_level"] = level

    def fake_uvicorn_run(app: object, **kwargs: object) -> None:
        calls["app"] = app
        calls["uvicorn_kwargs"] = kwargs

    monkeypatch.setattr(cli, "create_app", fake_create_app)
    monkeypatch.setattr(cli, "configure_logging", fake_configure_logging)
    monkeypatch.setattr(uvicorn, "run", fake_uvicorn_run)

    exit_code = cli.main(["serve", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert calls["app"] is fake_app
    assert calls["log_level"] == "INFO"
    assert calls["uvicorn_kwargs"] == {
        "host": "127.0.0.1",
        "port": 18140,
        "access_log": False,
        "proxy_headers": False,
        "server_header": False,
        "date_header": False,
        "log_config": None,
    }
    config = calls["config"]
    assert isinstance(config, GatewayConfig)
    provider = config.providers["det"]
    assert isinstance(provider, DeterministicProviderConfig)
    assert provider.response_text == response_sentinel


# ── vulcan check: credential availability without value exposure ─────────────


def test_check_reports_present_and_missing_credentials_without_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_sentinel = "sk-super-secret-value-77aa88bb"
    monkeypatch.setenv("VULCAN_CLI_TEST_OPENAI_KEY", secret_sentinel)
    monkeypatch.delenv("VULCAN_CLI_TEST_ANTHROPIC_KEY", raising=False)
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)
    config_path = _write_config(tmp_path, MULTI_PROVIDER_CONFIG)

    exit_code = cli.main(["check", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1  # valid config, one credential missing
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["config"] == "valid"
    assert report["schema_version"] == 2
    assert report["models_configured"] == 2
    assert report["credentials_missing"] == 1
    by_id = {entry["id"]: entry for entry in report["providers"]}
    assert by_id["local-ollama"] == {"id": "local-ollama", "type": "ollama", "models": 1}
    assert by_id["openai"] == {
        "id": "openai",
        "type": "openai_compatible",
        "models": 1,
        "api_key_env": "VULCAN_CLI_TEST_OPENAI_KEY",
        "credential": "present",
    }
    assert by_id["anthropic"] == {
        "id": "anthropic",
        "type": "anthropic",
        "models": 0,
        "api_key_env": "VULCAN_CLI_TEST_ANTHROPIC_KEY",
        "credential": "missing",
    }
    assert secret_sentinel not in captured.out


def test_check_exits_zero_when_all_credentials_are_present(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VULCAN_CLI_TEST_OPENAI_KEY", "sk-present-one")
    monkeypatch.setenv("VULCAN_CLI_TEST_ANTHROPIC_KEY", "sk-present-two")
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)
    config_path = _write_config(tmp_path, MULTI_PROVIDER_CONFIG)

    exit_code = cli.main(["check", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    report = json.loads(captured.out)
    assert report["credentials_missing"] == 0
    assert "sk-present-one" not in captured.out
    assert "sk-present-two" not in captured.out


@pytest.mark.parametrize("unusable", ["", "   ", "with space", "line\nbreak", "tab\tchar"])
def test_check_treats_blank_or_non_printable_values_as_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    unusable: str,
) -> None:
    monkeypatch.setenv("VULCAN_CLI_TEST_OPENAI_KEY", unusable)
    monkeypatch.setenv("VULCAN_CLI_TEST_ANTHROPIC_KEY", "sk-usable")
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)
    config_path = _write_config(tmp_path, MULTI_PROVIDER_CONFIG)

    exit_code = cli.main(["check", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    report = json.loads(captured.out)
    by_id = {entry["id"]: entry for entry in report["providers"]}
    assert by_id["openai"]["credential"] == "missing"


def test_check_invalid_config_uses_the_same_sanitized_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uvicorn, "run", _forbid_server_start)
    config_path = _write_config(tmp_path, "schema_version = 2\n")

    exit_code = cli.main(["check", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "configuration_error"


# ── Gateway read subcommands: usage and models ───────────────────────────────

READ_ERROR_SENTINEL = "transport-detail-must-not-escape-4f19"


def _read_config(tmp_path: Path) -> Path:
    return _write_config(
        tmp_path,
        """
        schema_version = 2

        [server]
        host = "127.0.0.1"
        port = 8140

        [providers.det]
        type = "deterministic"
        response_text = "canned"

        [[models]]
        id = "alias-one"
        provider = "det"
        provider_model = "native"
        capabilities = ["chat"]
        """,
    )


def _mock_gateway_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: object,
) -> None:
    def factory(config: object) -> httpx.Client:
        return httpx.Client(
            base_url="http://127.0.0.1:8140",
            transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        )

    monkeypatch.setattr(cli, "_gateway_client", factory)


def test_usage_subcommand_passes_gateway_json_through_verbatim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"object": "usage", "by_seat": [{"seat": "fable", "totals": {"requests": 1}}]}
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json=body)

    _mock_gateway_client(monkeypatch, handler)

    exit_code = cli.main(["usage", "--config", str(_read_config(tmp_path))])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert seen_paths == ["/v1/usage"]
    assert json.loads(captured.out) == body
    assert captured.err == ""


def test_models_subcommand_reads_v1_models(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json={"object": "list", "data": []})

    _mock_gateway_client(monkeypatch, handler)

    exit_code = cli.main(["models", "--config", str(_read_config(tmp_path))])

    assert exit_code == 0
    assert seen_paths == ["/v1/models"]


def test_gateway_read_failure_is_sanitized_and_names_the_fix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(READ_ERROR_SENTINEL)

    _mock_gateway_client(monkeypatch, handler)

    exit_code = cli.main(["usage", "--config", str(_read_config(tmp_path))])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["error"]["code"] == "gateway_unreachable"
    assert payload["error"]["retryable"] is True
    assert "vulcan serve" in payload["error"]["message"]
    # Sanitized like every other CLI error: the transport detail never escapes.
    assert READ_ERROR_SENTINEL not in captured.err


def test_gateway_read_passes_non_success_status_through_with_exit_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "not_ready"}})

    _mock_gateway_client(monkeypatch, handler)

    exit_code = cli.main(["usage", "--config", str(_read_config(tmp_path))])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert json.loads(captured.out)["error"]["code"] == "not_ready"


def test_gateway_read_with_invalid_config_uses_the_same_sanitized_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "nope.toml"

    exit_code = cli.main(["usage", "--config", str(missing)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert json.loads(captured.err)["error"]["code"] == "configuration_error"
