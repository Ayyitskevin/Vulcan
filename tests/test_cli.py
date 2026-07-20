"""CLI boundary tests that never start a listener or contact a provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import uvicorn

import vulcan.cli as cli
from vulcan.config import DeterministicProviderConfig, GatewayConfig


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
schema_version = 1

[provider]
kind = "ollama"
    timeout_seconds = 60
base_url = "http://local-user:{credential_sentinel}@127.0.0.1:11434"

[[models]]
id = "public-model"
runtime_name = "{runtime_sentinel}"
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
                    "path": "provider.ollama.base_url",
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


def test_malformed_toml_never_echoes_source_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_sentinel = "malformed-source-secret-56bfe272"
    config_path = _write_config(
        tmp_path,
        f'[provider]\nkind = "deterministic"\nresponse_text = "{source_sentinel}\n',
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
schema_version = 1

[server]
host = "127.0.0.1"
port = 18140
log_level = "INFO"

[provider]
kind = "deterministic"
response_text = "{response_sentinel}"

[[models]]
id = "local-test"
runtime_name = "local-test-runtime"
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
    assert isinstance(config.provider, DeterministicProviderConfig)
    assert config.provider.response_text == response_sentinel
