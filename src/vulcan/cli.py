"""Command-line entry points for the loopback-only gateway."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn

from vulcan.api import create_app
from vulcan.config import ConfigLoadError, GatewayConfig, load_config
from vulcan.observability import configure_logging
from vulcan.providers.http import credential_available


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vulcan", description="Local-only AI inference gateway")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="start the local gateway")
    serve.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
    check = subcommands.add_parser(
        "check",
        help="validate a config and report credential availability without revealing values",
    )
    check.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
    return parser


def _write_config_error(exc: ConfigLoadError) -> None:
    payload = {
        "error": {
            "code": "configuration_error",
            "message": exc.reason,
            "retryable": False,
            "validation": [issue.model_dump(mode="json") for issue in exc.issues] or None,
        }
    }
    sys.stderr.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")


def _check_report(config: GatewayConfig) -> tuple[dict[str, Any], int]:
    """Build the safe check report: names and presence only, never values."""

    models_by_provider: dict[str, int] = {}
    for model in config.models:
        models_by_provider[model.provider] = models_by_provider.get(model.provider, 0) + 1

    providers: list[dict[str, Any]] = []
    credentials_missing = 0
    for provider_id, provider in config.providers.items():
        entry: dict[str, Any] = {
            "id": provider_id,
            "type": provider.type,
            "models": models_by_provider.get(provider_id, 0),
        }
        api_key_env = getattr(provider, "api_key_env", None)
        if api_key_env is not None:
            present = credential_available(api_key_env)
            entry["api_key_env"] = api_key_env
            entry["credential"] = "present" if present else "missing"
            if not present:
                credentials_missing += 1
        providers.append(entry)

    report = {
        "config": "valid",
        "schema_version": config.schema_version,
        "providers": providers,
        "models_configured": len(config.models),
        "credentials_missing": credentials_missing,
    }
    return report, (1 if credentials_missing else 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config)
    except ConfigLoadError as exc:
        _write_config_error(exc)
        return 2

    if args.command == "check":
        report, exit_code = _check_report(config)
        sys.stdout.write(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
        return exit_code

    configure_logging(config.server.log_level)
    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        log_config=None,
    )
    return 0
