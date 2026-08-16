"""Command-line entry points for the loopback-only gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from vulcan.api import create_app
from vulcan.config import HOSTED_PROVIDER_TYPES, ConfigLoadError, GatewayConfig, load_config
from vulcan.observability import configure_logging
from vulcan.providers.http import credential_available, verify_hosted_credential


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
    check.add_argument(
        "--verify-credentials",
        action="store_true",
        help=(
            "additionally make one metadata call per hosted provider to confirm the "
            "credential is accepted (never automatic; values and bodies are never printed)"
        ),
    )
    usage = subcommands.add_parser(
        "usage",
        help="print /v1/usage from the running gateway named by the config",
    )
    usage.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
    models = subcommands.add_parser(
        "models",
        help="print /v1/models from the running gateway named by the config",
    )
    models.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
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


async def _verify_hosted_credentials(config: GatewayConfig) -> dict[str, str]:
    """One metadata call per hosted provider; verdicts only, never values."""

    verdicts: dict[str, str] = {}
    for provider_id, provider in config.providers.items():
        if provider.type in HOSTED_PROVIDER_TYPES:
            verdicts[provider_id] = await verify_hosted_credential(provider)
    return verdicts


def _check_report(config: GatewayConfig, *, verify: bool = False) -> tuple[dict[str, Any], int]:
    """Build the safe check report: names and presence only, never values.

    With ``verify``, each hosted provider is additionally probed once with its
    configured credential. This is the only place Vulcan calls an authenticated
    endpoint outside serving a client request, and it happens solely because an
    operator asked for it.
    """

    verdicts = asyncio.run(_verify_hosted_credentials(config)) if verify else {}

    models_by_provider: dict[str, int] = {}
    for model in config.models:
        models_by_provider[model.provider] = models_by_provider.get(model.provider, 0) + 1

    providers: list[dict[str, Any]] = []
    credentials_missing = 0
    verification_failures = 0
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
        if provider_id in verdicts:
            entry["verification"] = verdicts[provider_id]
            if verdicts[provider_id] != "verified":
                verification_failures += 1
        providers.append(entry)

    report: dict[str, Any] = {
        "config": "valid",
        "schema_version": config.schema_version,
        "providers": providers,
        "models_configured": len(config.models),
        "credentials_missing": credentials_missing,
    }
    if verify:
        report["credentials_verified"] = verify
        report["verification_failures"] = verification_failures
    return report, (1 if credentials_missing or verification_failures else 0)


def _gateway_base_url(config: GatewayConfig) -> str:
    """The running gateway's base URL, IPv6-safe.

    ServerConfig accepts the IPv6 loopback ``::1``, which must be bracketed
    in a URL — ``http://::1:8140`` is invalid, ``http://[::1]:8140`` is not.
    """

    host = config.server.host
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{config.server.port}"


def _gateway_client(config: GatewayConfig) -> httpx.Client:
    """A hardened loopback client for reading the running gateway."""

    return httpx.Client(
        base_url=_gateway_base_url(config),
        timeout=httpx.Timeout(5.0),
        trust_env=False,
        follow_redirects=False,
    )


def _read_gateway(config: GatewayConfig, path: str) -> int:
    """GET one gateway endpoint and pass its JSON through verbatim.

    The gateway's responses are already content-safe; printing them unchanged
    adds no new surface. Failures are sanitized: the underlying exception text
    is never echoed, matching every other CLI error path.
    """

    try:
        with _gateway_client(config) as client:
            response = client.get(path)
    # InvalidURL is not an HTTPError; catching it keeps the sanitization
    # guarantee total even if a future host form slips past _gateway_base_url.
    except (httpx.HTTPError, httpx.InvalidURL):
        payload = {
            "error": {
                "code": "gateway_unreachable",
                "message": (
                    f"No running gateway answered at {_gateway_base_url(config)}{path}. "
                    "Start it with: vulcan serve --config <same config>."
                ),
                "retryable": True,
            }
        }
        sys.stderr.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        return 1

    sys.stdout.write(response.text.rstrip("\n") + "\n")
    return 0 if response.is_success else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        config = load_config(args.config)
    except ConfigLoadError as exc:
        _write_config_error(exc)
        return 2

    if args.command == "check":
        report, exit_code = _check_report(config, verify=args.verify_credentials)
        sys.stdout.write(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
        return exit_code

    if args.command == "usage":
        return _read_gateway(config, "/v1/usage")

    if args.command == "models":
        return _read_gateway(config, "/v1/models")

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
