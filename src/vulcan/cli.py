"""Command-line entry points for the loopback-only gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from vulcan.api import create_app
from vulcan.config import (
    HOSTED_PROVIDER_TYPES,
    ConfigLoadError,
    GatewayConfig,
    OllamaProviderConfig,
    load_config,
)
from vulcan.observability import configure_logging
from vulcan.providers.http import build_client, credential_available, verify_hosted_credential
from vulcan.usage import LedgerError


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
    ps = subcommands.add_parser(
        "ps",
        help="list resident models on each Ollama provider, mapped to configured aliases",
    )
    ps.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
    unload = subcommands.add_parser(
        "unload",
        help="unload the resident Ollama model backing an alias (one keep_alive 0 call)",
    )
    unload.add_argument("alias", help="configured alias on an ollama-typed provider")
    unload.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
    warmup = subcommands.add_parser(
        "warmup",
        help="pre-load the Ollama model backing an alias (one empty generate)",
    )
    warmup.add_argument("alias", help="configured alias on an ollama-typed provider")
    warmup.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
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


def _ollama_client(provider: OllamaProviderConfig) -> httpx.AsyncClient:
    """One hardened client for an explicit operator action against Ollama."""

    return build_client(base_url=provider.base_url, timeout_seconds=provider.timeout_seconds)


def _cli_error(code: str, message: str, *, retryable: bool) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "retryable": retryable}}


async def _ps_report(
    config: GatewayConfig,
    *,
    make_client: Callable[[OllamaProviderConfig], httpx.AsyncClient] | None = None,
) -> tuple[dict[str, Any], int]:
    """Resident models per Ollama provider, mapped back to configured aliases.

    Non-Ollama providers are listed as skipped and are never contacted. Native
    model names appear only in this report — it is the operator's own terminal
    and the config file itself holds them; unload/warmup output names the
    public alias only.
    """

    factory = make_client or _ollama_client
    alias_by_native = {(model.provider, model.provider_model): model.id for model in config.models}
    providers: list[dict[str, Any]] = []
    failures = 0
    for provider_id, provider in config.providers.items():
        if provider.type != "ollama":
            providers.append({"id": provider_id, "type": provider.type, "status": "skipped"})
            continue
        client = factory(provider)
        try:
            response = await client.get("/api/ps")
            entries = response.json().get("models") if response.is_success else None
            if not isinstance(entries, list):
                entries = None
        except (httpx.HTTPError, ValueError):
            entries = None
        finally:
            await client.aclose()
        if entries is None:
            failures += 1
            providers.append({"id": provider_id, "type": "ollama", "status": "unreachable"})
            continue
        resident: list[dict[str, Any]] = []
        for entry in entries:
            name = entry.get("name", "")
            alias = alias_by_native.get((provider_id, name))
            row: dict[str, Any] = {
                "name": name,
                "alias": alias,
                "size_vram": entry.get("size_vram"),
                "expires_at": entry.get("expires_at"),
            }
            if alias is None:
                row["unmanaged"] = True
            resident.append(row)
        providers.append({"id": provider_id, "type": "ollama", "status": "ok", "models": resident})
    return {"providers": providers}, (1 if failures else 0)


async def _alias_action(
    config: GatewayConfig,
    alias: str,
    action: str,
    *,
    make_client: Callable[[OllamaProviderConfig], httpx.AsyncClient] | None = None,
) -> tuple[dict[str, Any], int]:
    """One explicit operator action — unload or warmup — on one Ollama alias.

    Exactly one upstream call, only ever to the alias's own provider, and only
    ever an Ollama one; hosted or deterministic aliases are refused before any
    client is built. The response body is never read, and the native
    provider_model never appears in the output — classification is by status
    alone, so upstream text cannot leak into the operator's terminal.
    """

    factory = make_client or _ollama_client
    model = next((candidate for candidate in config.models if candidate.id == alias), None)
    if model is None:
        return _cli_error(
            "model_not_found",
            f"No configured alias named {alias!r}. Run: vulcan models --config <same config>.",
            retryable=False,
        ), 2
    provider = config.providers[model.provider]
    if provider.type != "ollama":
        return _cli_error(
            "unsupported_provider_type",
            f"Alias {alias!r} rides provider {model.provider!r} (type {provider.type!r}); "
            f"{action} applies only to ollama-typed providers.",
            retryable=False,
        ), 2

    body: dict[str, Any] = {"model": model.provider_model}
    if action == "unload":
        # Ollama's documented unload: keep_alive 0 expires residency now.
        body["keep_alive"] = 0
    else:
        # An empty non-streaming generate loads the model without generating.
        body["prompt"] = ""
        body["stream"] = False
        if model.keep_alive is not None:
            body["keep_alive"] = model.keep_alive

    client = factory(provider)
    try:
        response = await client.post("/api/generate", json=body)
    except httpx.HTTPError:
        return _cli_error(
            "provider_unreachable",
            f"Provider {model.provider!r} did not answer; is Ollama running?",
            retryable=True,
        ), 1
    finally:
        await client.aclose()

    if response.status_code == 404:
        return _cli_error(
            "model_unavailable",
            f"The model backing alias {alias!r} is not installed on provider {model.provider!r}.",
            retryable=False,
        ), 1
    if not response.is_success:
        return _cli_error(
            "provider_error",
            f"Provider {model.provider!r} answered HTTP {response.status_code}.",
            retryable=response.status_code >= 500,
        ), 1

    result: dict[str, Any] = {
        "alias": alias,
        "provider": model.provider,
        "action": action,
        "status": "unloaded" if action == "unload" else "warm",
    }
    if action == "warmup" and model.keep_alive is not None:
        result["keep_alive"] = model.keep_alive
    return result, 0


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

    if args.command == "ps":
        report, exit_code = asyncio.run(_ps_report(config))
        sys.stdout.write(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
        return exit_code

    if args.command in ("unload", "warmup"):
        payload, exit_code = asyncio.run(_alias_action(config, args.alias, args.command))
        stream = sys.stdout if exit_code == 0 else sys.stderr
        stream.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        return exit_code

    configure_logging(config.server.log_level)
    try:
        app = create_app(config)
    except LedgerError as exc:
        payload = {
            "error": {
                "code": "ledger_error",
                "message": (
                    f"The usage ledger at {exc.path} could not be opened "
                    f"({exc.reason}). Fix the path or remove the [usage] "
                    "section; Vulcan never silently falls back to in-memory "
                    "counters."
                ),
                "retryable": False,
            }
        }
        sys.stderr.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        return 2
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        access_log=False,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        log_config=None,
    )
    return 0
