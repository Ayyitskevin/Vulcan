"""Command-line entry point for the loopback-only server."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from vulcan.api import create_app
from vulcan.config import ConfigLoadError, load_config
from vulcan.observability import configure_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vulcan", description="Local-only AI inference gateway")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="start the local gateway")
    serve.add_argument("--config", type=Path, required=True, help="path to a Vulcan TOML config")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.command != "serve":
        return 2
    try:
        config = load_config(args.config)
    except ConfigLoadError as exc:
        _write_config_error(exc)
        return 2

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
