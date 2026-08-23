"""Live contract check against the RUNNING gateway and its real Ollama.

Gate command 5 — operator-run on the deployment host, never CI. The pytest
suite may not contact a real API (ROADMAP §1.4), which means it can only pin
what we BELIEVE Ollama's contract is; this script is the counterpart that
proves the belief against the real thing. It exists because exactly that gap
shipped a bug: the suite's mock transport accepted `/api/generate` for an
embedding alias that real Ollama refuses (fixed in ac2889f).

What it exercises, end to end, using the same TOML the service runs on:

* `/healthz` answers and every ollama-typed provider is `available`;
* `/v1/models` lists every configured alias;
* one real chat completion and one real embedding through the gateway
  (transport contract only — a well-formed response with usage counts;
  model output quality is deliberately not judged);
* the memory-lifecycle CLI round trip on an embedding-only alias — warmup,
  resident in `ps`, unload, gone from `ps` — the exact surface the mock
  suite could not defend;
* `/v1/usage` totals grew by at least the gateway calls made here.

Side effects are declared, not hidden: the embedding model is loaded and
unloaded once; when it was already resident beforehand it is warmed again at
the end and the output says so. Output is one content-safe JSON line —
counts, aliases, and booleans, never message or embedding content.

Usage: uv run python scripts/live_check.py --config <live vulcan.toml>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

GATEWAY_TIMEOUT_SECONDS = 120.0  # a cold model may need a real load
CLI_TIMEOUT_SECONDS = 120.0
UNLOAD_SETTLE_SECONDS = 5.0


class CheckFailure(SystemExit):
    """One failed check — the message is the whole diagnosis, exit 1."""

    def __init__(self, message: str) -> None:
        super().__init__(f"live-check FAILED: {message}")


def _load_live_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CheckFailure(f"cannot read config {path}: {exc}") from exc


def _pick_aliases(config: dict[str, Any]) -> tuple[str, str]:
    """First ollama-backed chat alias and first ollama-backed embeddings-only
    alias — the two shapes whose upstream contracts differ."""

    ollama_providers = {
        provider_id
        for provider_id, provider in config.get("providers", {}).items()
        if provider.get("type") == "ollama"
    }
    if not ollama_providers:
        raise CheckFailure("config has no ollama-typed provider; nothing to live-check")
    chat_alias = embed_alias = None
    for model in config.get("models", []):
        if model.get("provider") not in ollama_providers:
            continue
        capabilities = set(model.get("capabilities", []))
        if chat_alias is None and "chat" in capabilities:
            chat_alias = model["id"]
        if embed_alias is None and capabilities == {"embeddings"}:
            embed_alias = model["id"]
    if chat_alias is None or embed_alias is None:
        raise CheckFailure(
            "need one ollama chat alias and one ollama embeddings-only alias "
            f"(found chat={chat_alias!r}, embed={embed_alias!r})"
        )
    return chat_alias, embed_alias


def _request(base_url: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise CheckFailure(f"{path} returned HTTP {exc.code}") from exc
    except OSError as exc:
        raise CheckFailure(f"gateway unreachable at {base_url}{path}: {exc}") from exc


def _cli(config_path: Path, *arguments: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "vulcan", *arguments, "--config", str(config_path)]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        # CLI errors are already sanitized; forward them verbatim.
        detail = (completed.stderr or completed.stdout).strip()
        raise CheckFailure(f"vulcan {' '.join(arguments)} exited {completed.returncode}: {detail}")
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise CheckFailure(f"vulcan {' '.join(arguments)} printed non-JSON output") from exc


def _resident_aliases(ps_report: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for provider in ps_report.get("providers", []):
        for row in provider.get("models", []) if provider.get("status") == "ok" else []:
            if row.get("alias"):
                aliases.add(row["alias"])
    return aliases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the RUNNING service's TOML")
    arguments = parser.parse_args()

    config = _load_live_config(arguments.config)
    server = config.get("server", {})
    base_url = f"http://{server.get('host', '127.0.0.1')}:{server.get('port', 8140)}"
    chat_alias, embed_alias = _pick_aliases(config)

    health = _request(base_url, "/healthz")
    if health.get("status") != "ok":
        raise CheckFailure(f"healthz status is {health.get('status')!r}")
    unavailable = [
        provider["id"]
        for provider in health.get("providers", [])
        if provider.get("type") == "ollama" and provider.get("availability") != "available"
    ]
    if unavailable:
        raise CheckFailure(f"ollama provider(s) not available: {unavailable}")

    listed = {model["id"] for model in _request(base_url, "/v1/models").get("data", [])}
    configured = {model["id"] for model in config.get("models", [])}
    if listed != configured:
        raise CheckFailure(f"/v1/models drift: config-only {configured - listed}")

    # Baseline BEFORE any gateway inference: the embeddings call below loads
    # the embed model itself, so a later reading would always say "resident".
    was_resident_before = embed_alias in _resident_aliases(_cli(arguments.config, "ps"))

    usage_before = _request(base_url, "/v1/usage")["totals"]["requests"]

    # Transport contract only: a well-formed choice and real token counts.
    # Content quality (or emptiness — reasoning models may spend the whole
    # budget thinking) is deliberately not this script's business.
    chat = _request(
        base_url,
        "/v1/chat/completions",
        {
            "model": chat_alias,
            "messages": [{"role": "user", "content": "Reply with one word."}],
            "max_tokens": 32,
        },
    )
    choice = chat["choices"][0]
    if choice["message"]["role"] != "assistant" or chat["usage"]["prompt_tokens"] <= 0:
        raise CheckFailure(f"chat via {chat_alias!r} returned a malformed completion")

    vectors = _request(base_url, "/v1/embeddings", {"model": embed_alias, "input": "live check"})[
        "data"
    ]
    dimensions = len(vectors[0]["embedding"]) if vectors else 0
    if dimensions <= 0:
        raise CheckFailure(f"embeddings via {embed_alias!r} returned no vector")

    warm = _cli(arguments.config, "warmup", embed_alias)
    if warm.get("status") != "warm":
        raise CheckFailure(f"warmup {embed_alias!r} answered {warm!r}")
    if embed_alias not in _resident_aliases(_cli(arguments.config, "ps")):
        raise CheckFailure(f"{embed_alias!r} not resident after warmup")

    unloaded = _cli(arguments.config, "unload", embed_alias)
    if unloaded.get("status") != "unloaded":
        raise CheckFailure(f"unload {embed_alias!r} answered {unloaded!r}")
    deadline = time.monotonic() + UNLOAD_SETTLE_SECONDS
    while embed_alias in _resident_aliases(_cli(arguments.config, "ps")):
        if time.monotonic() > deadline:
            raise CheckFailure(f"{embed_alias!r} still resident after unload")
        time.sleep(0.5)

    restored = False
    if was_resident_before:
        # Leave the box as found: it was resident when we arrived.
        restored = _cli(arguments.config, "warmup", embed_alias).get("status") == "warm"
        if not restored:
            raise CheckFailure(f"could not re-warm previously resident {embed_alias!r}")

    usage_after = _request(base_url, "/v1/usage")["totals"]["requests"]
    # >= because other live callers may land requests while this runs.
    if usage_after - usage_before < 2:
        raise CheckFailure(
            f"usage counted {usage_after - usage_before} new requests; expected the "
            "2 this script made through the gateway"
        )

    print(
        json.dumps(
            {
                "gateway": base_url,
                "chat_alias": chat_alias,
                "chat_finish_reason": choice.get("finish_reason"),
                "embed_alias": embed_alias,
                "embedding_dimensions": dimensions,
                "lifecycle_round_trip": True,
                "embed_was_resident_before": was_resident_before,
                "embed_restored": restored,
                "usage_requests_delta": usage_after - usage_before,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
