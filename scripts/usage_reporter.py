"""Daily Vulcan usage digest, posted to Athena's signed forge ingest.

One run = one delivery. Reads cumulative counters from the gateway's
``GET /v1/usage``, diffs them against the previous snapshot, and posts a
compact, content-safe digest — request and token counts only, never prompt or
message text, matching the gateway's own logging rules.

Two live-service facts shape the design:

* ``/v1/usage`` is cumulative (process- or ledger-scoped); the API has no
  per-day window. Daily numbers come from a snapshot diff, with state kept in
  ``vulcan-data`` next to the ledger. The first run posts a ``baseline``; a
  counter regression or scope change (ledger lost, gateway reconfigured)
  posts a ``reset`` and reports cumulative values rather than inventing
  negative deltas.
* Athena's forge speaks one dialect (``github``) and lands a delivery only
  where it names a real issue key, so the digest is a synthetic ``push``
  event whose commit messages each name ``ATHENA_ISSUE_KEY``. It arrives as
  imported history on that issue's trail. Athena clips a fact's summary at
  200 characters, so every line is built under that bound and declares any
  overflow instead of being silently cut.

Configuration is environment-only, supplied by the unit's EnvironmentFile —
the source name and its one-time secret never touch the repo. Failure is
loud: an unreachable endpoint, a refused signature, or a delivery that lands
nowhere exits non-zero with a journald-visible line. There are no retries; a
missed window folds into the next run's delta.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TIMEOUT_SECONDS = 10.0
DEFAULT_VULCAN_USAGE_URL = "http://127.0.0.1:8140/v1/usage"
DEFAULT_ATHENA_BASE_URL = "http://100.125.80.91:8300"
DEFAULT_STATE_PATH = "/home/kevin-lee/deploy/vulcan-data/usage-reporter-state.json"

#: Athena clips a forge fact's summary here (forge_events.MAX_SUMMARY_CHARS).
MAX_MESSAGE_CHARS = 200
#: The synthetic branch the digest's push event claims; carries no key itself.
DIGEST_REF = "refs/heads/vulcan-usage-digest"
#: Matches Athena's issue-key shape (links.ID_DIGITS bounds the digits).
ISSUE_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*-\d{1,19}$")

COUNTER_FIELDS = (
    "requests",
    "requests_with_usage",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)


class ReporterError(RuntimeError):
    """A loud, operator-facing failure — the run exits non-zero."""


class ConfigError(ReporterError):
    """Environment missing or malformed — exit 2, like ``vulcan check``."""


@dataclass(frozen=True)
class Config:
    vulcan_usage_url: str
    athena_url: str
    secret: str
    issue_key: str
    state_path: Path


def _load_config() -> Config:
    missing = [
        name
        for name in ("ATHENA_FORGE_SOURCE", "ATHENA_FORGE_SECRET", "ATHENA_ISSUE_KEY")
        if not os.environ.get(name, "").strip()
    ]
    if missing:
        raise ConfigError(f"missing required environment: {', '.join(missing)}")
    issue_key = os.environ["ATHENA_ISSUE_KEY"].strip().upper()
    if not ISSUE_KEY_RE.match(issue_key):
        raise ConfigError(f"ATHENA_ISSUE_KEY {issue_key!r} is not shaped like an issue key")
    vulcan_usage_url = os.environ.get("VULCAN_USAGE_URL", DEFAULT_VULCAN_USAGE_URL)
    base_url = os.environ.get("ATHENA_BASE_URL", DEFAULT_ATHENA_BASE_URL).rstrip("/")
    for name, url in (("VULCAN_USAGE_URL", vulcan_usage_url), ("ATHENA_BASE_URL", base_url)):
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"{name} {url!r} must be an http(s) URL")
    source = os.environ["ATHENA_FORGE_SOURCE"].strip()
    return Config(
        vulcan_usage_url=vulcan_usage_url,
        athena_url=f"{base_url}/forge/{source}",
        secret=os.environ["ATHENA_FORGE_SECRET"].strip(),
        issue_key=issue_key,
        state_path=Path(os.environ.get("USAGE_REPORTER_STATE", DEFAULT_STATE_PATH)),
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a redirected POST is a misconfiguration, not a path."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener() -> urllib.request.OpenerDirector:
    # No proxy inheritance, no redirects — the same hardening as the gateway's
    # own upstream clients.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _fetch_usage(opener: urllib.request.OpenerDirector, url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ReporterError(f"vulcan /v1/usage returned HTTP {exc.code}") from exc
    except OSError as exc:  # URLError and timeouts are OSError subclasses
        raise ReporterError(f"vulcan unreachable at {url}: {exc}") from exc
    except ValueError as exc:
        raise ReporterError("vulcan /v1/usage did not return valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("totals"), dict):
        raise ReporterError("vulcan /v1/usage returned an unexpected shape")
    return payload


def _load_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as exc:
        raise ReporterError(
            f"snapshot {path} is unreadable ({exc}); delete it to re-baseline"
        ) from exc
    if (
        not isinstance(snapshot, dict)
        or not isinstance(snapshot.get("totals"), dict)
        or not isinstance(snapshot.get("fetched_at"), int | float)
    ):
        raise ReporterError(f"snapshot {path} has an unexpected shape; delete it to re-baseline")
    return snapshot


def _write_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f"{path.name}.tmp")
        temp_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temp_path, path)
    except OSError as exc:
        raise ReporterError(
            f"digest posted but snapshot {path} could not be written ({exc}); "
            "the next run will re-report this window"
        ) from exc


def _counters(row: dict[str, Any]) -> dict[str, int]:
    return {field: int(row.get(field, 0)) for field in COUNTER_FIELDS}


def _diff(current: dict[str, int], previous: dict[str, Any] | None) -> dict[str, int]:
    base = previous or {}
    return {field: current[field] - int(base.get(field, 0)) for field in COUNTER_FIELDS}


def _join_fitted(prefix: str, items: list[str]) -> str:
    """Join ``items`` after ``prefix`` under Athena's 200-char summary clip.

    Overflow is declared (``+N more``), never silently clipped: the server
    cuts a fact's summary at MAX_SUMMARY_CHARS, so the line must fit by
    construction.
    """
    kept: list[str] = []
    for index, item in enumerate(items):
        candidate = prefix + " · ".join([*kept, item])
        remaining = len(items) - index - 1
        if remaining:
            candidate += f" · +{remaining} more"
        if len(candidate) <= MAX_MESSAGE_CHARS:
            kept.append(item)
        else:
            break
    remaining = len(items) - len(kept)
    if not kept:
        return f"{prefix}+{len(items)} more"
    line = prefix + " · ".join(kept)
    if remaining:
        line += f" · +{remaining} more"
    if len(line) > MAX_MESSAGE_CHARS:  # pathological: the prefix alone overflows
        return f"{prefix}+{len(items)} more"
    return line


def _build_digest(
    usage: dict[str, Any], previous: dict[str, Any] | None, now: float, key: str
) -> tuple[list[str], dict[str, Any], str]:
    scope = str(usage.get("scope", "unknown"))
    snapshot: dict[str, Any] = {
        "fetched_at": now,
        "scope": scope,
        "totals": _counters(usage["totals"]),
        "by_seat": {
            row["seat"]: _counters(row.get("totals", {}))
            for row in usage.get("by_seat", [])
            if isinstance(row, dict) and row.get("seat")
        },
        "by_model": {
            row["model"]: _counters(row.get("totals", {}))
            for row in usage.get("by_model", [])
            if isinstance(row, dict) and row.get("model")
        },
    }
    if previous is None:
        kind = "baseline"
    elif scope != previous.get("scope") or (
        snapshot["totals"]["requests"] < int(previous["totals"].get("requests", 0))
    ):
        # The ledger restarts all counters together, so one regression check on
        # the global total is the whole reset detector.
        kind = "reset"
    else:
        kind = "delta"
    delta_mode = kind == "delta"
    previous_totals = previous["totals"] if previous is not None else None
    previous_seats = previous.get("by_seat", {}) if previous is not None else {}
    previous_models = previous.get("by_model", {}) if previous is not None else {}

    def per_name(
        current: dict[str, dict[str, int]], earlier: dict[str, Any]
    ) -> list[tuple[str, dict[str, int]]]:
        view = {
            name: _diff(counters, earlier.get(name)) if delta_mode else counters
            for name, counters in current.items()
        }
        if delta_mode:
            # A seat or alias quiet all window is not news.
            view = {name: c for name, c in view.items() if c["requests"] > 0}
        return sorted(view.items(), key=lambda item: (-item[1]["requests"], item[0]))

    totals = _diff(snapshot["totals"], previous_totals) if delta_mode else snapshot["totals"]
    tag = "" if delta_mode else f" [{kind}]"
    date_label = time.strftime("%Y-%m-%d", time.localtime(now))
    if delta_mode:
        hours = max(0, round((now - previous["fetched_at"]) / 3600))
        window = f"[delta {hours}h]"
    else:
        window = f"[{kind}]"
    messages = [
        f"{key} vulcan {date_label} {window}: {totals['requests']} req "
        f"({totals['requests_with_usage']} with usage) "
        f"{totals['prompt_tokens']}in/{totals['completion_tokens']}out tok"
    ]

    seat_rows = per_name(snapshot["by_seat"], previous_seats)
    seat_items = [f"{name} {c['requests']}r/{c['total_tokens']}t" for name, c in seat_rows]
    unlabeled_requests = totals["requests"] - sum(c["requests"] for _, c in seat_rows)
    if unlabeled_requests > 0:
        unlabeled_tokens = totals["total_tokens"] - sum(c["total_tokens"] for _, c in seat_rows)
        seat_items.append(f"unlabeled {unlabeled_requests}r/{unlabeled_tokens}t")
    if seat_items:
        messages.append(_join_fitted(f"{key} seats{tag}: ", seat_items))

    model_items = [
        f"{name} {c['requests']}r/{c['total_tokens']}t"
        for name, c in per_name(snapshot["by_model"], previous_models)
    ]
    if model_items:
        messages.append(_join_fitted(f"{key} aliases{tag}: ", model_items))

    ledger = usage.get("ledger")
    if isinstance(ledger, dict):
        ledger_part = (
            f"ledger: replayed {ledger.get('replayed_requests', 0)} "
            f"skipped {ledger.get('skipped_lines', 0)} "
            f"write_failures {ledger.get('write_failures', 0)}"
        )
    else:
        ledger_part = "ledger: off (process scope)"
    budget_items = []
    for budget in usage.get("budgets", []):
        if not isinstance(budget, dict):
            continue
        token_cap = budget.get("hosted_tokens_per_day")
        resets = budget.get("window_resets_at")
        reset_label = (
            time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime(resets))
            if isinstance(resets, int | float)
            else "unknown"
        )
        budget_items.append(
            f"{budget.get('seat', '?')} "
            f"{budget.get('requests_today', 0)}/{budget.get('hosted_requests_per_day', '?')}r "
            f"{budget.get('tokens_today', 0)}/{token_cap if token_cap is not None else '-'}t "
            f"resets {reset_label}"
        )
    if budget_items:
        messages.append(_join_fitted(f"{key} {ledger_part} | budgets today: ", budget_items))
    else:
        messages.append(f"{key} {ledger_part}")
    return messages, snapshot, kind


def _post_digest(
    config: Config, opener: urllib.request.OpenerDirector, messages: list[str]
) -> dict[str, Any]:
    payload = {"ref": DIGEST_REF, "commits": [{"message": m} for m in messages]}
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    signature = hmac.new(config.secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        config.athena_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            # The only dialect the live forge parses; without this header the
            # delivery is accepted (202) and lands nowhere.
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": f"sha256={signature}",
        },
    )
    try:
        with opener.open(request, timeout=TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:200].decode("utf-8", errors="replace")
        raise ReporterError(f"athena refused the digest: HTTP {exc.code} {detail}") from exc
    except OSError as exc:
        raise ReporterError(f"athena unreachable at {config.athena_url}: {exc}") from exc
    except ValueError as exc:
        raise ReporterError("athena returned a non-JSON response") from exc
    landed = result.get("landed") if isinstance(result, dict) else None
    if landed != len(messages):
        raise ReporterError(
            f"athena accepted the delivery but landed {landed!r} of {len(messages)} rows — "
            f"is {config.issue_key} a real issue key, and is the source enabled?"
        )
    return result


def main() -> int:
    try:
        config = _load_config()
    except ConfigError as exc:
        print(f"usage-reporter configuration error: {exc}", file=sys.stderr)
        return 2
    try:
        opener = _opener()
        usage = _fetch_usage(opener, config.vulcan_usage_url)
        now = time.time()
        previous = _load_snapshot(config.state_path)
        messages, snapshot, kind = _build_digest(usage, previous, now, config.issue_key)
        result = _post_digest(config, opener, messages)
        _write_snapshot(config.state_path, snapshot)
    except ReporterError as exc:
        print(f"usage-reporter FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"usage-reporter: {kind} digest posted — {result['landed']} rows on "
        f"{config.issue_key} via {config.athena_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
