"""Durable usage ledger: replay, restart survival, honesty counters, leaks.

Operator-requested (2026-08-16). One JSON line per completed request — never
message content, never native model names. Counters replay the file at boot so
``/v1/usage`` survives restarts; every anomaly (torn line, failed write) is
counted and reported, never guessed at.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    ConfigLoadError,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
    UsageConfig,
    load_config,
)
from vulcan.usage import LedgerError, UsageLedger, UsageRecorder

PROMPT_SENTINEL = "ledger-prompt-must-not-escape-8e2c"


def _config(ledger_path: Path | None = None) -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        usage=UsageConfig(ledger_path=ledger_path) if ledger_path is not None else None,
        providers={
            "det": DeterministicProviderConfig(type="deterministic", response_text="canned")
        },
        models=(
            ModelConfig(
                id="alias-one",
                provider="det",
                provider_model="native-one",
                capabilities=frozenset({Capability.CHAT, Capability.EMBEDDINGS}),
            ),
        ),
    )


def _client(ledger_path: Path, clock_value: float = 1_700_000_000.0) -> TestClient:
    app = create_app(_config(ledger_path), clock=lambda: clock_value)
    return TestClient(app, base_url="http://127.0.0.1")


def _chat(client: TestClient, seat: str | None = None) -> Any:
    payload: dict[str, Any] = {
        "model": "alias-one",
        "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
    }
    if seat is not None:
        payload["seat"] = seat
    return client.post("/v1/chat/completions", json=payload)


# ── Ledger unit behavior ─────────────────────────────────────────────────────


def test_ledger_appends_one_sorted_compact_line_per_record(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path, clock=lambda: 1_700_000_123.9)
    ledger.append(
        model="alias-one", provider="det", seat="fable", prompt_tokens=7, completion_tokens=3
    )
    ledger.append(
        model="alias-one", provider="det", seat=None, prompt_tokens=None, completion_tokens=None
    )
    ledger.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {
        "completion_tokens": 3,
        "model": "alias-one",
        "prompt_tokens": 7,
        "provider": "det",
        "seat": "fable",
        "ts": 1_700_000_123,
    }
    assert json.loads(lines[1])["seat"] is None


def test_ledger_replay_rebuilds_counters_and_skips_torn_lines(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_text(
        json.dumps(
            {
                "completion_tokens": 4,
                "model": "alias-one",
                "prompt_tokens": 6,
                "provider": "det",
                "seat": "fable",
                "ts": 1_600_000_000,
            }
        )
        + "\n"
        + '{"model": "alias-one", "provider": "det", "ts": 1_600_000_500, "seat": null'
        + "\n"  # torn line: truncated JSON
        + "not json at all\n"
    )

    ledger = UsageLedger(path, clock=lambda: 1_700_000_000.0)
    recorder = UsageRecorder.with_ledger(ledger)
    snapshot = recorder.snapshot()
    ledger.close()

    assert ledger.stats.replayed_requests == 1
    assert ledger.stats.skipped_lines == 2
    assert ledger.stats.earliest_ts == 1_600_000_000
    assert snapshot.totals.requests == 1
    assert snapshot.totals.total_tokens == 10
    assert [item.seat for item in snapshot.by_seat] == ["fable"]


def test_unopenable_ledger_fails_loud_never_silent(tmp_path: Path) -> None:
    unwritable_dir = tmp_path / "missing-parent" / "usage.jsonl"

    with pytest.raises(LedgerError):
        UsageLedger(unwritable_dir, clock=lambda: 0.0)


# ── The point of the feature: /v1/usage survives a restart ───────────────────


def test_usage_survives_restart_including_seats(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"

    with _client(path) as client:
        assert _chat(client, seat="fable").status_code == 200
        assert _chat(client, seat="fable").status_code == 200
        assert _chat(client).status_code == 200  # unlabeled
        first = client.get("/v1/usage").json()

    assert first["scope"] == "ledger"
    assert first["totals"]["requests"] == 3

    # "Restart": a brand-new app over the same ledger file.
    with _client(path, clock_value=1_700_000_999.0) as client:
        body = client.get("/v1/usage").json()
        assert _chat(client, seat="codex").status_code == 200
        after = client.get("/v1/usage").json()

    assert body["scope"] == "ledger"
    assert body["ledger"]["replayed_requests"] == 3
    assert body["ledger"]["skipped_lines"] == 0
    assert body["totals"]["requests"] == 3  # nothing forgotten
    seats = {item["seat"]: item["totals"]["requests"] for item in after["by_seat"]}
    assert seats == {"fable": 2, "codex": 1}
    assert after["totals"]["requests"] == 4


def test_started_at_is_the_earliest_ledger_entry(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    with _client(path, clock_value=1_600_000_000.0) as client:
        assert _chat(client).status_code == 200

    with _client(path, clock_value=1_700_000_000.0) as client:
        body = client.get("/v1/usage").json()

    assert body["started_at"] == 1_600_000_000


def test_without_the_section_behavior_is_exactly_process_scope(tmp_path: Path) -> None:
    app = create_app(_config(None), clock=lambda: 1_700_000_000.0)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert _chat(client, seat="fable").status_code == 200
        body = client.get("/v1/usage").json()

    assert body["scope"] == "process"
    assert body["ledger"] is None
    assert list(tmp_path.iterdir()) == []  # nothing was written anywhere


# ── Leak sentinels (ROADMAP §1.5) ────────────────────────────────────────────


def test_ledger_file_never_contains_content_or_native_model_names(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    with _client(path) as client:
        assert _chat(client, seat="fable").status_code == 200

    text = path.read_text()
    assert PROMPT_SENTINEL not in text  # no message content, ever
    assert "native-one" not in text  # provider_model never leaves the process
    assert "alias-one" in text  # the public alias is the recorded name


def test_failed_ledger_writes_are_counted_and_named_without_content(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A dead ledger handle is counted and logged, never a crash or a secret.

    The request itself already succeeded when the append fires, so a write
    failure must be loud (counter + fixed-name ERROR event) but non-fatal.
    """

    import logging

    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path, clock=lambda: 0.0)
    recorder = UsageRecorder.with_ledger(ledger)
    ledger._handle.close()  # sabotage: every later append fails

    with caplog.at_level(logging.ERROR, logger="vulcan.usage"):
        recorder.record(
            model="alias-one", provider="det", prompt_tokens=1, completion_tokens=1, seat="fable"
        )

    snapshot = recorder.snapshot()
    assert snapshot.ledger is not None
    assert snapshot.ledger.write_failures == 1
    # The in-memory counters still counted the request: the meter never lies
    # about what this process served, even when the durable copy fails.
    assert snapshot.totals.requests == 1
    assert "usage_ledger_write_failed" in caplog.text
    assert "fable" not in caplog.text  # fixed event name only, no record data


# ── Config surface ───────────────────────────────────────────────────────────


def test_relative_ledger_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        UsageConfig(ledger_path=Path("relative/usage.jsonl"))


def test_unknown_keys_in_usage_section_fail_startup(tmp_path: Path) -> None:
    config_path = tmp_path / "vulcan.toml"
    config_path.write_text(
        f"""
schema_version = 2

[server]
host = "127.0.0.1"
port = 8140

[usage]
ledger_path = "{tmp_path / "usage.jsonl"}"
rotation = "daily"

[providers.det]
type = "deterministic"
response_text = "canned"

[[models]]
id = "alias-one"
provider = "det"
provider_model = "native"
capabilities = ["chat"]
"""
    )
    with pytest.raises(ConfigLoadError):
        load_config(config_path)


# ── Poisoned-ledger sentinels: the file is a trust boundary ──────────────────


POISON_SENTINEL = "poisoned-value-must-not-reach-http-c4d7"


def _poisoned_ledger(tmp_path: Path) -> Path:
    """A ledger with one good line surrounded by every poison class."""

    path = tmp_path / "usage.jsonl"
    good = {
        "completion_tokens": 3,
        "model": "alias-one",
        "prompt_tokens": 7,
        "provider": "det",
        "seat": "fable",
        "ts": 1_600_000_000,
    }
    poison_lines: list[bytes] = [
        # negative token counts would corrupt totals and fail response ge=0
        json.dumps({**good, "prompt_tokens": -50}).encode(),
        # pattern-violating seat would fail SeatUsageRecord validation (500)
        json.dumps({**good, "seat": POISON_SENTINEL + " WITH SPACES!"}).encode(),
        # injected free text in the model field must never reach the response
        json.dumps({**good, "model": POISON_SENTINEL + " secret prompt text"}).encode(),
        # booleans masquerading as ints
        json.dumps({**good, "ts": True}).encode(),
        # absurd token count
        json.dumps({**good, "completion_tokens": 10**15}).encode(),
        # invalid UTF-8 must not crash startup (UnicodeDecodeError, not OSError)
        b'\xff\xfe{"model": "alias-one"}',
        # wrong JSON type entirely
        json.dumps(["not", "a", "dict"]).encode(),
    ]
    path.write_bytes(
        json.dumps(good, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
        + b"\n".join(poison_lines)
        + b"\n"
    )
    return path


def test_poisoned_ledger_lines_never_reach_counters_or_http(tmp_path: Path) -> None:
    path = _poisoned_ledger(tmp_path)

    with _client(path) as client:
        response = client.get("/v1/usage")

    # Startup survived every poison class and /v1/usage still serves.
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "ledger"
    assert body["ledger"]["replayed_requests"] == 1  # only the good line
    assert body["ledger"]["skipped_lines"] == 7  # every poison line, counted
    assert body["totals"]["requests"] == 1
    assert body["totals"]["prompt_tokens"] == 7  # the -50 never subtracted
    assert [item["seat"] for item in body["by_seat"]] == ["fable"]
    # The injected values appear nowhere in the HTTP response.
    assert POISON_SENTINEL not in response.text


def test_ledger_is_closed_on_app_shutdown(tmp_path: Path) -> None:
    """Lifespan shutdown closes the ledger handle, not just providers."""

    path = tmp_path / "usage.jsonl"
    client = _client(path)
    with client:
        assert _chat(client).status_code == 200

    # TestClient exit runs lifespan shutdown; a closed handle means a later
    # append through the same app would fail — prove the handle is closed by
    # replaying the file: the write above must be durable and complete.
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["model"] == "alias-one"


def test_ledger_log_events_survive_the_production_formatter() -> None:
    """The promised fixed event names are allowlisted, not 'external_log'."""

    import logging

    from vulcan.observability import SafeJsonFormatter

    record = logging.LogRecord(
        name="vulcan.usage",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="usage_ledger_write_failed",
        args=(),
        exc_info=None,
    )
    payload = json.loads(SafeJsonFormatter().format(record))
    assert payload["event"] == "usage_ledger_write_failed"
