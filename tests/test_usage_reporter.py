"""Tests for ``scripts/usage_reporter.py`` — the Athena usage-digest reporter.

The reporter holds the deployment's only persisted shared secret and posts
off-loopback, so these tests pin its honesty rules (baseline/reset/delta), its
fail-loud exit codes, and above all that the secret and any prompt-shaped
payload content never reach the wire, the state file, or the operator's
terminal. Nothing here contacts a real API: the urllib opener is faked, the
same guarantee ``httpx.MockTransport`` provides for the gateway suite.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import io
import json
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any

import pytest

# scripts/ is not a package; load the reporter by path so the suite imports it
# exactly as the systemd unit executes it. The sys.modules registration is
# required: @dataclass resolves the module's namespace at decoration time.
_REPORTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "usage_reporter.py"
_spec = importlib.util.spec_from_file_location("usage_reporter", _REPORTER_PATH)
assert _spec is not None and _spec.loader is not None
usage_reporter = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = usage_reporter
_spec.loader.exec_module(usage_reporter)

SECRET_SENTINEL = "forge-secret-9c1f7e02"
PROMPT_SENTINEL = "private-prompt-4a8d31c7"

ENV_NAMES = (
    "ATHENA_FORGE_SOURCE",
    "ATHENA_FORGE_SECRET",
    "ATHENA_ISSUE_KEY",
    "VULCAN_USAGE_URL",
    "ATHENA_BASE_URL",
    "USAGE_REPORTER_STATE",
)

NOW = 1_800_000_000.0


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeOpener:
    """Stands in for ``urllib.request.OpenerDirector``; serves queued outcomes."""

    def __init__(self, outcomes: list[bytes | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        del timeout
        self.requests.append(request)
        assert self._outcomes, "reporter made an unexpected extra HTTP call"
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeResponse(outcome)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _headers(request: Any) -> dict[str, str]:
    return {key.lower(): value for key, value in request.headers.items()}


def _set_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    values = {
        "ATHENA_FORGE_SOURCE": "vulcan-usage",
        "ATHENA_FORGE_SECRET": SECRET_SENTINEL,
        "ATHENA_ISSUE_KEY": "vul-1",
    }
    values.update(overrides)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _counter_row(requests: int) -> dict[str, int]:
    return {
        "requests": requests,
        "requests_with_usage": requests,
        "prompt_tokens": requests * 10,
        "completion_tokens": requests * 5,
        "total_tokens": requests * 15,
    }


def _usage_payload(
    *,
    requests: int = 12,
    scope: str = "ledger",
    seats: dict[str, int] | None = None,
    models: dict[str, int] | None = None,
    ledger: dict[str, Any] | None = None,
    budgets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if ledger is None:
        ledger = {"replayed_requests": requests, "skipped_lines": 0, "write_failures": 0}
    if budgets is None:
        budgets = [
            {
                "seat": "kimi",
                "requests_today": 3,
                "hosted_requests_per_day": 500,
                "tokens_today": 900,
                "hosted_tokens_per_day": 100000,
                "window_resets_at": int(NOW),
            }
        ]
    return {
        "scope": scope,
        "totals": _counter_row(requests),
        "by_seat": [
            {"seat": seat, "totals": _counter_row(reqs)} for seat, reqs in (seats or {}).items()
        ],
        "by_model": [
            {"model": model, "totals": _counter_row(reqs)} for model, reqs in (models or {}).items()
        ],
        "ledger": ledger,
        "budgets": budgets,
    }


def _config(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Any:
    _set_env(monkeypatch, **overrides)
    return usage_reporter._load_config()


# --- _load_config -----------------------------------------------------------


def test_load_config_happy_path_normalizes_and_composes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(
        monkeypatch,
        ATHENA_BASE_URL="http://athena.example:8300/",
        VULCAN_USAGE_URL="http://127.0.0.1:9999/v1/usage",
    )

    assert config.vulcan_usage_url == "http://127.0.0.1:9999/v1/usage"
    assert config.athena_url == "http://athena.example:8300/forge/vulcan-usage"
    assert config.secret == SECRET_SENTINEL
    assert config.issue_key == "VUL-1"  # uppercased before the shape check
    assert config.state_path == Path(usage_reporter.DEFAULT_STATE_PATH)


def test_load_config_names_every_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(usage_reporter.ConfigError) as excinfo:
        usage_reporter._load_config()

    message = str(excinfo.value)
    for name in ("ATHENA_FORGE_SOURCE", "ATHENA_FORGE_SECRET", "ATHENA_ISSUE_KEY"):
        assert name in message


def test_load_config_error_never_echoes_secret_value(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, ATHENA_ISSUE_KEY="not-an-issue-key")

    with pytest.raises(usage_reporter.ConfigError) as excinfo:
        usage_reporter._load_config()

    assert "NOT-AN-ISSUE-KEY" in str(excinfo.value)  # operator config, safe to echo
    assert SECRET_SENTINEL not in str(excinfo.value)


def test_load_config_rejects_non_http_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, VULCAN_USAGE_URL="ftp://127.0.0.1/v1/usage")
    with pytest.raises(usage_reporter.ConfigError):
        usage_reporter._load_config()

    _set_env(monkeypatch, ATHENA_BASE_URL="file:///etc/passwd")
    with pytest.raises(usage_reporter.ConfigError):
        usage_reporter._load_config()


# --- _fetch_usage -----------------------------------------------------------


def test_fetch_usage_returns_payload() -> None:
    payload = _usage_payload()
    opener = _FakeOpener([_json_bytes(payload)])

    assert usage_reporter._fetch_usage(opener, "http://127.0.0.1:8140/v1/usage") == payload
    assert opener.requests[0].get_method() == "GET"


def test_fetch_usage_http_error_is_loud() -> None:
    opener = _FakeOpener([urllib.error.HTTPError("http://x", 500, "err", {}, io.BytesIO(b"nope"))])

    with pytest.raises(usage_reporter.ReporterError, match="HTTP 500"):
        usage_reporter._fetch_usage(opener, "http://127.0.0.1:8140/v1/usage")


def test_fetch_usage_unreachable_is_loud() -> None:
    opener = _FakeOpener([urllib.error.URLError("dns failed")])

    with pytest.raises(usage_reporter.ReporterError, match="unreachable"):
        usage_reporter._fetch_usage(opener, "http://127.0.0.1:8140/v1/usage")


def test_fetch_usage_rejects_invalid_json_and_unexpected_shape() -> None:
    with pytest.raises(usage_reporter.ReporterError, match="valid JSON"):
        usage_reporter._fetch_usage(_FakeOpener([b"not json"]), "http://x")

    with pytest.raises(usage_reporter.ReporterError, match="unexpected shape"):
        usage_reporter._fetch_usage(_FakeOpener([b'{"nope": 1}']), "http://x")


# --- snapshot load/write ------------------------------------------------------


def test_load_snapshot_missing_file_means_baseline(tmp_path: Path) -> None:
    assert usage_reporter._load_snapshot(tmp_path / "absent.json") is None


def test_load_snapshot_corrupt_or_misshapen_is_loud(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ not json", encoding="utf-8")
    with pytest.raises(usage_reporter.ReporterError, match="re-baseline"):
        usage_reporter._load_snapshot(corrupt)

    misshapen = tmp_path / "misshapen.json"
    misshapen.write_text('{"totals": "not-a-dict"}', encoding="utf-8")
    with pytest.raises(usage_reporter.ReporterError, match="unexpected shape"):
        usage_reporter._load_snapshot(misshapen)


def test_write_snapshot_round_trips_through_load(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    snapshot = {"fetched_at": NOW, "scope": "ledger", "totals": _counter_row(7)}

    usage_reporter._write_snapshot(path, snapshot)

    assert usage_reporter._load_snapshot(path) == snapshot
    assert not (tmp_path / "nested" / "state.json.tmp").exists()


def test_write_snapshot_failure_warns_about_re_reporting(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory", encoding="utf-8")

    with pytest.raises(usage_reporter.ReporterError, match="digest posted but snapshot"):
        usage_reporter._write_snapshot(blocker / "state.json", {"totals": {}})


# --- _build_digest ------------------------------------------------------------


def test_digest_baseline_reports_cumulative_counters() -> None:
    payload = _usage_payload(seats={"kimi": 5, "fable": 2}, models={"simple": 7})

    messages, snapshot, kind = usage_reporter._build_digest(payload, None, NOW, "VUL-1")

    date_label = time.strftime("%Y-%m-%d", time.localtime(NOW))
    assert kind == "baseline"
    assert messages[0] == (
        f"VUL-1 vulcan {date_label} [baseline]: 12 req (12 with usage) 120in/60out tok"
    )
    # 12 total - 7 seated = 5 unlabeled requests declared, not dropped.
    assert any("unlabeled 5r/75t" in message for message in messages)
    assert any(message.startswith("VUL-1 seats [baseline]: ") for message in messages)
    assert any(message.startswith("VUL-1 aliases [baseline]: ") for message in messages)
    assert any("ledger: replayed 12 skipped 0 write_failures 0" in m for m in messages)
    assert any("budgets today:" in message for message in messages)
    assert all(len(message) <= usage_reporter.MAX_MESSAGE_CHARS for message in messages)
    assert snapshot["scope"] == "ledger"
    assert snapshot["fetched_at"] == NOW
    assert snapshot["totals"] == _counter_row(12)
    assert snapshot["by_seat"]["kimi"] == _counter_row(5)


def test_digest_delta_reports_window_diffs_and_drops_quiet_names() -> None:
    _, previous, kind = usage_reporter._build_digest(
        _usage_payload(requests=12, seats={"kimi": 5, "fable": 2}), None, NOW - 24 * 3600, "VUL-1"
    )
    assert kind == "baseline"
    current = _usage_payload(requests=20, seats={"kimi": 9, "fable": 2}, models={"simple": 9})

    messages, _, kind = usage_reporter._build_digest(current, previous, NOW, "VUL-1")

    assert kind == "delta"
    assert "[delta 24h]" in messages[0]
    assert "8 req (8 with usage) 80in/40out tok" in messages[0]
    seats_line = next(m for m in messages if m.startswith("VUL-1 seats: "))
    assert "kimi 4r/60t" in seats_line
    assert "fable" not in seats_line  # quiet all window is not news
    assert "unlabeled 4r/60t" in seats_line
    assert all("[baseline]" not in message and "[reset]" not in message for message in messages)


def test_digest_reset_on_counter_regression_reports_cumulative() -> None:
    _, previous, _ = usage_reporter._build_digest(
        _usage_payload(requests=50), None, NOW - 3600, "VUL-1"
    )

    messages, _, kind = usage_reporter._build_digest(
        _usage_payload(requests=12), previous, NOW, "VUL-1"
    )

    assert kind == "reset"
    assert "[reset]: 12 req" in messages[0]  # cumulative, never a negative delta


def test_digest_reset_on_scope_change() -> None:
    _, previous, _ = usage_reporter._build_digest(
        _usage_payload(scope="process"), None, NOW - 3600, "VUL-1"
    )

    _, _, kind = usage_reporter._build_digest(
        _usage_payload(scope="ledger"), previous, NOW, "VUL-1"
    )

    assert kind == "reset"


def test_digest_ledger_off_and_budget_edge_labels() -> None:
    payload = _usage_payload(
        scope="process",
        ledger=None,
        budgets=[
            {
                "seat": "kimi",
                "requests_today": 1,
                "hosted_requests_per_day": 500,
                "tokens_today": 10,
                "hosted_tokens_per_day": None,
                "window_resets_at": "not-a-number",
            }
        ],
    )
    payload["ledger"] = None

    messages, _, _ = usage_reporter._build_digest(payload, None, NOW, "VUL-1")

    ledger_line = messages[-1]
    assert "ledger: off (process scope)" in ledger_line
    assert "kimi 1/500r 10/-t resets unknown" in ledger_line


def test_join_fitted_declares_overflow_and_bounds_pathological_prefix() -> None:
    items = [f"seat-number-{index:02d} 9r/999t" for index in range(30)]

    line = usage_reporter._join_fitted("VUL-1 seats: ", items)

    assert len(line) <= usage_reporter.MAX_MESSAGE_CHARS
    assert "more" in line
    assert "+" in line
    assert "seat-number-00" in line
    assert "seat-number-29" not in line

    pathological = usage_reporter._join_fitted("x" * 250, items)
    assert pathological == f"{'x' * 250}+30 more"


# --- _post_digest -------------------------------------------------------------


def test_post_digest_signs_exact_body_and_posts_github_push_dialect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)
    opener = _FakeOpener([_json_bytes({"landed": 2})])

    result = usage_reporter._post_digest(config, opener, ["row one", "row two"])

    assert result == {"landed": 2}
    request = opener.requests[0]
    assert request.full_url == "http://100.125.80.91:8300/forge/vulcan-usage"
    assert request.get_method() == "POST"
    expected_body = json.dumps(
        {
            "ref": usage_reporter.DIGEST_REF,
            "commits": [{"message": "row one"}, {"message": "row two"}],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    assert request.data == expected_body
    headers = _headers(request)
    assert headers["content-type"] == "application/json"
    assert headers["x-github-event"] == "push"
    expected_signature = hmac.new(
        SECRET_SENTINEL.encode("utf-8"), expected_body, hashlib.sha256
    ).hexdigest()
    assert headers["x-hub-signature-256"] == f"sha256={expected_signature}"
    # The raw secret appears nowhere on the wire — only its HMAC.
    assert SECRET_SENTINEL not in expected_body.decode("utf-8")
    assert all(SECRET_SENTINEL not in value for value in headers.values())


def test_post_digest_refused_delivery_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(monkeypatch)
    opener = _FakeOpener(
        [urllib.error.HTTPError("http://x", 401, "nope", {}, io.BytesIO(b"bad signature"))]
    )

    with pytest.raises(usage_reporter.ReporterError, match="HTTP 401 bad signature"):
        usage_reporter._post_digest(config, opener, ["row"])


def test_post_digest_unreachable_and_non_json_are_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)

    with pytest.raises(usage_reporter.ReporterError, match="unreachable"):
        usage_reporter._post_digest(config, _FakeOpener([urllib.error.URLError("down")]), ["r"])
    with pytest.raises(usage_reporter.ReporterError, match="non-JSON"):
        usage_reporter._post_digest(config, _FakeOpener([b"<html>"]), ["r"])


def test_post_digest_rejects_delivery_that_landed_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)
    opener = _FakeOpener([_json_bytes({"landed": 1})])

    with pytest.raises(usage_reporter.ReporterError, match="landed 1 of 2 rows"):
        usage_reporter._post_digest(config, opener, ["row one", "row two"])


def test_redirect_handler_refuses_redirects() -> None:
    assert usage_reporter._NoRedirect().redirect_request(None, None, None, None, None, None) is None


# --- main() end to end ----------------------------------------------------------


def test_main_success_posts_baseline_and_writes_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    _set_env(monkeypatch, USAGE_REPORTER_STATE=str(state_path))
    payload = _usage_payload(seats={"kimi": 5}, models={"simple": 7})
    # header + seats + aliases + ledger/budgets lines
    opener = _FakeOpener([_json_bytes(payload), _json_bytes({"landed": 4})])
    monkeypatch.setattr(usage_reporter, "_opener", lambda: opener)

    assert usage_reporter.main() == 0

    out, err = capsys.readouterr()
    assert "baseline digest posted — 4 rows on VUL-1" in out
    assert err == ""
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert set(state) == {"fetched_at", "scope", "totals", "by_seat", "by_model"}
    assert state["totals"] == _counter_row(12)


def test_main_second_run_reports_delta(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    _set_env(monkeypatch, USAGE_REPORTER_STATE=str(state_path))
    _, previous, _ = usage_reporter._build_digest(
        _usage_payload(requests=12), None, NOW - 24 * 3600, "VUL-1"
    )
    usage_reporter._write_snapshot(state_path, previous)
    opener = _FakeOpener([_json_bytes(_usage_payload(requests=15)), _json_bytes({"landed": 3})])
    monkeypatch.setattr(usage_reporter, "_opener", lambda: opener)

    assert usage_reporter.main() == 0

    out, _ = capsys.readouterr()
    assert "delta digest posted" in out


def test_main_failures_exit_nonzero_without_writing_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    _set_env(monkeypatch, USAGE_REPORTER_STATE=str(state_path))

    # Unreachable vulcan: exit 1, loud, no state.
    opener = _FakeOpener([urllib.error.URLError("down")])
    monkeypatch.setattr(usage_reporter, "_opener", lambda: opener)
    assert usage_reporter.main() == 1
    assert "usage-reporter FAILED" in capsys.readouterr().err
    assert not state_path.exists()

    # Delivery lands nowhere: exit 1, no state (the window re-reports next run).
    opener = _FakeOpener([_json_bytes(_usage_payload()), _json_bytes({"landed": 0})])
    monkeypatch.setattr(usage_reporter, "_opener", lambda: opener)
    assert usage_reporter.main() == 1
    assert not state_path.exists()


def test_main_config_error_exits_2_before_any_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    def _forbidden_opener() -> Any:
        raise AssertionError("opener constructed despite a config error")

    monkeypatch.setattr(usage_reporter, "_opener", _forbidden_opener)

    assert usage_reporter.main() == 2
    assert "configuration error" in capsys.readouterr().err


# --- sentinel leak pins ---------------------------------------------------------


def test_secret_never_reaches_wire_state_or_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_path = tmp_path / "state.json"
    _set_env(monkeypatch, USAGE_REPORTER_STATE=str(state_path))
    opener = _FakeOpener([_json_bytes(_usage_payload()), _json_bytes({"landed": 3})])
    monkeypatch.setattr(usage_reporter, "_opener", lambda: opener)

    assert usage_reporter.main() == 0

    out, err = capsys.readouterr()
    request = opener.requests[1]
    leak_surfaces = [
        request.data.decode("utf-8"),
        *request.headers.values(),
        out,
        err,
        state_path.read_text(encoding="utf-8"),
    ]
    for surface in leak_surfaces:
        assert SECRET_SENTINEL not in surface


def test_prompt_shaped_payload_content_never_reaches_digest_or_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only seat/alias names and counters propagate; every other field is dropped."""
    state_path = tmp_path / "state.json"
    _set_env(monkeypatch, USAGE_REPORTER_STATE=str(state_path))
    payload = _usage_payload(seats={"kimi": 12}, models={"simple": 12})
    payload["echo"] = PROMPT_SENTINEL
    payload["totals"]["debug_note"] = PROMPT_SENTINEL
    payload["by_seat"][0]["sample_prompt"] = PROMPT_SENTINEL
    payload["by_model"][0]["sample_response"] = PROMPT_SENTINEL
    payload["ledger"]["last_line"] = PROMPT_SENTINEL
    payload["budgets"][0]["operator_note"] = PROMPT_SENTINEL
    opener = _FakeOpener([_json_bytes(payload), _json_bytes({"landed": 4})])
    monkeypatch.setattr(usage_reporter, "_opener", lambda: opener)

    assert usage_reporter.main() == 0

    posted_body = opener.requests[1].data.decode("utf-8")
    assert "kimi" in posted_body  # names propagate by design
    assert PROMPT_SENTINEL not in posted_body
    assert PROMPT_SENTINEL not in state_path.read_text(encoding="utf-8")
