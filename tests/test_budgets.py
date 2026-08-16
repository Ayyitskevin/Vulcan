"""Per-seat daily hosted budgets: gate, windowing, restart survival, honesty.

Design contract (operator-approved 2026-08-16): hosted-only, tokens + request
caps per UTC day, fail-closed when the section exists, refuse-loudly with the
reset time — the caller owns any fallback, Vulcan NEVER reroutes (one alias,
one provider). Local providers are never budgeted. Seats are voluntary labels:
this guards against accidents, not adversaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.budgets import BudgetBook, SeatLimits
from vulcan.config import (
    BudgetsConfig,
    Capability,
    ConfigLoadError,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
    OpenAICompatibleProviderConfig,
    SeatBudgetConfig,
    UsageConfig,
    load_config,
)
from vulcan.errors import BudgetExhaustedError, BudgetUnconfiguredError, SeatRequiredError
from vulcan.providers.base import Provider
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider

OPENAI_KEY_ENV = "VULCAN_BUDGET_TEST_OPENAI_KEY"

DAY = 86_400
NOON = 1_700_000_000 - (1_700_000_000 % DAY) + 43_200  # noon UTC of that day


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, "sk-budget-test-secret-1a2b")


# ── BudgetBook unit behavior ─────────────────────────────────────────────────


def _book(
    *,
    limits: dict[str, SeatLimits] | None = None,
    default: SeatLimits | None = None,
    now: list[float] | None = None,
) -> BudgetBook:
    state = now if now is not None else [float(NOON)]
    return BudgetBook(
        limits=limits or {},
        default=default,
        hosted_providers=frozenset({"hosted"}),
        clock=lambda: state[0],
    )


def test_local_providers_are_never_budgeted() -> None:
    book = _book()  # no limits at all
    book.check(seat=None, provider_id="local")  # no seat, no entry: fine locally
    book.check(seat="anyone", provider_id="local")


def test_hosted_without_seat_is_refused() -> None:
    book = _book(default=SeatLimits(1000, None))
    with pytest.raises(SeatRequiredError):
        book.check(seat=None, provider_id="hosted")


def test_hosted_seat_without_entry_fails_closed() -> None:
    book = _book(limits={"fable": SeatLimits(1000, None)})  # no default
    book.check(seat="fable", provider_id="hosted")
    with pytest.raises(BudgetUnconfiguredError):
        book.check(seat="stranger", provider_id="hosted")


def test_token_budget_exhausts_and_names_the_reset_time() -> None:
    book = _book(limits={"fable": SeatLimits(100, None)})
    book.check(seat="fable", provider_id="hosted")
    book.spend(seat="fable", provider_id="hosted", tokens=100)

    with pytest.raises(BudgetExhaustedError) as excinfo:
        book.check(seat="fable", provider_id="hosted")

    details = excinfo.value.details
    assert details is not None
    assert details["seat"] == "fable"
    assert details["window_resets_at"] == ((NOON // DAY) + 1) * DAY


def test_request_cap_catches_providers_that_omit_token_counts() -> None:
    book = _book(default=SeatLimits(None, 2))
    for _ in range(2):
        book.check(seat="fable", provider_id="hosted")
        book.spend(seat="fable", provider_id="hosted", tokens=None)  # no usage reported

    with pytest.raises(BudgetExhaustedError):
        book.check(seat="fable", provider_id="hosted")


def test_window_rolls_at_utc_midnight() -> None:
    now = [float(NOON)]
    book = _book(limits={"fable": SeatLimits(100, None)}, now=now)
    book.spend(seat="fable", provider_id="hosted", tokens=100)
    with pytest.raises(BudgetExhaustedError):
        book.check(seat="fable", provider_id="hosted")

    now[0] = float(((NOON // DAY) + 1) * DAY + 60)  # one minute past midnight
    book.check(seat="fable", provider_id="hosted")  # fresh day, fresh budget
    assert book.snapshot()[0].tokens_today == 0


def test_replayed_spend_from_a_previous_day_never_counts() -> None:
    book = _book(limits={"fable": SeatLimits(100, None)})
    book.spend(seat="fable", provider_id="hosted", tokens=90, ts=float(NOON - DAY))
    book.check(seat="fable", provider_id="hosted")  # yesterday is not today
    assert book.snapshot()[0].tokens_today == 0


def test_overshoot_is_inherent_and_documented() -> None:
    """Pre-flight passes with 1 token of headroom; the response may cost more."""

    book = _book(limits={"fable": SeatLimits(100, None)})
    book.spend(seat="fable", provider_id="hosted", tokens=99)
    book.check(seat="fable", provider_id="hosted")  # allowed at 99/100
    book.spend(seat="fable", provider_id="hosted", tokens=500)  # single-request overshoot
    assert book.snapshot()[0].tokens_today == 599
    with pytest.raises(BudgetExhaustedError):
        book.check(seat="fable", provider_id="hosted")


# ── HTTP contract ────────────────────────────────────────────────────────────


def _config(
    budgets: BudgetsConfig | None,
    ledger_path: Path | None = None,
) -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        budgets=budgets,
        usage=UsageConfig(ledger_path=ledger_path) if ledger_path is not None else None,
        providers={
            "det": DeterministicProviderConfig(type="deterministic", response_text="canned"),
            "openai": OpenAICompatibleProviderConfig(
                type="openai_compatible",
                base_url="https://api.example.test/v1",
                api_key_env=OPENAI_KEY_ENV,
                timeout_seconds=1.0,
            ),
        },
        models=(
            ModelConfig(
                id="local-chat",
                provider="det",
                provider_model="native-local",
                capabilities=frozenset({Capability.CHAT}),
            ),
            ModelConfig(
                id="hosted-chat",
                provider="openai",
                provider_model="native-hosted",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )


def _providers(config: GatewayConfig, usage: tuple[int, int] | None = (60, 40)) -> dict[str, Any]:
    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = {
            "choices": [
                {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": None,
        }
        if usage is not None:
            body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
        return httpx.Response(200, json=body)

    out: dict[str, Provider] = {}
    for provider_id, provider_config in config.providers.items():
        if provider_config.type == "deterministic":
            out[provider_id] = DeterministicProvider(provider_id, provider_config)
        else:
            out[provider_id] = OpenAICompatibleProvider(
                provider_id,
                provider_config,
                client=httpx.AsyncClient(
                    base_url=provider_config.base_url,
                    transport=httpx.MockTransport(handler),
                    trust_env=False,
                ),
            )
    return out


def _budgets(**seats: SeatBudgetConfig) -> BudgetsConfig:
    return BudgetsConfig(seats=seats)


def _client(
    config: GatewayConfig,
    clock_value: float = float(NOON),
    usage: tuple[int, int] | None = (60, 40),
) -> TestClient:
    app = create_app(config, providers=_providers(config, usage), clock=lambda: clock_value)
    return TestClient(app, base_url="http://127.0.0.1")


def _chat(client: TestClient, model: str, seat: str | None = None) -> httpx.Response:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
    }
    if seat is not None:
        payload["seat"] = seat
    return client.post("/v1/chat/completions", json=payload)


def test_hosted_request_within_budget_succeeds_and_is_metered() -> None:
    config = _config(_budgets(fable=SeatBudgetConfig(hosted_tokens_per_day=1000)))
    with _client(config) as client:
        assert _chat(client, "hosted-chat", seat="fable").status_code == 200
        body = client.get("/v1/usage").json()

    rows = {row["seat"]: row for row in body["budgets"]}
    assert rows["fable"]["tokens_today"] == 100
    assert rows["fable"]["requests_today"] == 1
    assert rows["fable"]["hosted_tokens_per_day"] == 1000


def test_exhausted_seat_gets_429_with_reset_time_and_no_upstream_call() -> None:
    config = _config(_budgets(fable=SeatBudgetConfig(hosted_tokens_per_day=150)))
    with _client(config) as client:
        assert _chat(client, "hosted-chat", seat="fable").status_code == 200  # 100 tokens
        assert _chat(client, "hosted-chat", seat="fable").status_code == 200  # 200 > 150
        refused = _chat(client, "hosted-chat", seat="fable")
        body = client.get("/v1/usage").json()

    assert refused.status_code == 429
    error = refused.json()["error"]
    assert error["code"] == "budget_exhausted"
    assert error["retryable"] is True
    assert error["details"]["seat"] == "fable"
    assert error["details"]["window_resets_at"] == ((NOON // DAY) + 1) * DAY
    # The refused request never reached the upstream: only 2 requests metered.
    assert body["totals"]["requests"] == 2


def test_unlabeled_hosted_request_is_refused_when_budgets_exist() -> None:
    config = _config(_budgets(default=SeatBudgetConfig(hosted_tokens_per_day=1000)))
    with _client(config) as client:
        response = _chat(client, "hosted-chat")

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "seat_required"


def test_local_requests_stay_unbudgeted_and_label_optional() -> None:
    config = _config(_budgets(fable=SeatBudgetConfig(hosted_tokens_per_day=0)))
    with _client(config) as client:
        assert _chat(client, "local-chat").status_code == 200  # no label needed
        assert _chat(client, "local-chat", seat="fable").status_code == 200  # 0-budget seat


def test_unbudgeted_seat_fails_closed_with_403() -> None:
    config = _config(_budgets(fable=SeatBudgetConfig(hosted_tokens_per_day=1000)))
    with _client(config) as client:
        response = _chat(client, "hosted-chat", seat="stranger")

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "budget_unconfigured"


def test_default_entry_covers_unnamed_seats() -> None:
    config = _config(_budgets(default=SeatBudgetConfig(hosted_requests_per_day=1)))
    with _client(config) as client:
        assert _chat(client, "hosted-chat", seat="anyone").status_code == 200
        assert _chat(client, "hosted-chat", seat="anyone").status_code == 429


def test_without_the_section_hosted_requests_behave_as_before() -> None:
    config = _config(None)
    with _client(config) as client:
        assert _chat(client, "hosted-chat").status_code == 200  # unlabeled, unbudgeted
        body = client.get("/v1/usage").json()

    assert body["budgets"] is None


def test_budget_spend_survives_restart_via_the_ledger(tmp_path: Path) -> None:
    """The whole reason gate 4 came first: allowances are restart-proof."""

    path = tmp_path / "usage.jsonl"
    budgets = _budgets(fable=SeatBudgetConfig(hosted_tokens_per_day=150))

    with _client(_config(budgets, ledger_path=path)) as client:
        assert _chat(client, "hosted-chat", seat="fable").status_code == 200  # 100 tokens

    # Restart: same day, same ledger — the spend must still count.
    with _client(_config(budgets, ledger_path=path), clock_value=float(NOON + 600)) as client:
        body = client.get("/v1/usage").json()
        second = _chat(client, "hosted-chat", seat="fable")  # 100 more → over 150
        third = _chat(client, "hosted-chat", seat="fable")

    rows = {row["seat"]: row for row in body["budgets"]}
    assert rows["fable"]["tokens_today"] == 100  # replayed, not forgotten
    assert second.status_code == 200  # headroom existed at pre-flight (overshoot rule)
    assert third.status_code == 429  # now genuinely exhausted


def test_config_rejects_empty_entry_and_bad_seat_names() -> None:
    with pytest.raises(ValueError, match="at least one limit"):
        SeatBudgetConfig()
    with pytest.raises(ValueError, match="invalid budget seat name"):
        BudgetsConfig(seats={"NOT VALID": SeatBudgetConfig(hosted_tokens_per_day=1)})


def test_unknown_keys_in_budgets_section_fail_startup(tmp_path: Path) -> None:
    config_path = tmp_path / "vulcan.toml"
    config_path.write_text(
        """
schema_version = 2

[providers.det]
type = "deterministic"
response_text = "canned"

[[models]]
id = "alias-one"
provider = "det"
provider_model = "native"
capabilities = ["chat"]

[budgets.seats.fable]
hosted_tokens_per_day = 1000
grace_period = "1h"
"""
    )
    with pytest.raises(ConfigLoadError):
        load_config(config_path)


def test_budget_errors_leak_nothing_but_safe_fields() -> None:
    config = _config(_budgets(fable=SeatBudgetConfig(hosted_tokens_per_day=0)))
    with _client(config) as client:
        response = _chat(client, "hosted-chat", seat="fable")

    payload = response.json()
    assert response.status_code == 429
    # Exactly the documented safe fields: ours two, plus the provider ID the
    # HTTP boundary adds to every provider-scoped failure (configured ID only).
    assert set(payload["error"]["details"]) == {"seat", "window_resets_at", "provider"}
    assert payload["error"]["details"]["provider"] == "openai"
    assert "native-hosted" not in json.dumps(payload)  # no native model names
