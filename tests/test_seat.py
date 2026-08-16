"""Optional seat attribution: recorder, HTTP contract, and upstream sentinels.

A ``seat`` is an operator-chosen caller label on chat and embedding requests.
It exists only for ``/v1/usage`` attribution: labeled requests aggregate under
``by_seat``, unlabeled requests still count everywhere else, and the label is
NEVER forwarded upstream — the sentinel tests at the bottom pin that.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
    OpenAICompatibleProviderConfig,
)
from vulcan.providers.base import Provider
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider
from vulcan.usage import UsageRecorder

OPENAI_KEY_ENV = "VULCAN_SEAT_TEST_OPENAI_KEY"
OPENAI_KEY_SENTINEL = "sk-seat-openai-secret-9c41"
SEAT_SENTINEL = "seat-must-not-escape-7b3d"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, OPENAI_KEY_SENTINEL)


# ── Recorder unit behavior ───────────────────────────────────────────────────


def test_recorder_aggregates_per_seat_only_when_labeled() -> None:
    recorder = UsageRecorder()
    recorder.record(model="a", provider="p1", prompt_tokens=3, completion_tokens=4, seat="claude")
    recorder.record(model="a", provider="p1", prompt_tokens=1, completion_tokens=1, seat="codex")
    recorder.record(model="a", provider="p1", prompt_tokens=10, completion_tokens=10)  # unlabeled

    snapshot = recorder.snapshot()

    assert snapshot.totals.requests == 3  # unlabeled requests still count overall
    by_seat = {item.seat: item.totals for item in snapshot.by_seat}
    assert set(by_seat) == {"claude", "codex"}
    assert by_seat["claude"].requests == 1
    assert by_seat["claude"].total_tokens == 7
    assert by_seat["codex"].total_tokens == 2


def test_recorder_seat_snapshot_is_sorted_and_stable() -> None:
    recorder = UsageRecorder()
    for seat in ("zeta", "alpha", "mid"):
        recorder.record(model="a", provider="p1", seat=seat)

    snapshot = recorder.snapshot()

    assert [item.seat for item in snapshot.by_seat] == ["alpha", "mid", "zeta"]
    # Snapshots are point-in-time copies: later records do not mutate them.
    recorder.record(model="a", provider="p1", seat="alpha")
    assert snapshot.by_seat[0].totals.requests == 1


# ── HTTP contract ────────────────────────────────────────────────────────────


def _config() -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
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


def _client() -> TestClient:
    app = create_app(_config(), clock=lambda: 1_700_000_000.0)
    return TestClient(app, base_url="http://127.0.0.1")


def _chat(
    client: TestClient,
    seat: str | None = None,
    stream: bool = False,
) -> httpx.Response:
    payload: dict[str, Any] = {
        "model": "alias-one",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": stream,
    }
    if seat is not None:
        payload["seat"] = seat
    return client.post("/v1/chat/completions", json=payload)


def test_labeled_chat_and_embeddings_appear_under_their_seat() -> None:
    with _client() as client:
        assert _chat(client, seat="claude").status_code == 200
        assert _chat(client, seat="claude").status_code == 200
        embed = client.post(
            "/v1/embeddings",
            json={"model": "alias-one", "input": "hello", "seat": "codex"},
        )
        assert embed.status_code == 200
        body = client.get("/v1/usage").json()

    seats = {item["seat"]: item["totals"] for item in body["by_seat"]}
    assert set(seats) == {"claude", "codex"}
    assert seats["claude"]["requests"] == 2
    assert seats["codex"]["requests"] == 1


def test_streamed_chat_records_its_seat() -> None:
    with _client() as client:
        response = _chat(client, seat="kimi", stream=True)
        assert response.status_code == 200
        response.read()  # drain the stream so usage is recorded
        body = client.get("/v1/usage").json()

    assert [item["seat"] for item in body["by_seat"]] == ["kimi"]


def test_unlabeled_requests_count_overall_but_never_under_a_seat() -> None:
    with _client() as client:
        assert _chat(client).status_code == 200
        body = client.get("/v1/usage").json()

    assert body["totals"]["requests"] == 1
    assert body["by_seat"] == []


@pytest.mark.parametrize(
    "seat",
    ["", "UPPER", "-leading-dash", "has space", "x" * 65, 7, "sneaky\nnewline"],
)
def test_invalid_seat_labels_are_rejected(seat: Any) -> None:
    with _client() as client:
        response = _chat(client, seat=seat)

    assert response.status_code == 422


def test_seat_is_rejected_on_unknown_endpoints_payloads() -> None:
    """StrictSchema still forbids extras: seat exists only where declared."""

    with _client() as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "alias-one",
                "messages": [{"role": "user", "content": "hello"}],
                "seat": "claude",
                "chair": "not-a-field",
            },
        )

    assert response.status_code == 422


# ── Upstream sentinels: the label never leaves the process ───────────────────


def _openai_config() -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        providers={
            "openai": OpenAICompatibleProviderConfig(
                type="openai_compatible",
                base_url="https://api.example.test/v1",
                api_key_env=OPENAI_KEY_ENV,
                timeout_seconds=1.0,
            ),
            "det": DeterministicProviderConfig(type="deterministic", response_text="canned"),
        },
        models=(
            ModelConfig(
                id="hosted-chat",
                provider="openai",
                provider_model="hosted-native",
                capabilities=frozenset({Capability.CHAT, Capability.EMBEDDINGS}),
            ),
        ),
    )


def _capturing_client(captured: list[httpx.Request]) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0]}], "usage": None},
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": None,
            },
        )

    config = _openai_config()
    providers: dict[str, Provider] = {}
    for provider_id, provider_config in config.providers.items():
        if provider_config.type == "deterministic":
            providers[provider_id] = DeterministicProvider(provider_id, provider_config)
            continue
        providers[provider_id] = OpenAICompatibleProvider(
            provider_id,
            provider_config,
            client=httpx.AsyncClient(
                base_url=provider_config.base_url,
                transport=httpx.MockTransport(handler),
                trust_env=False,
            ),
        )
    app = create_app(config, providers=providers, clock=lambda: 1_700_000_000.0)
    return TestClient(app, base_url="http://127.0.0.1")


def test_seat_never_appears_in_upstream_chat_body() -> None:
    captured: list[httpx.Request] = []
    with _capturing_client(captured) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hosted-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "seat": SEAT_SENTINEL,
            },
        )
        assert response.status_code == 200

    assert len(captured) == 1
    upstream = json.loads(captured[0].content.decode())
    assert "seat" not in upstream
    assert SEAT_SENTINEL not in captured[0].content.decode()
    assert SEAT_SENTINEL not in str(captured[0].headers)


def test_seat_never_appears_in_upstream_embeddings_body() -> None:
    captured: list[httpx.Request] = []
    with _capturing_client(captured) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "hosted-chat", "input": "hello", "seat": SEAT_SENTINEL},
        )
        assert response.status_code == 200

    assert len(captured) == 1
    assert SEAT_SENTINEL not in captured[0].content.decode()
    assert SEAT_SENTINEL not in str(captured[0].headers)


def _sse(*payloads: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(item)}\n\n" for item in payloads) + "data: [DONE]\n\n"
    return httpx.Response(
        200,
        content=body.encode(),
        headers={"content-type": "text/event-stream"},
    )


def test_seat_never_appears_in_upstream_or_downstream_stream(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ROADMAP §1.5: the streamed path gets its own sentinel, request and reply."""

    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return _sse(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": None},
        )

    config = _openai_config()
    providers: dict[str, Provider] = {
        "det": DeterministicProvider("det", config.providers["det"]),
        "openai": OpenAICompatibleProvider(
            "openai",
            config.providers["openai"],
            client=httpx.AsyncClient(
                base_url=config.providers["openai"].base_url,
                transport=httpx.MockTransport(handler),
                trust_env=False,
            ),
        ),
    }
    app = create_app(config, providers=providers, clock=lambda: 1_700_000_000.0)
    with (
        caplog.at_level(logging.INFO, logger="vulcan.gateway"),
        TestClient(app, base_url="http://127.0.0.1") as client,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hosted-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "seat": SEAT_SENTINEL,
            },
        )
        assert response.status_code == 200
        streamed = response.read().decode()
        usage_body = client.get("/v1/usage").json()

    assert len(captured) == 1
    assert SEAT_SENTINEL not in captured[0].content.decode()
    assert SEAT_SENTINEL not in str(captured[0].headers)
    assert SEAT_SENTINEL not in streamed
    # Positive capture proof first, so the negative assertion cannot be vacuous.
    assert "chat_completed" in caplog.text
    assert SEAT_SENTINEL not in caplog.text
    # The stream still recorded its seat once fully drained.
    assert [item["seat"] for item in usage_body["by_seat"]] == [SEAT_SENTINEL]


def test_seat_never_appears_in_error_responses_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A labeled request that fails upstream echoes neither seat nor detail."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "upstream boom"}})

    config = _openai_config()
    providers: dict[str, Provider] = {
        "det": DeterministicProvider("det", config.providers["det"]),
        "openai": OpenAICompatibleProvider(
            "openai",
            config.providers["openai"],
            client=httpx.AsyncClient(
                base_url=config.providers["openai"].base_url,
                transport=httpx.MockTransport(handler),
                trust_env=False,
            ),
        ),
    }
    app = create_app(config, providers=providers, clock=lambda: 1_700_000_000.0)
    with (
        caplog.at_level(logging.INFO, logger="vulcan.gateway"),
        TestClient(app, base_url="http://127.0.0.1") as client,
    ):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "hosted-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "seat": SEAT_SENTINEL,
            },
        )
        usage_body = client.get("/v1/usage").json()

    assert response.status_code >= 500
    assert SEAT_SENTINEL not in response.text
    # The failure was logged (positive capture proof), just never with the seat.
    assert caplog.text != ""
    assert SEAT_SENTINEL not in caplog.text
    # Failures are never usage: the seat records nothing.
    assert usage_body["by_seat"] == []


def test_labeled_success_logs_never_carry_the_seat(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Content-safe logs stay seat-free on the success path too."""

    with (
        caplog.at_level(logging.INFO, logger="vulcan.gateway"),
        _client() as client,
    ):
        assert _chat(client, seat=SEAT_SENTINEL).status_code == 200

    # Positive capture proof first, so the negative assertion cannot be vacuous.
    assert "chat_completed" in caplog.text
    assert SEAT_SENTINEL not in caplog.text


def _streaming_client(handler: Any, captured: list[httpx.Request]) -> TestClient:
    def capturing(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return handler(request)

    config = _openai_config()
    providers: dict[str, Provider] = {
        "det": DeterministicProvider("det", config.providers["det"]),
        "openai": OpenAICompatibleProvider(
            "openai",
            config.providers["openai"],
            client=httpx.AsyncClient(
                base_url=config.providers["openai"].base_url,
                transport=httpx.MockTransport(capturing),
                trust_env=False,
            ),
        ),
    }
    app = create_app(config, providers=providers, clock=lambda: 1_700_000_000.0)
    return TestClient(app, base_url="http://127.0.0.1")


def _labeled_stream(client: TestClient) -> str:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "hosted-chat",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "seat": SEAT_SENTINEL,
        },
    )
    return response.read().decode()


def test_seat_records_when_upstream_omits_the_done_terminator(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """EOF without [DONE] is tolerated upstream, so the request completes.

    The adapter yields a normal StreamEnd at upstream EOF (finish_reason
    None); the gateway records it as a completed request with no reported
    tokens. The seat rides along into by_seat and escapes nowhere.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = 'data: {"choices": [{"delta": {"content": "partial"}, "finish_reason": null}]}\n\n'
        return httpx.Response(
            200,
            content=body.encode(),  # no terminal chunk, no [DONE]
            headers={"content-type": "text/event-stream"},
        )

    captured: list[httpx.Request] = []
    with (
        caplog.at_level(logging.INFO, logger="vulcan.gateway"),
        _streaming_client(handler, captured) as client,
    ):
        streamed = _labeled_stream(client)
        usage_body = client.get("/v1/usage").json()

    assert SEAT_SENTINEL not in captured[0].content.decode()
    assert SEAT_SENTINEL not in streamed
    assert "chat_completed" in caplog.text
    assert SEAT_SENTINEL not in caplog.text
    seats = {item["seat"]: item["totals"] for item in usage_body["by_seat"]}
    assert seats[SEAT_SENTINEL]["requests"] == 1
    assert seats[SEAT_SENTINEL]["requests_with_usage"] == 0


def test_seat_never_escapes_on_mid_stream_protocol_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A malformed chunk after real deltas is the true mid-stream failure.

    The adapter raises ProviderProtocolError, the gateway records no usage,
    and the seat appears in neither the partial bytes nor the logs.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"choices": [{"delta": {"content": "partial"}, "finish_reason": null}]}\n\n'
            "data: {not-json\n\n"
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    captured: list[httpx.Request] = []
    with (
        caplog.at_level(logging.INFO, logger="vulcan.gateway"),
        _streaming_client(handler, captured) as client,
    ):
        streamed = ""
        # The mid-body failure may surface as a transport-level error; the
        # sentinels below still apply to whatever was produced before it died.
        with contextlib.suppress(Exception):
            streamed = _labeled_stream(client)
        usage_body = client.get("/v1/usage").json()

    assert SEAT_SENTINEL not in captured[0].content.decode()
    assert SEAT_SENTINEL not in streamed
    assert caplog.text != ""  # the failure produced log records...
    assert SEAT_SENTINEL not in caplog.text  # ...none of which carry the seat
    # A stream that failed mid-flight is not usage, so the seat records nothing.
    assert usage_body["by_seat"] == []
