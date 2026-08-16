"""Process-lifetime usage counters: recorder, gateway wiring, HTTP contract.

Counters are in-memory only and count *completed* requests; failures never
count as usage, and token totals only cover requests whose upstream actually
reported counts.
"""

from __future__ import annotations

import json
from typing import Any, Literal

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
from vulcan.usage import UsageRecorder

OPENAI_KEY_ENV = "VULCAN_USAGE_TEST_OPENAI_KEY"
OPENAI_KEY_SENTINEL = "sk-usage-openai-secret-2f6a"
PROMPT_SENTINEL = "usage-prompt-must-not-escape-51ee"


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, OPENAI_KEY_SENTINEL)


# ── Recorder unit behavior ───────────────────────────────────────────────────


def test_recorder_starts_empty() -> None:
    snapshot = UsageRecorder().snapshot()

    assert snapshot.totals.requests == 0
    assert snapshot.totals.total_tokens == 0
    assert snapshot.by_model == ()
    assert snapshot.by_provider == ()


def test_recorder_aggregates_per_model_and_per_provider() -> None:
    recorder = UsageRecorder()
    recorder.record(model="a", provider="p1", prompt_tokens=3, completion_tokens=4)
    recorder.record(model="b", provider="p1", prompt_tokens=1, completion_tokens=1)
    recorder.record(model="c", provider="p2", prompt_tokens=10, completion_tokens=0)

    snapshot = recorder.snapshot()

    assert snapshot.totals.requests == 3
    assert snapshot.totals.prompt_tokens == 14
    assert snapshot.totals.completion_tokens == 5
    assert snapshot.totals.total_tokens == 19
    by_provider = {item.provider: item.totals for item in snapshot.by_provider}
    assert by_provider["p1"].requests == 2
    assert by_provider["p1"].total_tokens == 9
    assert by_provider["p2"].total_tokens == 10
    by_model = {item.model: item for item in snapshot.by_model}
    assert by_model["a"].provider == "p1"
    assert by_model["c"].totals.requests == 1


def test_recorder_counts_requests_without_reported_tokens_separately() -> None:
    recorder = UsageRecorder()
    recorder.record(model="a", provider="p1")  # upstream reported nothing
    recorder.record(model="a", provider="p1", prompt_tokens=5, completion_tokens=2)

    totals = recorder.snapshot().totals

    assert totals.requests == 2
    assert totals.requests_with_usage == 1
    assert totals.total_tokens == 7


def test_recorder_accepts_prompt_only_usage_for_embeddings() -> None:
    recorder = UsageRecorder()
    recorder.record(model="embed", provider="p1", prompt_tokens=12)

    totals = recorder.snapshot().totals

    assert totals.requests_with_usage == 1
    assert totals.prompt_tokens == 12
    assert totals.completion_tokens == 0
    assert totals.total_tokens == 12


def test_recorder_snapshot_is_sorted_and_stable() -> None:
    recorder = UsageRecorder()
    for model, provider in (("z", "p2"), ("a", "p1"), ("m", "p1")):
        recorder.record(model=model, provider=provider)

    snapshot = recorder.snapshot()

    assert [item.model for item in snapshot.by_model] == ["a", "m", "z"]
    assert [item.provider for item in snapshot.by_provider] == ["p1", "p2"]
    # Snapshots are point-in-time copies: later records do not mutate them.
    recorder.record(model="a", provider="p1")
    assert snapshot.totals.requests == 3


# ── HTTP contract ────────────────────────────────────────────────────────────


class _StubProvider:
    """Chat/embeddings double whose reported usage is configurable."""

    provider_type: Literal["deterministic"] = "deterministic"

    def __init__(
        self,
        provider_id: str = "det",
        usage: tuple[int, int] | None = (7, 3),
        failure: Exception | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.usage = usage
        self.failure = failure

    async def chat(self, request: Any) -> Any:
        from vulcan.providers.base import ProviderChatResult, ProviderTokenUsage

        if self.failure is not None:
            raise self.failure
        usage = None
        if self.usage is not None:
            usage = ProviderTokenUsage(prompt_tokens=self.usage[0], completion_tokens=self.usage[1])
        return ProviderChatResult(content="reply", finish_reason="stop", usage=usage)

    async def chat_stream(self, request: Any) -> Any:
        from vulcan.providers.base import ProviderTokenUsage, StreamDelta, StreamEnd

        if self.failure is not None:
            raise self.failure
        yield StreamDelta(text="reply")
        usage = None
        if self.usage is not None:
            usage = ProviderTokenUsage(prompt_tokens=self.usage[0], completion_tokens=self.usage[1])
        yield StreamEnd(finish_reason="stop", usage=usage)

    async def embed(self, request: Any) -> Any:
        from vulcan.providers.base import ProviderEmbeddingResult, ProviderEmbeddingUsage

        if self.failure is not None:
            raise self.failure
        usage = None
        if self.usage is not None:
            usage = ProviderEmbeddingUsage(prompt_tokens=self.usage[0], total_tokens=self.usage[0])
        return ProviderEmbeddingResult(vectors=tuple((1.0,) for _ in request.inputs), usage=usage)

    async def discover_runtime(self) -> Any:
        from vulcan.readiness import RuntimeProbe

        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        return None


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
            ModelConfig(
                id="alias-two",
                provider="det",
                provider_model="native-two",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )


def _client(provider: _StubProvider | None = None) -> TestClient:
    app = create_app(
        _config(),
        providers={"det": provider or _StubProvider()},
        clock=lambda: 1_700_000_000.0,
    )
    return TestClient(app, base_url="http://127.0.0.1")


def _chat(client: TestClient, model: str = "alias-one", stream: bool = False) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
            "stream": stream,
        },
    )


def test_usage_starts_empty_and_reports_process_scope() -> None:
    with _client() as client:
        body = client.get("/v1/usage").json()

    assert body == {
        "object": "usage",
        "scope": "process",
        "started_at": 1_700_000_000,
        "totals": {
            "requests": 0,
            "requests_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "by_model": [],
        "by_provider": [],
        "by_seat": [],
    }


def test_usage_counts_buffered_chat_per_alias_and_provider() -> None:
    with _client() as client:
        assert _chat(client, "alias-one").status_code == 200
        assert _chat(client, "alias-two").status_code == 200
        body = client.get("/v1/usage").json()

    assert body["totals"] == {
        "requests": 2,
        "requests_with_usage": 2,
        "prompt_tokens": 14,
        "completion_tokens": 6,
        "total_tokens": 20,
    }
    assert [item["model"] for item in body["by_model"]] == ["alias-one", "alias-two"]
    assert all(item["provider"] == "det" for item in body["by_model"])
    assert body["by_provider"] == [
        {
            "provider": "det",
            "totals": {
                "requests": 2,
                "requests_with_usage": 2,
                "prompt_tokens": 14,
                "completion_tokens": 6,
                "total_tokens": 20,
            },
        }
    ]


def test_usage_counts_streaming_chat_after_the_stream_completes() -> None:
    with _client() as client:
        response = _chat(client, stream=True)
        assert response.status_code == 200
        assert "[DONE]" in response.text
        body = client.get("/v1/usage").json()

    assert body["totals"]["requests"] == 1
    assert body["totals"]["total_tokens"] == 10


def test_usage_counts_embeddings_as_prompt_tokens_only() -> None:
    with _client() as client:
        response = client.post("/v1/embeddings", json={"model": "alias-one", "input": ["a", "b"]})
        assert response.status_code == 200
        body = client.get("/v1/usage").json()

    assert body["totals"] == {
        "requests": 1,
        "requests_with_usage": 1,
        "prompt_tokens": 7,
        "completion_tokens": 0,
        "total_tokens": 7,
    }


def test_usage_never_invents_tokens_when_the_upstream_reports_none() -> None:
    with _client(_StubProvider(usage=None)) as client:
        assert _chat(client).status_code == 200
        body = client.get("/v1/usage").json()

    assert body["totals"]["requests"] == 1
    assert body["totals"]["requests_with_usage"] == 0
    assert body["totals"]["total_tokens"] == 0


def test_failed_requests_are_not_counted_as_usage() -> None:
    from vulcan.errors import ProviderRateLimitError

    with _client(_StubProvider(failure=ProviderRateLimitError())) as client:
        assert _chat(client).status_code == 429
        assert _chat(client, stream=True).status_code == 429
        embed = client.post("/v1/embeddings", json={"model": "alias-one", "input": "x"})
        assert embed.status_code == 429
        body = client.get("/v1/usage").json()

    assert body["totals"]["requests"] == 0
    assert body["by_model"] == []


def test_unknown_alias_and_unsupported_capability_are_not_counted() -> None:
    with _client() as client:
        assert _chat(client, "missing-alias").status_code == 404
        assert (
            client.post("/v1/embeddings", json={"model": "alias-two", "input": "x"}).status_code
            == 422
        )
        body = client.get("/v1/usage").json()

    assert body["totals"]["requests"] == 0


def test_usage_survives_across_requests_within_one_process() -> None:
    with _client() as client:
        for _ in range(3):
            assert _chat(client).status_code == 200
        body = client.get("/v1/usage").json()

    assert body["totals"]["requests"] == 3
    assert body["by_model"][0]["totals"]["requests"] == 3


def test_usage_is_per_app_instance_and_resets_with_the_process() -> None:
    with _client() as client:
        _chat(client)
        assert client.get("/v1/usage").json()["totals"]["requests"] == 1

    # A fresh app is a fresh process lifetime.
    with _client() as client:
        assert client.get("/v1/usage").json()["totals"]["requests"] == 0


def test_usage_never_exposes_prompts_native_names_or_credentials() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "safe"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    from vulcan.providers.openai_compatible import OpenAICompatibleProvider

    provider_config = OpenAICompatibleProviderConfig(
        type="openai_compatible",
        base_url="https://api.compat-mock.example/v1",
        api_key_env=OPENAI_KEY_ENV,
        timeout_seconds=1.0,
    )
    config = GatewayConfig(
        schema_version=2,
        providers={"compat": provider_config},
        models=(
            ModelConfig(
                id="alias-one",
                provider="compat",
                provider_model="native-secret-name",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )
    provider = OpenAICompatibleProvider(
        "compat",
        provider_config,
        client=httpx.AsyncClient(
            base_url=provider_config.base_url,
            transport=httpx.MockTransport(handler),
            trust_env=False,
        ),
    )
    with TestClient(
        create_app(config, providers={"compat": provider}), base_url="http://127.0.0.1"
    ) as client:
        assert _chat(client).status_code == 200
        response = client.get("/v1/usage")

    assert response.status_code == 200
    assert response.json()["totals"]["total_tokens"] == 5
    for sentinel in (PROMPT_SENTINEL, OPENAI_KEY_SENTINEL, "native-secret-name", "safe"):
        assert sentinel not in response.text


def test_usage_rejects_non_loopback_hosts_like_every_other_route() -> None:
    with _client() as client:
        response = client.get("/v1/usage", headers={"host": "example.com"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_host"


def test_usage_is_documented_in_the_openapi_schema() -> None:
    with _client() as client:
        schema = client.get("/openapi.json").json()

    assert "/v1/usage" in schema["paths"]
    assert "get" in schema["paths"]["/v1/usage"]


def test_usage_response_shape_is_strict() -> None:
    """A stray field would be a contract break, so the model forbids extras."""

    from pydantic import ValidationError

    from vulcan.schemas import UsageResponse

    payload = {
        "object": "usage",
        "scope": "process",
        "started_at": 1,
        "totals": {
            "requests": 0,
            "requests_with_usage": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "by_model": [],
        "by_provider": [],
        "by_seat": [],
        "cost_usd": 0.0,
    }
    with pytest.raises(ValidationError):
        UsageResponse.model_validate(payload)

    del payload["cost_usd"]
    assert UsageResponse.model_validate(payload).totals.requests == 0


def test_usage_json_is_stable_for_operators() -> None:
    """The endpoint is meant to be scriptable: keys stay predictable."""

    with _client() as client:
        _chat(client)
        body = client.get("/v1/usage").json()

    assert set(body) == {
        "object",
        "scope",
        "started_at",
        "totals",
        "by_model",
        "by_provider",
        "by_seat",
    }
    assert set(body["totals"]) == {
        "requests",
        "requests_with_usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    assert json.dumps(body)  # serializable without custom encoders
