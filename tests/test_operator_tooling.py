"""Phase 3: single-flight probes, credential verification, socket hygiene.

Every upstream exchange goes through ``httpx.MockTransport``; no test contacts
a real API. Credentials are synthetic sentinels so the leak assertions mean
something.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
import uvicorn

import vulcan.cli as cli
from vulcan.config import (
    AnthropicProviderConfig,
    Capability,
    ModelConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
)
from vulcan.errors import ProviderProtocolError
from vulcan.gateway import Gateway
from vulcan.providers.anthropic import ANTHROPIC_VERSION, AnthropicProvider
from vulcan.providers.base import ProviderChatRequest, ProviderMessage
from vulcan.providers.http import verify_hosted_credential
from vulcan.providers.ollama import OllamaProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider
from vulcan.readiness import RuntimeProbe
from vulcan.registry import ModelRegistry

OPENAI_KEY_ENV = "VULCAN_TOOLING_TEST_OPENAI_KEY"
ANTHROPIC_KEY_ENV = "VULCAN_TOOLING_TEST_ANTHROPIC_KEY"
OPENAI_KEY_SENTINEL = "sk-tooling-openai-secret-5c19"
ANTHROPIC_KEY_SENTINEL = "sk-tooling-ant-secret-8ba2"
BODY_SENTINEL = "tooling-upstream-body-must-not-escape-4d77"

MULTI_PROVIDER_CONFIG = f"""\
schema_version = 2

[providers.local-ollama]
type = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 1.0

[providers.openai]
type = "openai_compatible"
base_url = "https://api.compat-mock.example/v1"
api_key_env = "{OPENAI_KEY_ENV}"
timeout_seconds = 1.0

[providers.anthropic]
type = "anthropic"
base_url = "https://api.anthropic-mock.example"
api_key_env = "{ANTHROPIC_KEY_ENV}"
timeout_seconds = 1.0

[[models]]
id = "local-chat"
provider = "local-ollama"
provider_model = "some-installed-model"
capabilities = ["chat"]
"""


@pytest.fixture(autouse=True)
def _credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OPENAI_KEY_ENV, OPENAI_KEY_SENTINEL)
    monkeypatch.setenv(ANTHROPIC_KEY_ENV, ANTHROPIC_KEY_SENTINEL)


# ── Single-flight readiness probes ───────────────────────────────────────────


class _SlowProbeProvider:
    """Counts probes and yields control mid-probe so callers can overlap."""

    provider_type: Literal["ollama"] = "ollama"

    def __init__(self, provider_id: str = "local-ollama") -> None:
        self.provider_id = provider_id
        self.probe_calls = 0

    async def discover_runtime(self) -> RuntimeProbe:
        self.probe_calls += 1
        await asyncio.sleep(0)  # let a concurrent caller reach the lock
        await asyncio.sleep(0)
        return RuntimeProbe(
            live=True,
            provider_availability="available",
            runtime_names=frozenset({"native-model"}),
        )

    async def chat(self, request: ProviderChatRequest) -> Any:  # pragma: no cover
        raise AssertionError("chat is not exercised here")

    async def aclose(self) -> None:
        return None


def _registry(provider_id: str = "local-ollama") -> ModelRegistry:
    return ModelRegistry(
        (
            ModelConfig(
                id="public-chat",
                provider=provider_id,
                provider_model="native-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        )
    )


def test_concurrent_readiness_calls_share_one_probe() -> None:
    provider = _SlowProbeProvider()
    gateway = Gateway(_registry(), {"local-ollama": provider})

    async def run() -> list[Any]:
        return list(await asyncio.gather(*(gateway.readiness() for _ in range(5))))

    reports = asyncio.run(run())

    assert provider.probe_calls == 1
    assert all(report.models[0].availability == "available" for report in reports)


def test_concurrent_chats_share_one_preflight_probe() -> None:
    from vulcan.schemas import ChatCompletionRequest, ChatMessage, MessageRole

    class _ChattyProvider(_SlowProbeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.chat_calls = 0

        async def chat(self, request: ProviderChatRequest) -> Any:
            from vulcan.providers.base import ProviderChatResult

            self.chat_calls += 1
            return ProviderChatResult(content="ok", finish_reason="stop")

    provider = _ChattyProvider()
    gateway = Gateway(_registry(), {"local-ollama": provider})
    request = ChatCompletionRequest(
        model="public-chat",
        messages=(ChatMessage(role=MessageRole.USER, content="hello"),),
    )

    async def run() -> None:
        await asyncio.gather(*(gateway.chat(request) for _ in range(4)))

    asyncio.run(run())

    # One shared probe, but every request still reached the provider.
    assert provider.probe_calls == 1
    assert provider.chat_calls == 4


def test_forced_refresh_still_probes_even_when_cached() -> None:
    provider = _SlowProbeProvider()
    gateway = Gateway(_registry(), {"local-ollama": provider})

    async def run() -> None:
        await gateway.readiness()
        await gateway.readiness(force=True)

    asyncio.run(run())

    assert provider.probe_calls == 2


def test_single_flight_is_per_provider_not_global() -> None:
    first = _SlowProbeProvider("first")
    second = _SlowProbeProvider("second")
    registry = ModelRegistry(
        (
            ModelConfig(
                id="a",
                provider="first",
                provider_model="native-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
            ModelConfig(
                id="b",
                provider="second",
                provider_model="native-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        )
    )
    gateway = Gateway(registry, {"first": first, "second": second})

    asyncio.run(gateway.readiness())

    assert first.probe_calls == 1
    assert second.probe_calls == 1


# ── Credential verification (operator-invoked only) ──────────────────────────


def _compat_config() -> OpenAICompatibleProviderConfig:
    return OpenAICompatibleProviderConfig(
        type="openai_compatible",
        base_url="https://api.compat-mock.example/v1",
        api_key_env=OPENAI_KEY_ENV,
        timeout_seconds=1.0,
    )


def _anthropic_config() -> AnthropicProviderConfig:
    return AnthropicProviderConfig(
        type="anthropic",
        base_url="https://api.anthropic-mock.example",
        api_key_env=ANTHROPIC_KEY_ENV,
        timeout_seconds=1.0,
    )


def _verify(config: Any, handler: Any) -> str:
    """Run one verification against a mocked transport."""

    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    return asyncio.run(verify_hosted_credential(config, client=client))


def test_verification_calls_the_openai_compatible_models_endpoint() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": []})

    assert _verify(_compat_config(), handler) == "verified"
    assert captured[0].method == "GET"
    assert str(captured[0].url) == "https://api.compat-mock.example/v1/models"
    assert captured[0].headers["Authorization"] == f"Bearer {OPENAI_KEY_SENTINEL}"


def test_verification_calls_the_anthropic_models_endpoint_with_version_header() -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"data": []})

    assert _verify(_anthropic_config(), handler) == "verified"
    assert str(captured[0].url) == "https://api.anthropic-mock.example/v1/models"
    assert captured[0].headers["x-api-key"] == ANTHROPIC_KEY_SENTINEL
    assert captured[0].headers["anthropic-version"] == ANTHROPIC_VERSION
    assert "authorization" not in captured[0].headers


@pytest.mark.parametrize(
    ("status", "expected"),
    [(200, "verified"), (401, "auth_failed"), (403, "auth_failed"), (429, "error"), (500, "error")],
)
def test_verification_verdicts_come_from_status_codes_only(status: int, expected: str) -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": BODY_SENTINEL}})

    assert _verify(_compat_config(), handler) == expected


def test_verification_reports_unreachable_for_transport_and_timeout_failures() -> None:
    async def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert _verify(_compat_config(), refused) == "unreachable"

    async def timed_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    assert _verify(_compat_config(), timed_out) == "unreachable"


def test_verification_reports_missing_without_any_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)

    async def handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("no upstream call may happen without a credential")

    assert _verify(_compat_config(), handler) == "missing"


# ── vulcan check --verify-credentials ────────────────────────────────────────


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "vulcan.toml"
    path.write_text(MULTI_PROVIDER_CONFIG, encoding="utf-8")
    return path


def _patch_verification(monkeypatch: pytest.MonkeyPatch, verdicts: dict[str, str]) -> list[str]:
    """Record which providers were verified, returning verdicts by type."""

    seen: list[str] = []

    async def fake_verify(config: Any) -> str:
        seen.append(config.type)
        return verdicts[config.type]

    monkeypatch.setattr(cli, "verify_hosted_credential", fake_verify)
    return seen


def test_check_without_the_flag_makes_no_network_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden(config: Any) -> str:
        raise AssertionError("verification must not run without --verify-credentials")

    monkeypatch.setattr(cli, "verify_hosted_credential", forbidden)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("no server"))

    exit_code = cli.main(["check", "--config", str(_write_config(tmp_path))])

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "credentials_verified" not in report
    assert all("verification" not in entry for entry in report["providers"])


def test_check_verify_credentials_reports_per_provider_verdicts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch_verification(
        monkeypatch, {"openai_compatible": "verified", "anthropic": "auth_failed"}
    )

    exit_code = cli.main(
        ["check", "--config", str(_write_config(tmp_path)), "--verify-credentials"]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    by_id = {entry["id"]: entry for entry in report["providers"]}
    assert exit_code == 1  # one hosted provider did not verify
    assert report["credentials_verified"] is True
    assert report["verification_failures"] == 1
    assert by_id["openai"]["verification"] == "verified"
    assert by_id["anthropic"]["verification"] == "auth_failed"
    # Local providers hold no credential and are never probed.
    assert "verification" not in by_id["local-ollama"]
    assert sorted(seen) == ["anthropic", "openai_compatible"]
    for sentinel in (OPENAI_KEY_SENTINEL, ANTHROPIC_KEY_SENTINEL, BODY_SENTINEL):
        assert sentinel not in captured.out
        assert sentinel not in captured.err


def test_check_verify_credentials_exits_zero_when_every_hosted_provider_verifies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_verification(monkeypatch, {"openai_compatible": "verified", "anthropic": "verified"})

    exit_code = cli.main(
        ["check", "--config", str(_write_config(tmp_path)), "--verify-credentials"]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["verification_failures"] == 0


@pytest.mark.parametrize("verdict", ["auth_failed", "unreachable", "error", "missing"])
def test_check_verify_credentials_fails_for_any_non_verified_verdict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    verdict: str,
) -> None:
    _patch_verification(monkeypatch, {"openai_compatible": verdict, "anthropic": "verified"})

    exit_code = cli.main(
        ["check", "--config", str(_write_config(tmp_path)), "--verify-credentials"]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["verification_failures"] == 1


def test_check_verify_credentials_still_rejects_an_invalid_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def forbidden(config: Any) -> str:
        raise AssertionError("an invalid config must fail before any verification")

    monkeypatch.setattr(cli, "verify_hosted_credential", forbidden)
    path = tmp_path / "vulcan.toml"
    path.write_text("schema_version = 2\n", encoding="utf-8")

    exit_code = cli.main(["check", "--config", str(path), "--verify-credentials"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"]["code"] == "configuration_error"


# ── Socket hygiene on streaming error paths ──────────────────────────────────


class _RecordingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def _stream_request() -> ProviderChatRequest:
    return ProviderChatRequest(
        provider_model="native-model",
        messages=(ProviderMessage(role="user", content="hello"),),
        temperature=None,
        max_tokens=None,
    )


def _drain_expecting_error(provider: Any) -> None:
    async def run() -> None:
        try:
            async for _ in provider.chat_stream(_stream_request()):
                pass
        finally:
            await provider.aclose()

    with pytest.raises(ProviderProtocolError):
        asyncio.run(run())


def test_compat_stream_closes_upstream_on_a_malformed_frame() -> None:
    stream = _RecordingStream(
        [b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n', b"data: {\n\n"]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url="https://api.compat-mock.example/v1",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    _drain_expecting_error(OpenAICompatibleProvider("compat", _compat_config(), client=client))

    assert stream.closed is True


def test_anthropic_stream_closes_upstream_on_a_protocol_error() -> None:
    stream = _RecordingStream(
        [
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 1}}}\n\n',
            b'data: {"type": "content_block_start", "content_block": {"type": "tool_use"}}\n\n',
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url="https://api.anthropic-mock.example",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    _drain_expecting_error(AnthropicProvider("anthropic", _anthropic_config(), client=client))

    assert stream.closed is True


def test_ollama_stream_closes_upstream_on_a_truncated_stream() -> None:
    stream = _RecordingStream(
        [b'{"message": {"role": "assistant", "content": "x"}, "done": false}\n']
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    config = OllamaProviderConfig(
        type="ollama", base_url="http://127.0.0.1:11434", timeout_seconds=1.0
    )
    client = httpx.AsyncClient(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    _drain_expecting_error(OllamaProvider("local-ollama", config, client=client))

    assert stream.closed is True


def test_streams_close_upstream_on_a_clean_finish() -> None:
    stream = _RecordingStream(
        [
            b'data: {"choices": [{"delta": {"content": "done"}}]}\n\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, headers={"content-type": "text/event-stream"})

    client = httpx.AsyncClient(
        base_url="https://api.compat-mock.example/v1",
        transport=httpx.MockTransport(handler),
        trust_env=False,
    )
    provider = OpenAICompatibleProvider("compat", _compat_config(), client=client)

    async def run() -> None:
        try:
            async for _ in provider.chat_stream(_stream_request()):
                pass
        finally:
            await provider.aclose()

    asyncio.run(run())

    assert stream.closed is True
