"""CLI memory-lifecycle commands (ps/unload/warmup) against Ollama.

These are explicit operator actions: exactly one upstream call per command,
only to the alias's own Ollama provider, with hosted and deterministic
providers refused before any client is built. All upstream traffic is mocked
at the transport seam; the pins that matter are the refusal-before-network
guarantee and that native model names never reach the operator's terminal
except in the documented `ps` unmanaged-rows exception.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import vulcan.cli as cli

NATIVE_CHAT = "private-native-chat-tag"
NATIVE_EMBED = "private-native-embed-tag"
NATIVE_HOSTED = "private-vendor-model"

CONFIG = f"""\
schema_version = 2

[server]
host = "127.0.0.1"
port = 18140
log_level = "INFO"

[providers.local-ollama]
type = "ollama"
base_url = "http://127.0.0.1:11434"
timeout_seconds = 60.0

[providers.openai]
type = "openai_compatible"
base_url = "https://api.example-vendor.com/v1"
api_key_env = "VULCAN_CLI_OPS_TEST_KEY"
timeout_seconds = 60.0

[providers.det]
type = "deterministic"
response_text = "canned"

[[models]]
id = "local-chat"
provider = "local-ollama"
provider_model = "{NATIVE_CHAT}"
capabilities = ["chat"]
keep_alive = "1h"

[[models]]
id = "local-chat-alt"
provider = "local-ollama"
provider_model = "{NATIVE_CHAT}"
capabilities = ["chat"]

[[models]]
id = "local-embed"
provider = "local-ollama"
provider_model = "{NATIVE_EMBED}"
capabilities = ["embeddings"]

[[models]]
id = "cloud-chat"
provider = "openai"
provider_model = "{NATIVE_HOSTED}"
capabilities = ["chat"]

[[models]]
id = "canned-chat"
provider = "det"
provider_model = "canned"
capabilities = ["chat"]
"""


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "vulcan.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def _factory(
    recorded: list[str], handler: Callable[[httpx.Request], object]
) -> Callable[[Any], httpx.AsyncClient]:
    """A client factory that records which providers got clients and mocks I/O."""

    def make(provider: Any) -> httpx.AsyncClient:
        base_url = provider.base_url
        recorded.append(base_url)
        return httpx.AsyncClient(
            base_url=base_url,
            transport=httpx.MockTransport(handler),
            timeout=5.0,
        )

    return make


def test_ps_maps_resident_models_to_aliases_and_flags_unmanaged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": NATIVE_CHAT, "size_vram": 18_000_000_000, "expires_at": "forever"},
                    {"name": "rogue-model:latest", "size_vram": 5_000, "expires_at": "soon"},
                ]
            },
        )

    recorded: list[str] = []
    monkeypatch.setattr(cli, "_ollama_client", _factory(recorded, handler))

    exit_code = cli.main(["ps", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    # Only the Ollama provider got a client; hosted/deterministic are skipped
    # without any network object being built for them.
    assert recorded == ["http://127.0.0.1:11434"]
    report = json.loads(capsys.readouterr().out)
    by_id = {entry["id"]: entry for entry in report["providers"]}
    assert by_id["openai"] == {"id": "openai", "type": "openai_compatible", "status": "skipped"}
    assert by_id["det"] == {"id": "det", "type": "deterministic", "status": "skipped"}
    ollama = by_id["local-ollama"]
    assert ollama["status"] == "ok"
    mapped, rogue = ollama["models"]
    assert mapped == {
        "name": NATIVE_CHAT,
        "alias": "local-chat",
        "size_vram": 18_000_000_000,
        "expires_at": "forever",
    }
    # The documented exception: unmanaged rows carry the native name because
    # this is the operator's own terminal and the config file holds it anyway.
    assert rogue["alias"] is None
    assert rogue["unmanaged"] is True
    assert rogue["name"] == "rogue-model:latest"


def test_ps_shared_native_model_maps_to_first_configured_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # local-chat and local-chat-alt both ride NATIVE_CHAT; the first
    # configured alias wins the ps mapping, deterministically.
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"models": [{"name": NATIVE_CHAT, "size_vram": 1, "expires_at": "soon"}]},
        )

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["ps", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    ollama = next(e for e in report["providers"] if e["id"] == "local-ollama")
    assert ollama["models"][0]["alias"] == "local-chat"


def test_ps_treats_non_object_entries_as_a_malformed_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A models list holding anything but objects gets the same handling as no
    # answer at all — a sanitized unreachable row, never a traceback.
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [f"rogue-string-{NATIVE_CHAT}"]})

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["ps", "--config", str(_write_config(tmp_path))])

    assert exit_code == 1
    output = capsys.readouterr().out
    ollama = next(e for e in json.loads(output)["providers"] if e["id"] == "local-ollama")
    assert ollama["status"] == "unreachable"
    assert NATIVE_CHAT not in output


def test_ps_marks_unreachable_without_echoing_exception_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"connection refused by {NATIVE_CHAT} host")

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["ps", "--config", str(_write_config(tmp_path))])

    assert exit_code == 1
    output = capsys.readouterr().out
    ollama = next(e for e in json.loads(output)["providers"] if e["id"] == "local-ollama")
    assert ollama["status"] == "unreachable"
    # The exception text (which here embeds a native model name) is never echoed.
    assert "connection refused" not in output
    assert NATIVE_CHAT not in output


def test_unload_sends_keep_alive_zero_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"response": ""})

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["unload", "local-chat", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].url.path == "/api/generate"
    assert json.loads(captured[0].content) == {"model": NATIVE_CHAT, "keep_alive": 0}
    out = capsys.readouterr().out
    assert json.loads(out) == {
        "action": "unload",
        "alias": "local-chat",
        "provider": "local-ollama",
        "status": "unloaded",
    }
    # The native model name never reaches the operator's terminal.
    assert NATIVE_CHAT not in out


def test_warmup_sends_empty_generate_with_configured_keep_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"response": ""})

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["warmup", "local-chat", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    assert len(captured) == 1
    assert json.loads(captured[0].content) == {
        "model": NATIVE_CHAT,
        "prompt": "",
        "stream": False,
        "keep_alive": "1h",
    }
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] == "warm"
    assert result["keep_alive"] == "1h"
    assert NATIVE_CHAT not in out


def test_warmup_embeddings_alias_rides_the_embed_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Embedding-only models refuse /api/generate ("does not support generate"),
    # so their load request is an empty-input /api/embed. No configured
    # keep_alive on this alias means the key is omitted from wire and output.
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"model": NATIVE_EMBED, "embeddings": []})

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["warmup", "local-embed", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].url.path == "/api/embed"
    assert json.loads(captured[0].content) == {"model": NATIVE_EMBED, "input": []}
    out = capsys.readouterr().out
    assert json.loads(out)["status"] == "warm"
    assert "keep_alive" not in out
    assert NATIVE_EMBED not in out


def test_unload_embeddings_alias_rides_the_embed_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json={"model": NATIVE_EMBED, "embeddings": []})

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["unload", "local-embed", "--config", str(_write_config(tmp_path))])

    assert exit_code == 0
    assert len(captured) == 1
    assert captured[0].url.path == "/api/embed"
    assert json.loads(captured[0].content) == {
        "model": NATIVE_EMBED,
        "input": [],
        "keep_alive": 0,
    }
    out = capsys.readouterr().out
    assert json.loads(out)["status"] == "unloaded"
    assert NATIVE_EMBED not in out


def test_hosted_and_deterministic_aliases_are_refused_before_any_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a refused alias must never reach the transport")

    recorded: list[str] = []
    monkeypatch.setattr(cli, "_ollama_client", _factory(recorded, handler))

    for alias in ("cloud-chat", "canned-chat"):
        for action in ("unload", "warmup"):
            exit_code = cli.main([action, alias, "--config", str(_write_config(tmp_path))])
            assert exit_code == 2
            assert json.loads(capsys.readouterr().err)["error"]["code"] == (
                "unsupported_provider_type"
            )

    # Not even a client was built for the refused aliases.
    assert recorded == []


def test_unknown_alias_exits_2_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorded: list[str] = []
    monkeypatch.setattr(cli, "_ollama_client", _factory(recorded, None))

    exit_code = cli.main(["unload", "no-such-alias", "--config", str(_write_config(tmp_path))])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "model_not_found"
    assert recorded == []


def test_ollama_404_is_model_unavailable_without_echoing_the_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": f"model '{NATIVE_CHAT}' not found"})

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["warmup", "local-chat", "--config", str(_write_config(tmp_path))])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert json.loads(err)["error"]["code"] == "model_unavailable"
    # The upstream body names the native model; it must not be echoed.
    assert NATIVE_CHAT not in err
    assert "not found" not in err


def test_ollama_down_is_a_sanitized_provider_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused while loading {NATIVE_CHAT}")

    monkeypatch.setattr(cli, "_ollama_client", _factory([], handler))

    exit_code = cli.main(["unload", "local-chat", "--config", str(_write_config(tmp_path))])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert json.loads(err)["error"]["code"] == "provider_unreachable"
    assert NATIVE_CHAT not in err
    assert "refused while loading" not in err


def test_hosted_native_name_never_appears_in_action_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorded: list[str] = []
    monkeypatch.setattr(cli, "_ollama_client", _factory(recorded, None))

    exit_code = cli.main(["unload", "cloud-chat", "--config", str(_write_config(tmp_path))])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert NATIVE_HOSTED not in captured.out
    assert NATIVE_HOSTED not in captured.err
