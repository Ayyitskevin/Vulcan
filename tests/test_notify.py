"""The raw-socket sd_notify client (vulcan.notify) and its lifespan wiring.

Real AF_UNIX datagram receivers on tmp paths — no mocking, no systemd
required. Pins: no-op without NOTIFY_SOCKET, READY=1 plus watchdog
heartbeats when set, abstract-namespace translation, fail-loud startup on
an undeliverable socket, and clean heartbeat shutdown with the app.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vulcan import notify
from vulcan.api import create_app
from vulcan.config import (
    Capability,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
)


def _receiver(path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(path)
    sock.settimeout(0.3)
    return sock


def _collect(sock: socket.socket) -> list[bytes]:
    messages: list[bytes] = []
    while True:
        try:
            messages.append(sock.recv(64))
        except TimeoutError:
            return messages


def test_no_notify_socket_is_a_complete_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)

    assert notify.start() is None  # no loop, no socket, no task
    notify.notify_ready()  # also a no-op, and must not raise


def test_ready_and_heartbeats_arrive_and_stop_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _receiver(str(tmp_path / "notify.sock"))
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "notify.sock"))
    monkeypatch.setenv("WATCHDOG_USEC", "200000")  # 0.2s → beats every 0.1s

    async def run() -> asyncio.Task[None]:
        task = notify.start()
        assert task is not None
        await asyncio.sleep(0.35)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return task

    task = asyncio.run(run())

    assert task.done()
    messages = _collect(receiver)
    assert messages[0] == b"READY=1"
    assert messages.count(b"WATCHDOG=1") >= 2


def test_abstract_namespace_addresses_are_translated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert notify._socket_path("@vulcan-test") == "\0vulcan-test"
    assert notify._socket_path("/run/notify.sock") == "/run/notify.sock"

    receiver = _receiver("\0vulcan-test-abstract-delivery")
    monkeypatch.setenv("NOTIFY_SOCKET", "@vulcan-test-abstract-delivery")

    notify.notify_ready()

    assert _collect(receiver) == [b"READY=1"]


def test_undeliverable_notify_socket_fails_startup_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "missing.sock"))

    with pytest.raises(OSError):
        notify.notify_ready()


@pytest.mark.parametrize(
    ("usec", "expected"),
    [("200000", 0.1), ("60000000", 30.0), (None, None), ("garbage", None), ("0", None)],
)
def test_watchdog_interval_parsing(
    monkeypatch: pytest.MonkeyPatch, usec: str | None, expected: float | None
) -> None:
    if usec is None:
        monkeypatch.delenv("WATCHDOG_USEC", raising=False)
    else:
        monkeypatch.setenv("WATCHDOG_USEC", usec)

    assert notify._heartbeat_interval_seconds() == expected


def test_lifespan_announces_ready_and_stops_heartbeats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _receiver(str(tmp_path / "app-notify.sock"))
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "app-notify.sock"))
    monkeypatch.setenv("WATCHDOG_USEC", "100000")  # 0.1s → beats every 0.05s
    config = GatewayConfig(
        schema_version=2,
        providers={
            "test-provider": DeterministicProviderConfig(
                type="deterministic",
                response_text="unused",
            )
        },
        models=(
            ModelConfig(
                id="public-model",
                provider="test-provider",
                provider_model="provider-runtime-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )

    with TestClient(create_app(config), base_url="http://127.0.0.1") as client:
        assert client.get("/healthz").status_code == 200
        time.sleep(0.2)  # let a few beats fire
    # After the client exits, the heartbeat task is cancelled with the app.

    during = _collect(receiver)
    assert b"READY=1" in during
    assert b"WATCHDOG=1" in during
    quiet = _collect(receiver)
    assert b"WATCHDOG=1" not in quiet
