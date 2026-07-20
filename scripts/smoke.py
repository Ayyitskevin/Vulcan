"""Real-process localhost smoke test for Vulcan's deterministic contract."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PROMPT_SENTINEL = "VULCAN_SMOKE_PROMPT_SENTINEL"
RESPONSE_SENTINEL = "VULCAN_SMOKE_RESPONSE_SENTINEL"


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _request(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with opener.open(request, timeout=1.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _wait_for_health(opener: urllib.request.OpenerDirector, base_url: str) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            status, _ = _request(opener, f"{base_url}/healthz")
            if status == 200:
                return
        except (OSError, urllib.error.URLError, TimeoutError):
            time.sleep(0.05)
    raise RuntimeError("Vulcan did not become healthy within 10 seconds")


def _assert_loopback_listener(port: int) -> str:
    result = subprocess.run(
        ["ss", "-ltn"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    expected = f"127.0.0.1:{port}"
    listeners: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[3].rsplit(":", 1)[-1] == str(port):
            listeners.append(fields[3])
    if expected not in listeners:
        raise AssertionError(f"expected a listener on {expected}")
    if any(address != expected for address in listeners):
        raise AssertionError("smoke server was exposed beyond loopback")
    return expected


def main() -> int:
    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    with tempfile.TemporaryDirectory(prefix="vulcan-smoke-") as temp_dir:
        config_path = Path(temp_dir) / "vulcan.toml"
        config_path.write_text(
            f'''schema_version = 1

[server]
host = "127.0.0.1"
port = {port}
log_level = "INFO"

[provider]
kind = "deterministic"
response_text = "{RESPONSE_SENTINEL}"

[[models]]
id = "vulcan-smoke"
runtime_name = "vulcan-smoke"
capabilities = ["chat"]
''',
            encoding="utf-8",
        )
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            [sys.executable, "-m", "vulcan", "serve", "--config", str(config_path)],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = ""
        try:
            _wait_for_health(opener, base_url)
            listener = _assert_loopback_listener(port)
            health_status, health = _request(opener, f"{base_url}/healthz")
            models_status, models = _request(opener, f"{base_url}/v1/models")
            capabilities_status, capabilities = _request(opener, f"{base_url}/v1/capabilities")
            chat_status, chat = _request(
                opener,
                f"{base_url}/v1/chat/completions",
                payload={
                    "model": "vulcan-smoke",
                    "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                },
            )
            assert health_status == models_status == capabilities_status == chat_status == 200
            assert health["provider"] == {"kind": "deterministic", "availability": "unchecked"}
            assert models["discovery"] == {
                "source": "configuration",
                "live": False,
                "availability": "unchecked",
            }
            assert models["data"][0]["id"] == "vulcan-smoke"
            assert capabilities["chat_completions"]["streaming"] is False
            assert chat["choices"][0]["message"]["content"] == RESPONSE_SENTINEL
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                output, _ = process.communicate(timeout=5)
                raise RuntimeError("Vulcan did not stop after SIGINT") from None

        if process.returncode != 0:
            raise RuntimeError(f"Vulcan exited with status {process.returncode}")
        if PROMPT_SENTINEL in output or RESPONSE_SENTINEL in output:
            raise AssertionError("prompt or response content appeared in server logs")

    print(
        json.dumps(
            {
                "listener": listener,
                "provider": "deterministic",
                "statuses": {
                    "healthz": health_status,
                    "models": models_status,
                    "capabilities": capabilities_status,
                    "chat_completions": chat_status,
                },
                "chat_response_verified": True,
                "content_absent_from_logs": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
