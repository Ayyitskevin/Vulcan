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


def _stream(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
) -> tuple[int, str, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=5.0) as response:
        return (
            response.status,
            response.headers.get("Content-Type", ""),
            response.read().decode("utf-8"),
        )


def _sse_frames(body: str) -> list[str]:
    return [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]


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
            f'''schema_version = 2

[server]
host = "127.0.0.1"
port = {port}
log_level = "INFO"

[providers.smoke]
type = "deterministic"
response_text = "{RESPONSE_SENTINEL}"

[[models]]
id = "vulcan-smoke"
provider = "smoke"
provider_model = "vulcan-smoke"
capabilities = ["chat", "embeddings"]
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
            model_status, model = _request(opener, f"{base_url}/v1/models/vulcan-smoke")
            capabilities_status, capabilities = _request(opener, f"{base_url}/v1/capabilities")
            chat_status, chat = _request(
                opener,
                f"{base_url}/v1/chat/completions",
                payload={
                    "model": "vulcan-smoke",
                    "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                },
            )
            statuses = (health_status, models_status, model_status, capabilities_status)
            embeddings_status, embeddings = _request(
                opener,
                f"{base_url}/v1/embeddings",
                payload={"model": "vulcan-smoke", "input": [PROMPT_SENTINEL, "second input"]},
            )
            stream_status, stream_content_type, stream_body = _stream(
                opener,
                f"{base_url}/v1/chat/completions",
                {
                    "model": "vulcan-smoke",
                    "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
                    "stream": True,
                },
            )
            assert statuses == (200, 200, 200, 200) and chat_status == 200
            assert embeddings_status == 200
            assert embeddings["object"] == "list"
            assert embeddings["provider"] == "smoke"
            assert [record["index"] for record in embeddings["data"]] == [0, 1]
            assert all(len(record["embedding"]) == 8 for record in embeddings["data"])
            assert embeddings["usage"] is None
            assert stream_status == 200
            assert stream_content_type.startswith("text/event-stream")
            stream_frames = _sse_frames(stream_body)
            assert stream_frames[-1] == "[DONE]"
            stream_chunks = [json.loads(frame) for frame in stream_frames[:-1]]
            assert [chunk["choices"][0]["delta"] for chunk in stream_chunks] == [
                {"role": "assistant"},
                {"content": RESPONSE_SENTINEL},
                {},
            ]
            assert [chunk["choices"][0]["finish_reason"] for chunk in stream_chunks] == [
                None,
                None,
                "stop",
            ]
            assert all(chunk["object"] == "chat.completion.chunk" for chunk in stream_chunks)
            assert all(chunk["provider"] == "smoke" for chunk in stream_chunks)
            assert health["providers"] == [
                {"id": "smoke", "type": "deterministic", "availability": "available"}
            ]
            assert models["discovery"] == {"source": "configuration"}
            assert models["data"][0]["id"] == "vulcan-smoke"
            assert models["data"][0]["provider"] == "smoke"
            assert models["data"][0]["provider_type"] == "deterministic"
            assert models["data"][0]["availability"] == "available"
            assert model["id"] == "vulcan-smoke"
            assert model["provider"] == "smoke"
            assert model["availability"] == "available"
            assert capabilities["chat_completions"]["streaming"] is True
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
                "provider": "smoke",
                "statuses": {
                    "healthz": health_status,
                    "models": models_status,
                    "model_retrieve": model_status,
                    "capabilities": capabilities_status,
                    "chat_completions": chat_status,
                    "chat_completions_stream": stream_status,
                    "embeddings": embeddings_status,
                },
                "embeddings_verified": True,
                "stream_verified": True,
                "chat_response_verified": True,
                "content_absent_from_logs": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
