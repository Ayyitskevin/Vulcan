"""Privacy and failure-safety tests for Vulcan's structured logs."""

from __future__ import annotations

import io
import json
import logging
import sys
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from vulcan.api import create_app
from vulcan.config import (
    Capability,
    DeterministicProviderConfig,
    GatewayConfig,
    ModelConfig,
)
from vulcan.errors import ProviderUnavailableError
from vulcan.observability import REDACTED, SafeJsonFormatter, redact_metadata
from vulcan.providers.base import ProviderChatRequest, ProviderChatResult

PROMPT_SENTINEL = "private-prompt-1cc63a5d"
RESPONSE_SENTINEL = "private-response-b5f71ca2"
EXCEPTION_SENTINEL = "private-exception-ef091f92"


class ExplosiveValue:
    """A formatter must not call arbitrary object string/repr methods."""

    def __str__(self) -> str:
        raise AssertionError("unsafe __str__ call")

    def __repr__(self) -> str:
        raise AssertionError("unsafe __repr__ call")


class SequencedProvider:
    """One success followed by one safely normalized provider failure."""

    provider_type: Literal["deterministic"] = "deterministic"
    provider_id = "test-provider"

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        del request
        self.calls += 1
        if self.calls == 1:
            return ProviderChatResult(content=RESPONSE_SENTINEL, finish_reason="stop")
        try:
            raise RuntimeError(EXCEPTION_SENTINEL)
        except RuntimeError as exc:
            raise ProviderUnavailableError from exc

    async def discover_runtime(self):
        from vulcan.readiness import RuntimeProbe

        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        self.closed = True


def _config() -> GatewayConfig:
    return GatewayConfig(
        schema_version=2,
        providers={
            "test-provider": DeterministicProviderConfig(
                type="deterministic",
                response_text="unused configured response",
            )
        },
        models=(
            ModelConfig(
                id="safe-public-model",
                provider="test-provider",
                provider_model="private-runtime-model",
                capabilities=frozenset({Capability.CHAT}),
            ),
        ),
    )


def test_redact_metadata_recurses_through_mappings_and_sequences() -> None:
    long_safe_value = "x" * 300
    source: dict[str, object] = {
        "request_id": "safe-request-id",
        "prompt": PROMPT_SENTINEL,
        "access_token": EXCEPTION_SENTINEL,
        "refresh-token": EXCEPTION_SENTINEL,
        "system_prompt": EXCEPTION_SENTINEL,
        "raw_response": EXCEPTION_SENTINEL,
        "message_content": EXCEPTION_SENTINEL,
        "nested": {
            "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
            "safe_items": [
                {"provider": "deterministic", "response": RESPONSE_SENTINEL},
                {"x-api-key": "secret-key", "database_password": "secret-password"},
            ],
            "Authorization": "Bearer secret-token",
        },
        "body": {"safe-looking": "still private"},
        "safe_long_value": long_safe_value,
        "opaque": ExplosiveValue(),
    }

    redacted = redact_metadata(source)

    assert redacted == {
        "request_id": "safe-request-id",
        "prompt": REDACTED,
        "access_token": REDACTED,
        "refresh-token": REDACTED,
        "system_prompt": REDACTED,
        "raw_response": REDACTED,
        "message_content": REDACTED,
        "nested": {
            "messages": REDACTED,
            "safe_items": [
                {"provider": "deterministic", "response": REDACTED},
                {"x-api-key": REDACTED, "database_password": REDACTED},
            ],
            "Authorization": REDACTED,
        },
        "body": REDACTED,
        "safe_long_value": long_safe_value[:256],
        "opaque": "ExplosiveValue",
    }
    serialized = json.dumps(redacted)
    for sentinel in (
        PROMPT_SENTINEL,
        RESPONSE_SENTINEL,
        EXCEPTION_SENTINEL,
        "secret-key",
        "secret-password",
    ):
        assert sentinel not in serialized


def test_safe_json_formatter_emits_only_exception_class_and_redacted_metadata() -> None:
    try:
        raise RuntimeError(EXCEPTION_SENTINEL)
    except RuntimeError:
        exception_info = sys.exc_info()

    record = logging.LogRecord(
        name="vulcan.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="provider_failed",
        args=(),
        exc_info=exception_info,
    )
    record.__dict__["metadata"] = {
        "provider": "deterministic",
        "content": RESPONSE_SENTINEL,
        "nested": {"client_secret": EXCEPTION_SENTINEL},
        "opaque": ExplosiveValue(),
    }

    rendered = SafeJsonFormatter().format(record)
    payload = json.loads(rendered)

    assert payload["level"] == "ERROR"
    assert payload["logger"] == "vulcan.test"
    assert payload["event"] == "provider_failed"
    assert payload["provider"] == "deterministic"
    assert payload["content"] == REDACTED
    assert payload["nested"] == {"client_secret": REDACTED}
    assert payload["opaque"] == "ExplosiveValue"
    assert payload["exception_class"] == "RuntimeError"
    assert EXCEPTION_SENTINEL not in rendered
    assert RESPONSE_SENTINEL not in rendered
    assert "Traceback" not in rendered


def test_formatter_never_renders_unapproved_event_arguments_or_reserved_metadata() -> None:
    record = logging.LogRecord(
        name="vulcan.external",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="unsafe event %s",
        args=(PROMPT_SENTINEL,),
        exc_info=None,
    )
    record.__dict__["metadata"] = {
        "event": RESPONSE_SENTINEL,
        "timestamp": RESPONSE_SENTINEL,
        "message_content": RESPONSE_SENTINEL,
    }

    rendered = SafeJsonFormatter().format(record)
    payload = json.loads(rendered)

    assert payload["event"] == "external_log"
    assert payload["timestamp"] != RESPONSE_SENTINEL
    assert PROMPT_SENTINEL not in rendered
    assert RESPONSE_SENTINEL not in rendered


def test_api_logs_operational_counts_without_prompt_response_or_cause(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(SafeJsonFormatter())

    vulcan_logger = logging.getLogger("vulcan")
    monkeypatch.setattr(vulcan_logger, "handlers", [handler])
    monkeypatch.setattr(vulcan_logger, "propagate", False)
    # Logger.setLevel clears the logging cache used by child loggers.
    caplog.set_level(logging.INFO, logger="vulcan")
    for logger_name in ("vulcan.api", "vulcan.gateway"):
        child = logging.getLogger(logger_name)
        monkeypatch.setattr(child, "handlers", [])
        monkeypatch.setattr(child, "level", logging.NOTSET)
        monkeypatch.setattr(child, "propagate", True)

    provider = SequencedProvider()
    with TestClient(
        create_app(
            _config(),
            providers={"test-provider": provider},
            clock=lambda: 100.0,
            id_factory=lambda: "chatcmpl-log-test",
        ),
        base_url="http://127.0.0.1",
    ) as client:
        success = client.post(
            "/v1/chat/completions",
            json={
                "model": "safe-public-model",
                "messages": [{"role": "user", "content": PROMPT_SENTINEL}],
            },
        )
        failure_prompt = f"{PROMPT_SENTINEL}-second-request"
        failure = client.post(
            "/v1/chat/completions",
            json={
                "model": "safe-public-model",
                "messages": [{"role": "user", "content": failure_prompt}],
            },
        )

    assert success.status_code == 200
    assert success.json()["choices"][0]["message"]["content"] == RESPONSE_SENTINEL
    assert failure.status_code == 503
    assert failure.json()["error"]["code"] == "provider_unavailable"
    assert PROMPT_SENTINEL not in failure.text
    assert EXCEPTION_SENTINEL not in failure.text
    assert provider.closed is True

    rendered = stream.getvalue()
    for sentinel in (PROMPT_SENTINEL, failure_prompt, RESPONSE_SENTINEL, EXCEPTION_SENTINEL):
        assert sentinel not in rendered

    records = [json.loads(line) for line in rendered.splitlines()]
    completed = next(record for record in records if record["event"] == "chat_completed")
    failed = next(record for record in records if record["event"] == "chat_failed")
    request_records = [record for record in records if record["event"] == "request_complete"]

    assert completed["provider"] == "test-provider"
    assert completed["provider_type"] == "deterministic"
    assert completed["model"] == "safe-public-model"
    assert completed["turn_count"] == 1
    assert completed["request_id"] == success.headers["X-Request-ID"]
    assert failed["request_id"] == failure.headers["X-Request-ID"]
    assert completed["input_chars"] == len(PROMPT_SENTINEL)
    assert completed["output_chars"] == len(RESPONSE_SENTINEL)
    assert failed["provider"] == "test-provider"
    assert failed["error_code"] == "provider_unavailable"
    assert len(request_records) == 2
    assert {record["request_id"] for record in request_records} == {
        success.headers["X-Request-ID"],
        failure.headers["X-Request-ID"],
    }
    assert all(record["route"] == "/v1/chat/completions" for record in request_records)
