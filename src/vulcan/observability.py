"""Content-safe structured operational logging."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "body",
    "content",
    "cookie",
    "credential",
    "message",
    "password",
    "prompt",
    "response",
    "secret",
    "token",
)
_SAFE_EVENTS = frozenset(
    {
        "chat_completed",
        "chat_failed",
        "embeddings_completed",
        "embeddings_failed",
        "internal_error",
        "provider_failed",
        "readiness_probed",
        "readiness_reused",
        "request_complete",
        "usage_ledger_close_failed",
        "usage_ledger_write_failed",
    }
)
_RESERVED_FIELDS = frozenset({"event", "exception_class", "level", "logger", "timestamp"})


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(fragment in normalized for fragment in _SENSITIVE_FRAGMENTS)


def _redact(value: object, *, key: str | None = None) -> object:
    if key is not None and _is_sensitive_key(key):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact(child, key=str(child_key)) for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(child) for child in value]
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return type(value).__name__


def redact_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {key: _redact(value, key=key) for key, value in metadata.items()}


class SafeJsonFormatter(logging.Formatter):
    """Formatter that emits only an event name and explicitly supplied metadata."""

    def format(self, record: logging.LogRecord) -> str:
        event = (
            record.msg
            if isinstance(record.msg, str) and not record.args and record.msg in _SAFE_EVENTS
            else "external_log"
        )
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": event,
        }
        metadata = getattr(record, "metadata", None)
        if isinstance(metadata, Mapping):
            payload.update(
                {
                    key: value
                    for key, value in redact_metadata(metadata).items()
                    if key not in _RESERVED_FIELDS
                }
            )
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_class"] = record.exc_info[0].__name__
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SafeJsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
