"""In-memory, process-lifetime usage counters.

Deliberately minimal: counts of completed requests and the token counts that
upstreams actually reported, keyed by public alias, configured provider ID, and (when a request
carries one) the caller's optional seat label.
Nothing is persisted, no costs or currencies are computed, and the counters
reset when the process restarts — Vulcan is not a billing platform.

Token totals are only meaningful alongside ``requests_with_usage``: providers
that omit token counts contribute a request but no tokens, and Vulcan never
invents the difference.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from vulcan.config import PROVIDER_ID_PATTERN, PUBLIC_MODEL_PATTERN, SEAT_PATTERN

logger = logging.getLogger("vulcan.usage")

_MODEL_RE = re.compile(PUBLIC_MODEL_PATTERN)
_PROVIDER_RE = re.compile(PROVIDER_ID_PATTERN)
_SEAT_RE = re.compile(SEAT_PATTERN)
_MAX_TOKENS = 10**12


def _valid_ledger_record(record: object) -> bool:
    """Only lines that would have been legal to WRITE are legal to READ.

    Replayed values reach the HTTP response models, so the ledger file is a
    trust boundary: negative counts, pattern-violating names, or injected
    text must become ``skipped_lines``, never counters or response fields.
    """

    if not isinstance(record, dict):
        return False
    ts = record.get("ts")
    model = record.get("model")
    provider = record.get("provider")
    seat = record.get("seat")
    if not (isinstance(ts, int) and not isinstance(ts, bool) and ts >= 0):
        return False
    if not (isinstance(model, str) and _MODEL_RE.fullmatch(model)):
        return False
    if not (isinstance(provider, str) and _PROVIDER_RE.fullmatch(provider)):
        return False
    if seat is not None and not (isinstance(seat, str) and _SEAT_RE.fullmatch(seat)):
        return False
    for key in ("prompt_tokens", "completion_tokens"):
        value = record.get(key)
        if value is None:
            continue
        if not (
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_TOKENS
        ):
            return False
    return True


_MODEL_RE = re.compile(PUBLIC_MODEL_PATTERN)
_PROVIDER_RE = re.compile(PROVIDER_ID_PATTERN)
_SEAT_RE = re.compile(SEAT_PATTERN)
# Ledger entries feed HTTP response models on replay, so the file is a trust
# boundary: only lines that would have been legal to WRITE are legal to READ.
_MAX_TOKENS = 10**12


def _valid_ledger_record(record: object) -> bool:
    if not isinstance(record, dict):
        return False
    ts = record.get("ts")
    model = record.get("model")
    provider = record.get("provider")
    seat = record.get("seat")
    if not (isinstance(ts, int) and not isinstance(ts, bool) and ts >= 0):
        return False
    if not (isinstance(model, str) and _MODEL_RE.fullmatch(model)):
        return False
    if not (isinstance(provider, str) and _PROVIDER_RE.fullmatch(provider)):
        return False
    if seat is not None and not (isinstance(seat, str) and _SEAT_RE.fullmatch(seat)):
        return False
    for key in ("prompt_tokens", "completion_tokens"):
        value = record.get(key)
        if value is None:
            continue
        if not (
            isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _MAX_TOKENS
        ):
            return False
    return True


class LedgerError(Exception):
    """The ledger file cannot be opened or replayed. Startup must fail loud."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"usage ledger at {path}: {reason}")


@dataclass(slots=True)
class _Counter:
    requests: int = 0
    requests_with_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def record(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        self.requests += 1
        if prompt_tokens is None and completion_tokens is None:
            return
        self.requests_with_usage += 1
        self.prompt_tokens += prompt_tokens or 0
        self.completion_tokens += completion_tokens or 0


@dataclass(frozen=True, slots=True)
class UsageTotals:
    requests: int
    requests_with_usage: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ModelUsage:
    model: str
    provider: str
    totals: UsageTotals


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    provider: str
    totals: UsageTotals


@dataclass(frozen=True, slots=True)
class SeatUsage:
    seat: str
    totals: UsageTotals


@dataclass(slots=True)
class LedgerStats:
    """Operator-visible honesty counters for the durable ledger."""

    replayed_requests: int = 0
    skipped_lines: int = 0
    write_failures: int = 0
    earliest_ts: int | None = None


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    totals: UsageTotals
    by_model: tuple[ModelUsage, ...]
    by_provider: tuple[ProviderUsage, ...]
    by_seat: tuple[SeatUsage, ...]
    ledger: LedgerStats | None = None


class UsageLedger:
    """Append-only JSONL ledger of completed requests.

    One line per completed request: ``ts``, ``model`` (public alias),
    ``provider`` (configured ID), optional ``seat``, and the token counts the
    upstream actually reported (null when it reported none). Never message
    content, never native model names. Writes are flushed per line but not
    fsynced — a hard power cut may lose the tail, and the replay counters say
    so honestly rather than inventing the difference.
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self._path = path
        self._clock = clock
        self.stats = LedgerStats()
        try:
            self._replay_existing()
            self._handle: IO[str] = path.open("a", encoding="utf-8")
        except OSError as exc:
            # Fail loud at startup (Prime Directive: never silently fall back).
            raise LedgerError(path, exc.__class__.__name__) from exc

    @property
    def replayed(self) -> tuple[dict[str, object], ...]:
        return tuple(self._replayed)

    def _replay_existing(self) -> None:
        self._replayed: list[dict[str, object]] = []
        if not self._path.exists():
            return
        # Bytes first, decoded per line: one undecodable line is one skipped
        # line, never a startup crash (UnicodeDecodeError is not an OSError).
        for raw_line in self._path.read_bytes().split(b"\n"):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8").strip())
            except (UnicodeDecodeError, ValueError):
                # A torn, foreign, or undecodable line is skipped and counted,
                # never guessed at.
                self.stats.skipped_lines += 1
                continue
            if not _valid_ledger_record(record):
                self.stats.skipped_lines += 1
                continue
            self._replayed.append(record)
            self.stats.replayed_requests += 1
            ts = record["ts"]
            if isinstance(ts, int) and (
                self.stats.earliest_ts is None or ts < self.stats.earliest_ts
            ):
                self.stats.earliest_ts = ts

    def append(
        self,
        *,
        model: str,
        provider: str,
        seat: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> None:
        record = {
            "completion_tokens": completion_tokens,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "provider": provider,
            "seat": seat,
            "ts": int(self._clock()),
        }
        try:
            self._handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            self._handle.flush()
        # ValueError covers writes on a closed handle, which is not an OSError.
        except (OSError, ValueError):
            # Loud and visible: counted in /v1/usage and logged with a fixed
            # event name, no content. The request itself already succeeded.
            self.stats.write_failures += 1
            logger.error("usage_ledger_write_failed")

    def close(self) -> None:
        try:
            self._handle.close()
        except (OSError, ValueError):  # pragma: no cover - no recourse
            logger.error("usage_ledger_close_failed")


def _totals(counter: _Counter) -> UsageTotals:
    return UsageTotals(
        requests=counter.requests,
        requests_with_usage=counter.requests_with_usage,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        total_tokens=counter.prompt_tokens + counter.completion_tokens,
    )


@dataclass(slots=True)
class UsageRecorder:
    """Counts completed requests only; failures are never counted as usage."""

    _by_model: dict[tuple[str, str], _Counter] = field(default_factory=dict)
    _by_provider: dict[str, _Counter] = field(default_factory=dict)
    _by_seat: dict[str, _Counter] = field(default_factory=dict)
    _ledger: UsageLedger | None = None

    @classmethod
    def with_ledger(cls, ledger: UsageLedger) -> UsageRecorder:
        """A recorder whose counters start from the ledger and append to it."""

        recorder = cls(_ledger=ledger)
        for record in ledger.replayed:
            seat = record.get("seat")
            prompt = record.get("prompt_tokens")
            completion = record.get("completion_tokens")
            recorder._count(
                model=str(record["model"]),
                provider=str(record["provider"]),
                prompt_tokens=prompt if isinstance(prompt, int) else None,
                completion_tokens=completion if isinstance(completion, int) else None,
                seat=seat if isinstance(seat, str) else None,
            )
        return recorder

    def _count(
        self,
        *,
        model: str,
        provider: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        seat: str | None,
    ) -> None:
        # Single event loop, no awaits between read and write: plain dict
        # mutation is atomic enough here and needs no lock.
        self._by_model.setdefault((model, provider), _Counter()).record(
            prompt_tokens, completion_tokens
        )
        self._by_provider.setdefault(provider, _Counter()).record(prompt_tokens, completion_tokens)
        # Attribution is optional: unlabeled requests still count in the totals
        # and the model/provider views, they just never appear under a seat.
        if seat is not None:
            self._by_seat.setdefault(seat, _Counter()).record(prompt_tokens, completion_tokens)

    def record(
        self,
        *,
        model: str,
        provider: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        seat: str | None = None,
    ) -> None:
        self._count(
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            seat=seat,
        )
        if self._ledger is not None:
            self._ledger.append(
                model=model,
                provider=provider,
                seat=seat,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    def snapshot(self) -> UsageSnapshot:
        """A stable, sorted view of the counters at this moment."""

        overall = _Counter()
        for counter in self._by_provider.values():
            overall.requests += counter.requests
            overall.requests_with_usage += counter.requests_with_usage
            overall.prompt_tokens += counter.prompt_tokens
            overall.completion_tokens += counter.completion_tokens

        return UsageSnapshot(
            totals=_totals(overall),
            by_model=tuple(
                ModelUsage(model=model, provider=provider, totals=_totals(counter))
                for (model, provider), counter in sorted(self._by_model.items())
            ),
            by_provider=tuple(
                ProviderUsage(provider=provider, totals=_totals(counter))
                for provider, counter in sorted(self._by_provider.items())
            ),
            by_seat=tuple(
                SeatUsage(seat=seat, totals=_totals(counter))
                for seat, counter in sorted(self._by_seat.items())
            ),
            ledger=self._ledger.stats if self._ledger is not None else None,
        )
