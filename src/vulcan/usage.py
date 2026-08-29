"""Usage counters, with an opt-in durable ledger.

Counts of completed requests and the token counts that upstreams actually
reported, keyed by public alias, configured provider ID, and (when a request
carries one) the caller's optional seat label. In-memory and process-lifetime
by default — nothing is persisted and the counters reset on restart — unless
the operator sets ``[usage] ledger_path``: then every completed request also
appends one JSONL line (never message content, never provider-native model
names) and the counters are rebuilt by replaying that file at startup. No
costs or currencies are computed — Vulcan is not a billing platform.

Token totals are only meaningful alongside ``requests_with_usage``: providers
that omit token counts contribute a request but no tokens, and Vulcan never
invents the difference.
"""

from __future__ import annotations

import json
import logging
import os
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
# Exactly the keys append() writes — no more, no fewer. A forged minimal line
# and a line with smuggled extra keys are both poison.
_LEDGER_KEYS = frozenset({"completion_tokens", "model", "prompt_tokens", "provider", "seat", "ts"})
# Real lines are ~200 bytes; anything past this is poison and is skipped in
# bounded chunks so a single huge unterminated line cannot exhaust memory.
_MAX_LINE_BYTES = 8192
# DoS guard, same shape as budgets._MAX_TRACKED_SEATS: caller-chosen seat
# labels are unbounded, but per-label counter state must not be. Beyond the
# cap a new label is counted (untracked_seat_requests) but not attributed.
_MAX_TRACKED_SEATS = 4096


def _valid_ledger_record(record: object) -> bool:
    """Only lines that would have been legal to WRITE are legal to READ.

    Replayed values reach the HTTP response models, so the ledger file is a
    trust boundary: the exact write schema is enforced — key set, types,
    patterns, and ranges. Anything else becomes ``skipped_lines``, never
    counters or response fields.
    """

    if not isinstance(record, dict) or set(record) != _LEDGER_KEYS:
        return False
    ts = record["ts"]
    model = record["model"]
    provider = record["provider"]
    seat = record["seat"]
    if not (isinstance(ts, int) and not isinstance(ts, bool) and ts >= 0):
        return False
    if not (isinstance(model, str) and _MODEL_RE.fullmatch(model)):
        return False
    if not (isinstance(provider, str) and _PROVIDER_RE.fullmatch(provider)):
        return False
    if seat is not None and not (isinstance(seat, str) and _SEAT_RE.fullmatch(seat)):
        return False
    for key in ("prompt_tokens", "completion_tokens"):
        value = record[key]
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
    # Requests whose seat label arrived after the cardinality cap filled:
    # counted here, deliberately absent from by_seat.
    untracked_seat_requests: int = 0


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
        self._replay_done = False
        try:
            created = not path.exists()
            self._handle: IO[str] = path.open("a", encoding="utf-8")
            if created:
                # Alias/seat/token metadata deserves the same 0600 discipline
                # as the env files; the process umask alone would give 0644.
                path.chmod(0o600)
            try:
                import fcntl

                # Enforce (not merely document) one gateway per ledger file:
                # a second process fails startup loudly instead of interleaving.
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:  # pragma: no cover - non-POSIX platforms
                pass
        except OSError as exc:
            # Fail loud at startup (Prime Directive: never silently fall back).
            raise LedgerError(path, exc.__class__.__name__) from exc

    def replay(self, sink: Callable[[dict[str, object]], None]) -> None:
        """Stream valid history into ``sink``, retaining nothing.

        Memory is bounded regardless of file size: lines stream one at a time,
        each valid record goes to the sink and is dropped, and an oversized
        line (> _MAX_LINE_BYTES) is skipped in bounded chunks without ever
        being buffered whole. Replay is once-only — a second call would
        double-count, so it raises.
        """

        if self._replay_done:
            raise RuntimeError("ledger replay is once-only")
        self._replay_done = True
        if not self._path.exists():  # pragma: no cover - handle open created it
            return
        try:
            with self._path.open("rb") as handle:
                while True:
                    raw_line = handle.readline(_MAX_LINE_BYTES + 1)
                    if not raw_line:
                        break
                    if len(raw_line) > _MAX_LINE_BYTES:
                        # Oversized: count once, drain to newline in bounded
                        # chunks, never hold more than one chunk.
                        self.stats.skipped_lines += 1
                        while raw_line and not raw_line.endswith(b"\n"):
                            raw_line = handle.readline(_MAX_LINE_BYTES + 1)
                        continue
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8").strip())
                    except (UnicodeDecodeError, ValueError):
                        # A torn, foreign, or undecodable line is skipped and
                        # counted, never guessed at.
                        self.stats.skipped_lines += 1
                        continue
                    if not _valid_ledger_record(record):
                        self.stats.skipped_lines += 1
                        continue
                    self.stats.replayed_requests += 1
                    ts = record["ts"]
                    if isinstance(ts, int) and (
                        self.stats.earliest_ts is None or ts < self.stats.earliest_ts
                    ):
                        self.stats.earliest_ts = ts
                    sink(record)
        except OSError as exc:
            raise LedgerError(self._path, exc.__class__.__name__) from exc

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


def truncate_ledger(path: Path, *, cutoff_ts: int) -> dict[str, int]:
    """Rewrite the ledger keeping only valid records at or after ``cutoff_ts``.

    Bounds boot replay time for a long-lived gateway: history before the
    cutoff is gone for good (counters replay only what remains), so this is
    an explicit operator command, never automatic. Fails loud when the file
    is missing or locked — the same flock contract the writer holds, so a
    running gateway refuses the truncation instead of racing it. The rewrite
    is atomic: a 0600 temp file in the same directory, fsynced, then
    os.replace. Kept lines are copied byte-for-byte.
    """

    stats = {"kept": 0, "dropped_old": 0, "dropped_invalid": 0}
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise LedgerError(path, exc.__class__.__name__) from exc
    with handle:
        try:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                # A running gateway holds this lock for its whole lifetime.
                raise LedgerError(
                    path, f"locked by another process ({exc.__class__.__name__})"
                ) from exc
        except ImportError:  # pragma: no cover - non-POSIX platforms
            pass
        tmp_path = path.with_name(path.name + ".truncate-tmp")
        try:
            with tmp_path.open("wb") as tmp:
                tmp_path.chmod(0o600)
                while True:
                    raw_line = handle.readline(_MAX_LINE_BYTES + 1)
                    if not raw_line:
                        break
                    if len(raw_line) > _MAX_LINE_BYTES:
                        # Oversized: count once, drain to newline in bounded
                        # chunks, never hold more than one chunk.
                        stats["dropped_invalid"] += 1
                        while raw_line and not raw_line.endswith(b"\n"):
                            raw_line = handle.readline(_MAX_LINE_BYTES + 1)
                        continue
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line.decode("utf-8").strip())
                    except (UnicodeDecodeError, ValueError):
                        stats["dropped_invalid"] += 1
                        continue
                    if not _valid_ledger_record(record):
                        stats["dropped_invalid"] += 1
                        continue
                    ts = record["ts"]
                    if isinstance(ts, int) and ts >= cutoff_ts:
                        stats["kept"] += 1
                        tmp.write(raw_line if raw_line.endswith(b"\n") else raw_line + b"\n")
                    else:
                        stats["dropped_old"] += 1
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
    return stats


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
    _untracked_seat_requests: int = 0
    _seat_cap_warned: bool = False

    @classmethod
    def with_ledger(cls, ledger: UsageLedger, *, budget_book=None) -> UsageRecorder:
        """A recorder whose counters start from the ledger and append to it."""

        recorder = cls(_ledger=ledger)

        def sink(record: dict[str, object]) -> None:
            # History streams straight into the counters and is never
            # retained: startup memory is O(distinct aliases + providers +
            # seats), not O(ledger size).
            seat = record["seat"]
            prompt = record["prompt_tokens"]
            completion = record["completion_tokens"]
            recorder._count(
                model=str(record["model"]),
                provider=str(record["provider"]),
                prompt_tokens=prompt if isinstance(prompt, int) else None,
                completion_tokens=completion if isinstance(completion, int) else None,
                seat=seat if isinstance(seat, str) else None,
            )
            if budget_book is not None:
                # Replayed spend keeps budgets restart-proof; the book ignores
                # records from previous UTC days.
                tokens = (prompt if isinstance(prompt, int) else 0) + (
                    completion if isinstance(completion, int) else 0
                )
                ts = record["ts"]
                if isinstance(ts, int):
                    budget_book.replay_spend(
                        seat=seat if isinstance(seat, str) else None,
                        provider_id=str(record["provider"]),
                        tokens=tokens,
                        ts=float(ts),
                    )

        ledger.replay(sink)
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
            if seat in self._by_seat or len(self._by_seat) < _MAX_TRACKED_SEATS:
                self._by_seat.setdefault(seat, _Counter()).record(prompt_tokens, completion_tokens)
            else:
                # Cardinality guard tripped: the request still counted in the
                # totals and model/provider views above, and is counted here
                # honestly rather than silently attributed or silently dropped.
                self._untracked_seat_requests += 1
                if not self._seat_cap_warned:
                    self._seat_cap_warned = True
                    logger.warning(
                        "seat_cardinality_capped",
                        extra={"metadata": {"max_tracked_seats": _MAX_TRACKED_SEATS}},
                    )

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
            untracked_seat_requests=self._untracked_seat_requests,
        )
