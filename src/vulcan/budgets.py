"""Per-seat daily budgets for hosted providers.

Guardrail against accidents, not adversaries: seats are voluntary labels, so
this cannot stop a caller that lies about its name — it stops a well-meaning
seat in a retry loop from burning hosted credit overnight. Local providers are
never budgeted (they cost electricity, not credit), Vulcan never converts
tokens to money, and over-budget requests are refused loudly with the reset
time — the gateway never silently reroutes to another provider (one alias, one
provider, always).

Windows are UTC days. Spend survives restarts because it is recomputed from
the durable usage ledger at boot; a single in-flight request may overshoot its
remaining budget (token counts are only known after completion), and providers
that omit token counts under-meter — which is what the request cap is for.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vulcan.errors import BudgetExhaustedError, BudgetUnconfiguredError, SeatRequiredError

_DAY_SECONDS = 86_400
# DoS guard: a default budget admits unlimited caller-chosen labels; state per
# seat is tiny but must still be bounded. Real fleets have single-digit seats.
_MAX_TRACKED_SEATS = 4096


@dataclass(frozen=True, slots=True)
class SeatLimits:
    hosted_tokens_per_day: int | None
    hosted_requests_per_day: int | None


@dataclass(frozen=True, slots=True)
class SeatBudgetState:
    """One row of operator-visible budget state for /v1/usage."""

    seat: str
    hosted_tokens_per_day: int | None
    hosted_requests_per_day: int | None
    tokens_today: int
    requests_today: int
    window_resets_at: int


class BudgetBook:
    """Windowed per-seat spend against configured daily limits.

    Single event loop, no awaits between read and write — plain dict state
    needs no lock, same as the usage counters.
    """

    def __init__(
        self,
        *,
        limits: Mapping[str, SeatLimits],
        default: SeatLimits | None,
        hosted_providers: frozenset[str],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._limits = dict(limits)
        self._default = default
        self._hosted = hosted_providers
        self._clock = clock
        self._day = int(clock()) // _DAY_SECONDS
        self._tokens: dict[str, int] = {}
        self._requests: dict[str, int] = {}

    def _roll(self) -> None:
        day = int(self._clock()) // _DAY_SECONDS
        if day != self._day:
            self._day = day
            self._tokens.clear()
            self._requests.clear()

    def _resets_at(self) -> int:
        return (self._day + 1) * _DAY_SECONDS

    def _limits_for(self, seat: str) -> SeatLimits | None:
        return self._limits.get(seat, self._default)

    def check(self, *, seat: str | None, provider_id: str) -> int | None:
        """Pre-flight gate AND atomic request-slot reservation.

        Local providers pass without reserving (returns None). For hosted:
        the seat must exist, resolve to a budget (fail-closed), and have
        headroom — then the request slot is reserved in the same synchronous
        call, so N concurrent requests can never all pass the last slot of a
        request cap. Returns the reservation's UTC day; the caller MUST later
        call settle() (success) or release() (failure) with that value — the
        day makes midnight-straddling requests account identically live and
        on ledger replay (both attribute to the completion day).

        Token headroom is checked but cannot be reserved (counts arrive after
        completion): token overshoot is bounded by the number of concurrently
        in-flight requests per seat, which the request cap bounds in turn
        (config REQUIRES a request cap on every entry). Documented, not
        hidden.
        """

        if provider_id not in self._hosted:
            return None
        if seat is None:
            raise SeatRequiredError(provider_id)
        limits = self._limits_for(seat)
        if limits is None:
            raise BudgetUnconfiguredError(seat)
        self._roll()
        if (
            seat not in self._limits
            and seat not in self._requests
            and len(self._requests) >= _MAX_TRACKED_SEATS
        ):
            # Cardinality guard: an unbounded stream of fresh labels under a
            # default budget must not grow state without limit. Seats with
            # their own configured entry are exempt — they are bounded by the
            # config itself, and a label flood must never squeeze them out.
            raise BudgetUnconfiguredError(seat)
        resets_at = self._resets_at()
        if (
            limits.hosted_tokens_per_day is not None
            and self._tokens.get(seat, 0) >= limits.hosted_tokens_per_day
        ):
            raise BudgetExhaustedError(seat, window_resets_at=resets_at)
        if (
            limits.hosted_requests_per_day is not None
            and self._requests.get(seat, 0) >= limits.hosted_requests_per_day
        ):
            raise BudgetExhaustedError(seat, window_resets_at=resets_at)
        # No await between the checks above and this reservation: atomic on
        # the single event loop.
        self._requests[seat] = self._requests.get(seat, 0) + 1
        return self._day

    def settle(
        self,
        *,
        seat: str | None,
        provider_id: str,
        tokens: int | None,
        reservation_day: int,
    ) -> None:
        """Complete a reservation: spend lands in the completion day.

        If UTC midnight passed since check(), the roll cleared the reserved
        slot — re-count the request in the completion day, so live accounting
        matches what a later ledger replay (keyed on the completion
        timestamp) will reconstruct.
        """

        if provider_id not in self._hosted or seat is None:
            return
        self._roll()
        if reservation_day != self._day:
            self._requests[seat] = self._requests.get(seat, 0) + 1
        self._tokens[seat] = self._tokens.get(seat, 0) + (tokens or 0)

    def release(self, *, seat: str | None, provider_id: str, reservation_day: int) -> None:
        """Return a reserved request slot after failure or cancellation.

        Failures are never usage and never budget spend. After a UTC rollover
        the slot no longer exists (the roll cleared it) — nothing to return.
        """

        if provider_id not in self._hosted or seat is None:
            return
        self._roll()
        if reservation_day != self._day:
            return
        current = self._requests.get(seat, 0)
        if current > 0:
            self._requests[seat] = current - 1

    def replay_spend(
        self,
        *,
        seat: str | None,
        provider_id: str,
        tokens: int | None,
        ts: float,
    ) -> None:
        """Count one completed request from the ledger at boot.

        Records from a previous UTC day are ignored — yesterday's spend never
        counts against today's window.
        """

        if provider_id not in self._hosted or seat is None:
            return
        self._roll()
        if int(ts) // _DAY_SECONDS != self._day:
            return
        if (
            seat not in self._limits
            and seat not in self._requests
            and len(self._requests) >= _MAX_TRACKED_SEATS
        ):
            # Same cardinality guard as check(): a ledger crafted (or written
            # before this guard existed) with unbounded distinct labels must
            # not repopulate unbounded state at boot. Configured seats are
            # exempt (bounded by the config, and skipping them would
            # under-meter real spend); skipping unconfigured labels is the
            # safe direction for a refusal gate.
            return
        self._tokens[seat] = self._tokens.get(seat, 0) + (tokens or 0)
        self._requests[seat] = self._requests.get(seat, 0) + 1

    def snapshot(self) -> tuple[SeatBudgetState, ...]:
        """Every configured seat plus any seat with spend today, sorted."""

        self._roll()
        seats = sorted(set(self._limits) | set(self._tokens) | set(self._requests))
        resets_at = self._resets_at()
        rows = []
        for seat in seats:
            limits = self._limits_for(seat)
            rows.append(
                SeatBudgetState(
                    seat=seat,
                    hosted_tokens_per_day=limits.hosted_tokens_per_day if limits else None,
                    hosted_requests_per_day=limits.hosted_requests_per_day if limits else None,
                    tokens_today=self._tokens.get(seat, 0),
                    requests_today=self._requests.get(seat, 0),
                    window_resets_at=resets_at,
                )
            )
        return tuple(rows)
