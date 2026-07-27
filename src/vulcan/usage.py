"""In-memory, process-lifetime usage counters.

Deliberately minimal: counts of completed requests and the token counts that
upstreams actually reported, keyed by public alias and configured provider ID.
Nothing is persisted, no costs or currencies are computed, and the counters
reset when the process restarts — Vulcan is not a billing platform.

Token totals are only meaningful alongside ``requests_with_usage``: providers
that omit token counts contribute a request but no tokens, and Vulcan never
invents the difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
class UsageSnapshot:
    totals: UsageTotals
    by_model: tuple[ModelUsage, ...]
    by_provider: tuple[ProviderUsage, ...]


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

    def record(
        self,
        *,
        model: str,
        provider: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        # Single event loop, no awaits between read and write: plain dict
        # mutation is atomic enough here and needs no lock.
        self._by_model.setdefault((model, provider), _Counter()).record(
            prompt_tokens, completion_tokens
        )
        self._by_provider.setdefault(provider, _Counter()).record(prompt_tokens, completion_tokens)

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
        )
