"""Honest readiness/discovery reconciliation for configured public models.

The registry remains the only source of model *identity*. Live runtime probes
only annotate whether a configured runtime_name is present. Vulcan never invents
public IDs from the runtime list and never claims "available" when a probe fails.

Probe results may be reused for a short, explicit TTL (see
``READINESS_PROBE_TTL_SECONDS``) so health/models/chat share one inventory
without unbounded stale "available" memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vulcan.registry import ConfiguredModel

Availability = Literal["available", "unavailable", "unchecked"]

# Short explicit bound for reusing one readiness probe across health/models/chat.
# After this many seconds a new provider probe runs. No longer-lived cache.
READINESS_PROBE_TTL_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    """Outcome of one provider readiness probe.

    ``live`` is True only when the provider returned a usable runtime name set.
    ``runtime_names`` is None whenever the probe did not successfully list names.
    """

    live: bool
    provider_availability: Availability
    runtime_names: frozenset[str] | None = None


@dataclass(frozen=True, slots=True)
class ModelReadiness:
    model_id: str
    availability: Availability


@dataclass(frozen=True, slots=True)
class DiscoveryReadiness:
    source: Literal["configuration"]
    live: bool
    availability: Availability
    provider_availability: Availability
    models: tuple[ModelReadiness, ...]


def runtime_name_matches(configured_runtime: str, live_names: frozenset[str]) -> bool:
    """Whether a configured provider runtime name is present in a live inventory.

    Matching rule (conservative — never invents public IDs):
    1. Exact equality with a live name, or
    2. The configured name has no ``:`` tag and exactly one live name is that
       base with a single ``:tag`` suffix (e.g. config ``llama3`` matches only
       ``llama3:latest`` when no other ``llama3:*`` is listed).

    Multi-tag collisions (``llama3:latest`` and ``llama3:8b`` both present while
    config says ``llama3``) are **not** treated as available.
    """

    name = configured_runtime.strip()
    if not name:
        return False
    if name in live_names:
        return True
    if ":" in name:
        return False
    tagged = sorted(
        live for live in live_names if live.startswith(f"{name}:") and live.count(":") == 1
    )
    return len(tagged) == 1


def reconcile_configured_models(
    configured: tuple[ConfiguredModel, ...],
    probe: RuntimeProbe,
) -> DiscoveryReadiness:
    """Map configured models onto a probe without inventing IDs or loaded state.

    Rules:
    - Configured public IDs are the only models returned.
    - Successful live list → each model is available iff ``runtime_name_matches``
      its runtime_name against the live set; discovery.live is True.
    - Failed / incomplete probe → models stay unchecked (never fake available).
    - A non-live provider that is still known ready (deterministic) may mark
      models available without claiming a live runtime inventory.
    """

    if probe.live and probe.runtime_names is not None:
        names = probe.runtime_names
        models = tuple(
            ModelReadiness(
                model_id=model.id,
                availability=(
                    "available"
                    if runtime_name_matches(model.runtime_name, names)
                    else "unavailable"
                ),
            )
            for model in configured
        )
        return DiscoveryReadiness(
            source="configuration",
            live=True,
            availability=probe.provider_availability,
            provider_availability=probe.provider_availability,
            models=models,
        )

    # No successful live inventory. Deterministic (available, non-live) is known
    # ready in-process; transport/protocol failures stay fail-loud as unchecked
    # or unavailable at the provider layer and unchecked per model.
    if probe.provider_availability == "available" and not probe.live:
        models = tuple(
            ModelReadiness(model_id=model.id, availability="available") for model in configured
        )
        return DiscoveryReadiness(
            source="configuration",
            live=False,
            availability="available",
            provider_availability="available",
            models=models,
        )

    models = tuple(
        ModelReadiness(model_id=model.id, availability="unchecked") for model in configured
    )
    return DiscoveryReadiness(
        source="configuration",
        live=False,
        availability=probe.provider_availability,
        provider_availability=probe.provider_availability,
        models=models,
    )
