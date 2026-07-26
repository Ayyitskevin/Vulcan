"""Honest readiness/discovery reconciliation for configured public models.

The registry remains the only source of model *identity*. Live runtime probes
only annotate whether a configured provider_model is present. Vulcan never
invents public IDs from a runtime list and never claims "available" when a
probe fails. Hosted providers are never probed (that would call authenticated,
often billable endpoints just to render metadata) and stay "unchecked" until a
real request uses them.

Probe results may be reused for a short, explicit TTL (see
``READINESS_PROBE_TTL_SECONDS``) so health/models/chat share one inventory
without unbounded stale "available" memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from vulcan.registry import ConfiguredModel

Availability = Literal["available", "unavailable", "unchecked"]

# Default short bound for reusing one readiness probe across health/models/chat.
# Overridable via ``[readiness].probe_ttl_seconds`` (0..60). No longer-lived cache.
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
class ProviderReadiness:
    """One configured provider instance's probe outcome as safe metadata."""

    provider_id: str
    provider_type: str
    live: bool
    availability: Availability


@dataclass(frozen=True, slots=True)
class ModelReadiness:
    model_id: str
    availability: Availability


@dataclass(frozen=True, slots=True)
class GatewayReadiness:
    """Aggregate readiness across every configured provider and model."""

    source: Literal["configuration"]
    providers: tuple[ProviderReadiness, ...]
    models: tuple[ModelReadiness, ...]

    def model_availability(self, model_id: str) -> Availability:
        for item in self.models:
            if item.model_id == model_id:
                return item.availability
        return "unchecked"


def runtime_name_matches(configured_runtime: str, live_names: frozenset[str]) -> bool:
    """Whether a configured provider model name is present in a live inventory.

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


def _model_availability(model: ConfiguredModel, probe: RuntimeProbe) -> Availability:
    """One model's annotation from its own provider's probe.

    Rules (unchanged from v1, applied per provider):
    - Successful live list → available iff ``runtime_name_matches`` the
      configured provider_model; otherwise unavailable.
    - Non-live but known-ready provider (deterministic) → available.
    - Failed, incomplete, or deliberately skipped probe → unchecked.
    """

    if probe.live and probe.runtime_names is not None:
        if runtime_name_matches(model.provider_model, probe.runtime_names):
            return "available"
        return "unavailable"
    if probe.provider_availability == "available":
        return "available"
    return "unchecked"


def reconcile_configured_models(
    configured: tuple[ConfiguredModel, ...],
    probes: Mapping[str, RuntimeProbe],
    provider_types: Mapping[str, str],
) -> GatewayReadiness:
    """Map configured models onto per-provider probes without inventing state.

    ``probes`` and ``provider_types`` are keyed by configured provider ID and
    must cover every provider referenced by ``configured`` (guaranteed by
    config validation).
    """

    providers = tuple(
        ProviderReadiness(
            provider_id=provider_id,
            provider_type=provider_types[provider_id],
            live=probe.live,
            availability=probe.provider_availability,
        )
        for provider_id, probe in probes.items()
    )
    models = tuple(
        ModelReadiness(
            model_id=model.id,
            availability=_model_availability(model, probes[model.provider_id]),
        )
        for model in configured
    )
    return GatewayReadiness(source="configuration", providers=providers, models=models)
