"""Immutable configuration-owned model registry."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from vulcan.config import Capability, ModelConfig
from vulcan.errors import ModelNotFoundError, UnsupportedCapabilityError


@dataclass(frozen=True, slots=True)
class ConfiguredModel:
    id: str
    runtime_name: str
    capabilities: frozenset[Capability]
    description: str | None


class ModelRegistry:
    def __init__(self, models: tuple[ModelConfig, ...]) -> None:
        entries = tuple(
            ConfiguredModel(
                id=model.id,
                runtime_name=model.runtime_name,
                capabilities=model.capabilities,
                description=model.description,
            )
            for model in models
        )
        by_id = {model.id: model for model in entries}
        if len(by_id) != len(entries):
            raise ValueError("configured model IDs must be unique")
        self._models = entries
        self._by_id = MappingProxyType(by_id)

    def list(self) -> tuple[ConfiguredModel, ...]:
        return self._models

    def get(self, model_id: str) -> ConfiguredModel:
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise ModelNotFoundError(model_id) from exc

    def require_capability(self, model_id: str, capability: Capability) -> ConfiguredModel:
        model = self.get(model_id)
        if capability not in model.capabilities:
            raise UnsupportedCapabilityError(capability, model_id)
        return model
