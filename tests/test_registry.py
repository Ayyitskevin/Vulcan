from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from vulcan.config import Capability, ModelConfig
from vulcan.errors import ModelNotFoundError, UnsupportedCapabilityError
from vulcan.registry import ConfiguredModel, ModelRegistry


def _model(
    model_id: str,
    runtime_name: str,
    *capabilities: Capability,
) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        runtime_name=runtime_name,
        capabilities=frozenset(capabilities),
        description=f"Description for {model_id}",
    )


def test_registry_preserves_configuration_order_and_public_runtime_mapping() -> None:
    chat = _model("public-chat", "private/runtime-chat", Capability.CHAT)
    embed = _model("public-embed", "private/runtime-embed", Capability.EMBEDDINGS)
    registry = ModelRegistry((chat, embed))

    listed = registry.list()

    assert isinstance(listed, tuple)
    assert [model.id for model in listed] == ["public-chat", "public-embed"]
    assert [model.runtime_name for model in listed] == [
        "private/runtime-chat",
        "private/runtime-embed",
    ]
    assert registry.get("public-chat") is listed[0]
    assert registry.get("public-embed") is listed[1]


def test_registry_lookup_is_exact_and_does_not_accept_runtime_name() -> None:
    registry = ModelRegistry((_model("Public-Chat", "runtime-chat", Capability.CHAT),))

    for unknown in ("public-chat", "runtime-chat", "missing"):
        with pytest.raises(ModelNotFoundError) as raised:
            registry.get(unknown)
        assert raised.value.code == "model_not_found"
        assert raised.value.status_code == 404
        assert raised.value.details == {"model": unknown}


def test_registry_requires_declared_capability() -> None:
    model = _model("multi-model", "runtime-model", Capability.CHAT, Capability.EMBEDDINGS)
    registry = ModelRegistry((model,))

    assert registry.require_capability("multi-model", Capability.CHAT) is registry.list()[0]
    assert registry.require_capability("multi-model", Capability.EMBEDDINGS) is registry.list()[0]


def test_registry_rejects_undeclared_capability_with_stable_details() -> None:
    registry = ModelRegistry((_model("embedding-only", "runtime-embed", Capability.EMBEDDINGS),))

    with pytest.raises(UnsupportedCapabilityError) as raised:
        registry.require_capability("embedding-only", Capability.CHAT)

    assert raised.value.code == "unsupported_capability"
    assert raised.value.status_code == 422
    assert raised.value.retryable is False
    assert raised.value.details == {"capability": "chat", "model": "embedding-only"}


def test_registry_checks_model_existence_before_capability() -> None:
    registry = ModelRegistry((_model("known", "runtime", Capability.EMBEDDINGS),))

    with pytest.raises(ModelNotFoundError):
        registry.require_capability("unknown", Capability.CHAT)


def test_registry_rejects_duplicate_ids_even_without_gateway_config() -> None:
    first = _model("duplicate", "runtime-one", Capability.CHAT)
    second = _model("duplicate", "runtime-two", Capability.EMBEDDINGS)

    with pytest.raises(ValueError, match="configured model IDs must be unique"):
        ModelRegistry((first, second))


def test_empty_registry_is_stable_but_has_no_lookup_results() -> None:
    registry = ModelRegistry(())

    assert registry.list() == ()
    with pytest.raises(ModelNotFoundError):
        registry.get("anything")


def test_registry_public_results_are_deeply_immutable() -> None:
    registry = ModelRegistry((_model("chat", "runtime", Capability.CHAT),))
    listed = registry.list()
    entry = listed[0]

    assert isinstance(entry, ConfiguredModel)
    assert isinstance(entry.capabilities, frozenset)
    assert registry.list() is listed

    with pytest.raises(FrozenInstanceError):
        cast(Any, entry).id = "mutated"

    mutable_capabilities = cast(Any, entry.capabilities)
    with pytest.raises(AttributeError):
        mutable_capabilities.add(Capability.EMBEDDINGS)

    external_copy = list(listed)
    external_copy.clear()
    assert [model.id for model in registry.list()] == ["chat"]
