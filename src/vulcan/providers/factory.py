"""Exact provider construction with no fallback path."""

from typing import assert_never

import httpx

from vulcan.config import (
    AnthropicProviderConfig,
    DeterministicProviderConfig,
    GatewayConfig,
    OllamaProviderConfig,
    OpenAICompatibleProviderConfig,
    ProviderConfig,
)
from vulcan.providers.anthropic import AnthropicProvider
from vulcan.providers.base import Provider
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.ollama import OllamaProvider
from vulcan.providers.openai_compatible import OpenAICompatibleProvider


def build_provider(
    provider_id: str,
    config: ProviderConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> Provider:
    if isinstance(config, OllamaProviderConfig):
        return OllamaProvider(provider_id, config, client=client)
    if isinstance(config, AnthropicProviderConfig):
        return AnthropicProvider(provider_id, config, client=client)
    if isinstance(config, OpenAICompatibleProviderConfig):
        return OpenAICompatibleProvider(provider_id, config, client=client)
    if isinstance(config, DeterministicProviderConfig):
        return DeterministicProvider(provider_id, config)
    assert_never(config)


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    """Build every configured provider instance keyed by its configured ID."""

    return {
        provider_id: build_provider(provider_id, provider_config)
        for provider_id, provider_config in config.providers.items()
    }
