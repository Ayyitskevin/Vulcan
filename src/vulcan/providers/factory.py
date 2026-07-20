"""Exact provider selection with no fallback path."""

from typing import assert_never

import httpx

from vulcan.config import DeterministicProviderConfig, OllamaProviderConfig, ProviderConfig
from vulcan.providers.base import Provider
from vulcan.providers.deterministic import DeterministicProvider
from vulcan.providers.ollama import OllamaProvider


def build_provider(
    config: ProviderConfig,
    *,
    ollama_client: httpx.AsyncClient | None = None,
) -> Provider:
    if isinstance(config, OllamaProviderConfig):
        return OllamaProvider(config, client=ollama_client)
    if isinstance(config, DeterministicProviderConfig):
        return DeterministicProvider(config)
    assert_never(config)
