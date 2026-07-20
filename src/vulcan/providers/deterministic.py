"""Explicit no-I/O provider for tests and local contract checks."""

from typing import Literal

from vulcan.config import DeterministicProviderConfig
from vulcan.providers.base import ProviderChatRequest, ProviderChatResult
from vulcan.readiness import RuntimeProbe


class DeterministicProvider:
    kind: Literal["deterministic"] = "deterministic"

    def __init__(self, config: DeterministicProviderConfig) -> None:
        self._response_text = config.response_text

    async def chat(self, request: ProviderChatRequest) -> ProviderChatResult:
        del request
        return ProviderChatResult(content=self._response_text, finish_reason="stop")

    async def discover_runtime(self) -> RuntimeProbe:
        # In-process and non-network: known ready, but there is no external
        # runtime inventory to reconcile against (live stays false).
        return RuntimeProbe(live=False, provider_availability="available", runtime_names=None)

    async def aclose(self) -> None:
        return None
